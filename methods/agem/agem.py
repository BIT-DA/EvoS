import os
import time
import datetime
import numpy as np
import torch

from .buffer import Buffer
from ..base_trainer import BaseTrainer
from ..utils import prepare_data, forward_pass, MetricLogger
from ..dataloaders import FastDataLoader

class AGEM(BaseTrainer):
    """
    Averaged Gradient Episodic Memory (A-GEM)

    Code adapted from https://github.com/aimagelab/mammoth.

    Original Paper:

        @article{chaudhry2018efficient,
        title={Efficient lifelong learning with a-gem},
        author={Chaudhry, Arslan and Ranzato, Marc'Aurelio and Rohrbach, Marcus and Elhoseiny, Mohamed},
        journal={arXiv preprint arXiv:1812.00420},
        year={2018}
        }
    """
    def __init__(self, args, logger, dataset, network, criterion, optimizer, scheduler):
        super().__init__(args, logger, dataset, network, criterion, optimizer, scheduler)

        self.buffer = Buffer(self.args.buffer_size, self._device())
        self.grad_dims = []
        for param in self.network.parameters():
            self.grad_dims.append(param.data.numel())
        self.grad_xy = torch.Tensor(np.sum(self.grad_dims)).to(self._device())
        self.grad_er = torch.Tensor(np.sum(self.grad_dims)).to(self._device())

    def __str__(self):
        return f'AGEM-buffer_size={self.args.buffer_size}-{self.base_trainer_str}'

    def _device(self):
        return next(self.network.parameters()).device

    def end_task(self, dataloader):
        sample = next(iter(dataloader))
        cur_x, cur_y = sample
        cur_x, cur_y = prepare_data(cur_x, cur_y, str(self.train_dataset))
        self.buffer.add_data(
            examples=cur_x,
            labels=cur_y
        )

    def train_step(self, dataloader):
        self.logger.info("-------------------start training on timestamp {}-------------------".format(self.train_dataset.current_time))
        self.network.train()
        loss_all = []
        meters = MetricLogger(delimiter="  ")
        end = time.time()
        self.logger.info("self.train_dataset.len = {} x {} = {} samples".format(self.train_dataset.__len__() // self.args.mini_batch_size, self.args.mini_batch_size, self.train_dataset.__len__()))
        stop_iters = self.args.epochs * (self.train_dataset.__len__() // self.args.mini_batch_size) - 1
        for step, (x, y) in enumerate(dataloader):
            x, y = prepare_data(x, y, str(self.train_dataset))

            loss, logits, y = forward_pass(x, y, self.train_dataset, self.network, self.criterion, self.lisa, self.mixup,
                                           self.cut_mix, self.mix_alpha)
            loss_all.append(loss.item())
            loss.backward()

            if not self.buffer.is_empty():
                store_grad(self.network.parameters, self.grad_xy, self.grad_dims)

                buf_data = self.buffer.get_data(self.mini_batch_size, transform=None)
                if len(buf_data) > 2:
                    buf_inputs = [buf_data[0], buf_data[1]]
                    buf_labels = buf_data[2]
                else:
                    buf_inputs, buf_labels = buf_data
                buf_inputs, buf_labels = prepare_data(buf_inputs, buf_labels, str(self.train_dataset))
                self.network.zero_grad()
                penalty, buff_outputs, buf_labels = forward_pass(buf_inputs, buf_labels, self.train_dataset, self.network,
                                                                 self.criterion, self.lisa, self.mixup, self.cut_mix, self.mix_alpha)
                penalty.backward()
                store_grad(self.network.parameters, self.grad_er, self.grad_dims)

                dot_prod = torch.dot(self.grad_xy, self.grad_er)
                if dot_prod.item() < 0:
                    g_tilde = project(gxy=self.grad_xy, ger=self.grad_er)
                    overwrite_grad(self.network.parameters, g_tilde, self.grad_dims)
                else:
                    overwrite_grad(self.network.parameters, self.grad_xy, self.grad_dims)

            self.optimizer.step()

            if step == stop_iters:
                if self.scheduler is not None:
                    self.scheduler.step()
                self.end_task(dataloader)
                break
            # -----------------print log infromation------------
            batch_time = time.time() - end
            end = time.time()
            meters.update(time=batch_time)
            eta_seconds = meters.time.global_avg * (stop_iters - step)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
            meters.update(loss=(loss).item())
            if step % self.args.print_freq == 0:
                self.logger.info(
                    meters.delimiter.join(
                        [
                            "eta: {eta}",
                            "timestamp: {timestamp}",
                            f"[iter: {step}/{stop_iters}]",
                            "{meters}",
                            "max mem: {memory:.2f} GB",
                        ]
                    ).format(
                        eta=eta_string,
                        timestamp=self.train_dataset.current_time,
                        meters=str(meters),
                        memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0 / 1024.0,
                    )
                )
            if step % (stop_iters // 5) == 0:
                timestamp = self.train_dataset.current_time
                self.eval_dataset.mode = 1
                self.eval_dataset.update_current_timestamp(timestamp)
                test_id_dataloader = FastDataLoader(dataset=self.eval_dataset,
                                                    batch_size=self.mini_batch_size,
                                                    num_workers=self.num_workers, collate_fn=self.eval_collate_fn)
                acc = self.network_evaluation(test_id_dataloader)
                self.logger.info("[{}/{}]  ID timestamp = {}: \t {} is {:.3f}".format(step, stop_iters, timestamp, self.eval_metric, acc * 100.0))
        self.logger.info("-------------------end training on timestamp {}-------------------".format(self.train_dataset.current_time))


def store_grad(params, grads, grad_dims):
    """
    This stores parameter gradients of past tasks.
        pp: parameters
        grads: gradients
        grad_dims: list with number of parameters per layers
    """
    # store the gradients
    grads.fill_(0.0)
    count = 0
    for param in params():
        if param.grad is not None:
            begin = 0 if count == 0 else sum(grad_dims[:count])
            end = np.sum(grad_dims[:count + 1])
            grads[begin: end].copy_(param.grad.data.view(-1))
        count += 1


def overwrite_grad(params, newgrad, grad_dims):
    """
    This is used to overwrite the gradients with a new gradient
    vector, whenever violations occur.
        pp: parameters
        newgrad: corrected gradient
        grad_dims: list storing number of parameters at each layer
    """
    count = 0
    for param in params():
        if param.grad is not None:
            begin = 0 if count == 0 else sum(grad_dims[:count])
            end = sum(grad_dims[:count + 1])
            this_grad = newgrad[begin: end].contiguous().view(
                param.grad.data.size())
            param.grad.data.copy_(this_grad)
        count += 1


def project(gxy: torch.Tensor, ger: torch.Tensor) -> torch.Tensor:
    corr = torch.dot(gxy, ger) / torch.dot(ger, ger)
    return gxy - corr * ger
