import os

import torch
import time
import datetime
from .loss import LossComputer
from ..base_trainer import BaseTrainer
from ..utils import prepare_data, forward_pass, split_into_groups, MetricLogger
from ..dataloaders import FastDataLoader

class IRM(BaseTrainer):
    """
    Invariant risk minimization.

    Original paper:
        @article{arjovsky2019invariant,
          title={Invariant risk minimization},
          author={Arjovsky, Martin and Bottou, L{\'e}on and Gulrajani, Ishaan and Lopez-Paz, David},
          journal={arXiv preprint arXiv:1907.02893},
          year={2019}
        }

    Code adapted from https://github.com/p-lambda/wilds/blob/main/examples/algorithms/IRM.py.
    """
    def __init__(self, args, logger, dataset, network, criterion, optimizer, scheduler):
        super().__init__(args, logger, dataset, network, criterion, optimizer, scheduler)
        self.update_count = 0
        self.irm_lambda = args.irm_lambda
        self.irm_penalty_anneal_iters = args.irm_penalty_anneal_iters
        self.scale = torch.tensor(1.).requires_grad_()
        dataset.current_time = dataset.ENV[0]
        self.loss_computer = LossComputer(self.train_dataset, criterion, is_robust=True)

    def __str__(self):
        return f'IRM-irm_lambda={self.irm_lambda}-irm_penalty_anneal_iters={self.irm_penalty_anneal_iters}' \
               f'-{self.base_trainer_str}'

    def irm_penalty(self, losses):
        grad_1 = torch.autograd.grad(losses[0::2].mean(), [self.scale], create_graph=True)[0]
        grad_2 = torch.autograd.grad(losses[1::2].mean(), [self.scale], create_graph=True)[0]
        result = torch.sum(grad_1 * grad_2)
        return result

    def train_step(self, dataloader):
        self.network.train()
        loss_all = []
        meters = MetricLogger(delimiter="  ")
        end = time.time()
        self.logger.info("self.train_dataset.len = {} x {} = {} samples".format(self.train_dataset.__len__() // self.args.mini_batch_size, self.args.mini_batch_size, self.train_dataset.__len__()))
        stop_iters = self.args.epochs * (self.train_dataset.__len__() // self.args.mini_batch_size) - 1
        for step, (x, y, g) in enumerate(dataloader):
            x, y = prepare_data(x, y, str(self.train_dataset))
            g = g.squeeze(1).cuda()

            self.network.zero_grad()
            unique_groups, group_indices, _ = split_into_groups(g)
            n_groups_per_batch = unique_groups.numel()
            avg_loss = 0.
            penalty = 0.
            _, logits, y = forward_pass(x, y, self.train_dataset, self.network, self.criterion, self.lisa, self.mixup,
                                        self.cut_mix, self.mix_alpha)
            for i_group in group_indices:
                group_losses = self.criterion(self.scale * logits[i_group], y[i_group])
                if group_losses.numel() > 0:
                    avg_loss += group_losses.mean()
                penalty += self.irm_penalty(group_losses)
            avg_loss /= n_groups_per_batch
            penalty /= n_groups_per_batch

            if self.update_count >= self.irm_penalty_anneal_iters:
                penalty_weight = self.irm_lambda
            else:
                penalty_weight = 1.0

            loss = avg_loss + penalty * penalty_weight
            loss_all.append(loss.item())
            loss.backward()
            self.optimizer.step()

            if step == stop_iters:
                if self.scheduler is not None:
                    self.scheduler.step()
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
