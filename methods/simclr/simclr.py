import os
import time
import torch
import datetime
from ..dataloaders import InfiniteDataLoader, FastDataLoader
from lightly.loss import NTXentLoss
from ..base_trainer import BaseTrainer
from ..utils import prepare_data, forward_pass, MetricLogger


class SimCLR(BaseTrainer):
    """
    SimCLR

    Original paper:
        @inproceedings{chen2020simple,
            title={A simple framework for contrastive learning of visual representations},
            author={Chen, Ting and Kornblith, Simon and Norouzi, Mohammad and Hinton, Geoffrey},
            booktitle={International conference on machine learning},
            pages={1597--1607},
            year={2020},
            organization={PMLR}
        }

    Code uses Lightly, a Python library for self-supervised learning on images: https://github.com/lightly-ai/lightly.
    """

    def __init__(self, args, logger, dataset, network, criterion, optimizer, scheduler):
        assert str(dataset) == 'yearbook' or str(dataset) == 'fmow' or str(dataset) == 'rmnist', \
            'SimCLR on image classification datasets only'
        super().__init__(args, logger, dataset, network, criterion, optimizer, scheduler)
        self.logger = logger
        self.network.ssl_training = True
        self.ssl_criterion = NTXentLoss()

    def __str__(self):
        return f'SimCLR-{self.base_trainer_str}'

    def finetune_classifier(self):
        self.network.ssl_training = False
        self.train_dataset.ssl_training = False
        finetune_dataloader = InfiniteDataLoader(dataset=self.train_dataset, weights=None,
                                                 batch_size=self.mini_batch_size,
                                                 num_workers=self.num_workers, collate_fn=self.eval_collate_fn)
        self.network.train()
        loss_all = []

        num_iters = self.args.epochs * (self.train_dataset.__len__() // self.args.mini_batch_size)
        for step, (x, y) in enumerate(finetune_dataloader):
            x, y = prepare_data(x, y, str(self.train_dataset))
            loss, logits, y = forward_pass(x, y, self.train_dataset, self.network, self.criterion, self.lisa, self.mixup,
                                           self.cut_mix, self.mix_alpha)
            loss_all.append(loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if (step + 1) == num_iters:
                if self.scheduler is not None:
                    self.scheduler.step()
                break

    def train_step(self, dataloader):
        self.logger.info("-------------------start training on timestamp {}-------------------".format(self.train_dataset.current_time))
        self.network.train()
        self.network.ssl_training = True
        self.train_dataset.ssl_training = True
        loss_all = []
        meters = MetricLogger(delimiter="  ")
        end = time.time()
        self.logger.info("self.train_dataset.len = {} x {} = {} samples".format(self.train_dataset.__len__() // self.args.mini_batch_size, self.args.mini_batch_size, self.train_dataset.__len__()))
        stop_iters = self.args.epochs * (self.train_dataset.__len__() // self.args.mini_batch_size) - 1
        for step, (x, y, _) in enumerate(dataloader):
            (x0, x1), y = prepare_data(x, y, str(self.train_dataset))

            z0 = self.network(x0)
            z1 = self.network(x1)
            loss = self.ssl_criterion(z0, z1)

            loss_all.append(loss.item())
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            if step == stop_iters:
                if self.scheduler is not None:
                    self.scheduler.step()
                self.finetune_classifier()
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
        self.logger.info("-------------------end training on timestamp {}-------------------".format(self.train_dataset.current_time))
