import copy
import os
import math
import time
import numpy as np
import datetime
import torch
import torch.utils.data
from torch.nn import functional as F
from sklearn import metrics
from tdc import Evaluator
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torchvision import transforms

from ..base_trainer import BaseTrainer
from ..utils import prepare_data, MetricLogger
from ..dataloaders import  FastDataLoader




class EvoS(BaseTrainer):
    def __init__(self, args, logger, dataset, network, criterion, optimizer, scheduler):
        super().__init__(args, logger, dataset, network, criterion, optimizer, scheduler)
        self.logger = logger
        self.eps = 1e-6

    def __str__(self):
        str_all = f'Our_EvoS-tradeoff_adv={self.args.tradeoff_adv}-{self.base_trainer_str}'
        return str_all

    def train_step(self, dataloader):
        self.logger.info("-------------------start training on timestamp {}-------------------".format(self.train_dataset.current_time))
        self.network.train()
        meters = MetricLogger(delimiter="  ")
        end = time.time()
        self.logger.info("self.train_dataset.len = {} x {} = {} samples".format(self.train_dataset.__len__() // self.args.mini_batch_size, self.args.mini_batch_size, self.train_dataset.__len__()))
        timestamp = self.train_dataset.current_time
        stop_iters = self.args.epochs * (self.train_dataset.__len__() // self.args.mini_batch_size)

        print("timestamp={}, init_timestamp={}".format(timestamp, self.args.init_timestamp))
        if timestamp - self.args.init_timestamp >= 2:
            previous_mean_logStd = self.network.get_previous_mean_logStd(timestamp)

        if self.args.warm_max_iters is None:
            self.network.rest_discriminator_lr(self.args.warm_multiply * stop_iters)
        else:
            self.network.rest_discriminator_lr(max_iters=self.args.warm_max_iters)

        for step, (x, y) in enumerate(dataloader):
            total_loss = 0
            x, y = prepare_data(x, y, str(self.train_dataset))
            f = self.network.foward_encoder(x)

            # --------consistency loss--------
            if timestamp - self.args.init_timestamp >= 2:
                mean_logStd_t, loss_consistency = self.network.foward_for_FeatureDistritbuion(previous_mean_logStd)
                if loss_consistency is not None:
                    total_loss += loss_consistency
                    meters.update(loss_consistency=(loss_consistency).item())
            else:
                mean_logStd_t = self.network.init_pool[timestamp - self.args.init_timestamp]
                mean_logStd_t = mean_logStd_t.cuda()

            # --------cross-entropy loss--------
            mean, logStd = mean_logStd_t[:, :self.network.feature_dim], mean_logStd_t[:, self.network.feature_dim:]    # mean.shape: [1, A]
            normalized_f = (f - mean.detach()) / (torch.exp(logStd.detach()) + self.eps)
            logits = self.network.foward_classifier(normalized_f)
            loss_ce = self.criterion(logits, y)
            total_loss += loss_ce
            meters.update(loss_ce=(loss_ce).item())

            # --------loss for aligning with the standard normal distribution--------
            mean_of_f = torch.mean((f.detach() - mean) / (torch.exp(logStd) + self.eps), dim=0)
            var_of_f = torch.var((f.detach() - mean) / (torch.exp(logStd) + self.eps), dim=0)
            loss_standardization = torch.norm(mean_of_f - 0, p=2) + torch.norm(var_of_f - 1, p=2)
            total_loss += loss_standardization
            meters.update(loss_standardization=(loss_standardization).item())

            # adversarial loss
            if timestamp - self.args.init_timestamp >= 1:
                if timestamp - self.args.init_timestamp == 1:
                    loss_adv, acc_dis = self.network.forward_domain_discriminator(f, self.network.init_pool[0].cuda())
                else:
                    loss_adv, acc_dis = self.network.forward_domain_discriminator(f, previous_mean_logStd)
                total_loss += self.args.tradeoff_adv * loss_adv
                meters.update(loss_adv=(loss_adv).item())
                meters.update(acc_dis=(acc_dis).item())

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            meters.update(total_loss=(total_loss).item())

            #-----------------print log infromation------------
            batch_time = time.time() - end
            end = time.time()
            meters.update(time=batch_time)
            eta_seconds = meters.time.global_avg * (stop_iters - step)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
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

            if step == stop_iters:
                if self.scheduler is not None:
                    self.scheduler.step()
                break
        if timestamp - self.args.init_timestamp <= 1:
            self.network.memorize(timestamp, None)
        else:
            mean_logStd_t, _ = self.network.foward_for_FeatureDistritbuion(previous_mean_logStd)
            self.network.memorize(timestamp, mean_logStd_t)
        self.logger.info("-------------------end training on timestamp {}-------------------".format(self.train_dataset.current_time))

    def network_evaluation(self, test_time_dataloader):
        print("evaluate_time={}".format(self.eval_dataset.current_time))
        self.network.eval()
        pred_all = []
        y_all = []

        with torch.no_grad():
            if self.train_dataset.current_time - self.args.init_timestamp == 0:
                mean_logStd_t = self.network.init_pool[0]
                mean_logStd_t = mean_logStd_t.cuda()
            else:
                if self.eval_dataset.current_time - self.args.init_timestamp >= 2:
                    previous_mean_logStd = self.network.get_previous_mean_logStd(self.eval_dataset.current_time)
                    mean_logStd_t, _ = self.network.foward_for_FeatureDistritbuion(previous_mean_logStd)
                    if self.args.eval_fix:
                        if self.eval_dataset.current_time > self.args.split_time:
                            self.network.memorize(self.eval_dataset.current_time, mean_logStd_t)
                            print("------------memorizing during evaluation------------")
                    else:
                        if self.eval_dataset.current_time > self.train_dataset.current_time:
                            self.network.memorize(self.eval_dataset.current_time, mean_logStd_t)
                            print("------------memorizing during evaluation------------")
                else:
                    mean_logStd_t = self.network.init_pool[self.eval_dataset.current_time - self.args.init_timestamp]
                    mean_logStd_t = mean_logStd_t.cuda()
            mean, logStd = mean_logStd_t[:, :self.network.feature_dim], mean_logStd_t[:, self.network.feature_dim:]  # mean.shape: [1, A]

            for _, sample in enumerate(test_time_dataloader):
                if len(sample) == 3:
                    x, y, _ = sample
                else:
                    x, y = sample
                x, y = prepare_data(x, y, str(self.eval_dataset))
                logits = self.network.forward_evaluate(x, mean, torch.exp(logStd))
                pred = F.softmax(logits, dim=1).argmax(dim=1)
                pred_all = list(pred_all) + pred.detach().cpu().numpy().tolist()
                y_all = list(y_all) + y.cpu().numpy().tolist()

            pred_all = np.array(pred_all)
            y_all = np.array(y_all)
            correct = (pred_all == y_all).sum().item()
            metric = correct / float(y_all.shape[0])

        self.network.train()
        return metric


