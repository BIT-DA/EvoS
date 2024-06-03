import torch.optim
import time
import datetime
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
from tdc import Evaluator
from torch.utils.data import DataLoader
from typing import Union

from .network.vae_algorithms import LSSAE
from .wrapped_network import WrappedFeature, WrappedClassifier, get_out_shape_hook
from ..base_trainer import BaseTrainer
from ..dataloaders import InfiniteDataLoader, FastDataLoader
from ..utils import prepare_data, MetricLogger
from .split_dataset import SubTimeDataset
from math import ceil


class LSSAETrainer(BaseTrainer):
    def __init__(self, args, logger, dataset, network, criterion, optimizer, scheduler):
        model_func = WrappedFeature(network)
        handle = model_func.register_forward_hook(get_out_shape_hook)

        cla_func = WrappedClassifier(network)

        infor: dict = self.fake_batch(args, model_func, cla_func, dataset)
        handle.remove()

        args.defrost()
        self.extend_args(args, infor)
        args.freeze()

        hparams = build_hparams(args)

        network = LSSAE(model_func, cla_func, hparams)
        network.cuda()
        scheduler = None

        super(LSSAETrainer, self).__init__(args, logger, dataset, network, criterion, optimizer, scheduler)

        self.split_train_datasets = [SubTimeDataset(self.train_dataset, idx, time) for idx, time in enumerate(self.train_dataset.ENV) if time <= self.split_time]
        self.split_val_dataset = [SubTimeDataset(self.eval_dataset, idx, time) for idx, time in enumerate(self.train_dataset.ENV)]

    def fake_batch(self, args, model_func: nn.Module, cla_func: nn.Module, dataset):
        ret = {}
        dataset.mode = 0
        time = dataset.ENV[0]
        dataset.update_current_timestamp(time)

        loader = DataLoader(dataset, batch_size=64, drop_last=False)

        dl_it = iter(loader)

        x, y = next(dl_it)
        x, y = prepare_data(x, y, str(dataset))
        with torch.no_grad():
            feature = model_func(x)
            out = cla_func(feature)

        count = 0
        for time in dataset.ENV:
            if time <= args.split_time:
                count += 1

        ret["data_size"] = list(x.shape[1:])
        ret["source_domains"] = count
        ret["num_classes"] = dataset.num_classes

        return ret

    @staticmethod
    def extend_args(args, infor: dict):
        args.source_domains = infor["source_domains"]  # useless
        args.num_classes = infor["num_classes"]
        args.data_size = infor["data_size"]
        args.stochastic = True
        args.zdy_dim = args.lssae_zw_dim  # zdy_dim for DIVA only

        args.zv_dim = args.num_classes

    def train_offline(self):
        self.logger.info("+------------------+ LSSEA Offline Train +------------------+")
        self.train_dataset.mode = 0
        assert any([ds.mode == 0 for ds in self.split_train_datasets])
        self.network.train()
        meters = MetricLogger(delimiter="  ")
        end = time.time()

        num_domains = len(self.split_train_datasets)
        mini_bs_sub_domain = ceil(self.mini_batch_size / num_domains)
        train_loaders = [InfiniteDataLoader(dataset=sub_time_ds, weights=None,
                                            batch_size=mini_bs_sub_domain,
                                            num_workers=0, collate_fn=self.train_collate_fn)
                         for sub_time_ds in self.split_train_datasets]
        iter_train_loaders = zip(*train_loaders)
        num_samples = sum([len(ds) for ds in self.split_train_datasets])
        iter_per_epoch = num_samples // (mini_bs_sub_domain * num_domains)
        stop_iter = self.args.epochs * iter_per_epoch
        self.logger.info("mini_bs_sub_domain = {}, num_training_domains = {}".format(mini_bs_sub_domain, num_domains))
        self.logger.info("self.train_dataset.len = {} x {} = {} samples".format(iter_per_epoch, mini_bs_sub_domain * num_domains, num_samples))

        current_iter = -1
        while current_iter < stop_iter:
            current_iter += 1

            mini_batches = next(iter_train_loaders)
            mini_batches = [prepare_data(x, y, str(self.train_dataset)) for x, y in mini_batches]

            loss, pred, targets, str_log = self.network.update(mini_batches, None)

            if (current_iter + 1) % iter_per_epoch == 0:
                if self.scheduler is not None:
                    self.scheduler.step()

            # -----------------print log infromation------------
            batch_time = time.time() - end
            end = time.time()
            meters.update(time=batch_time)
            eta_seconds = meters.time.global_avg * (stop_iter - current_iter)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
            meters.update(loss=loss.item())
            if current_iter % self.args.print_freq == 0:
                self.logger.info(
                    meters.delimiter.join(
                        [
                            "eta: {eta}",
                            "timestamp: {timestamp}",
                            f"[iter: {current_iter}/{stop_iter}]",
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
                self.logger.info(str_log)
        self.logger.info("+------------------+ Finished LSSEA Offline Train +------------------+")

    def evaluate_offline(self):
        self.logger.info(f'\n=================================== Results (Eval-Fix) ===================================')
        self.logger.info(f'Metric: {self.eval_metric}\n')
        timestamps = self.eval_dataset.ENV
        metrics = []

        id_loaders = []
        domain_indices = []
        for i, timestamp in enumerate(timestamps):
            if timestamp < self.split_time:
                self.eval_dataset.mode = 1
                sub_ds = SubTimeDataset(self.eval_dataset, i, timestamp)
                domain_indices.append(i)
                id_loaders.append(FastDataLoader(
                        dataset=sub_ds,
                        batch_size=self.mini_batch_size,
                        num_workers=self.num_workers,
                        collate_fn=self.eval_collate_fn))
            elif timestamp == self.split_time:
                self.eval_dataset.mode = 1
                sub_ds = SubTimeDataset(self.eval_dataset, i, timestamp)
                domain_indices.append(i)
                id_loaders.append(FastDataLoader(
                    dataset=sub_ds,
                    batch_size=self.mini_batch_size,
                    num_workers=self.num_workers,
                    collate_fn=self.eval_collate_fn))
                id_metric = self.network_evaluation(id_loaders, domain_indices)
                self.logger.info("Merged ID test {}: \t{:.3f}\n".format(self.eval_metric, id_metric * 100.0))
            else:
                self.eval_dataset.mode = 2
                sub_ds = SubTimeDataset(self.eval_dataset, i, timestamp)
                test_ood_dataloader = FastDataLoader(dataset=sub_ds,
                                        batch_size=self.mini_batch_size,
                                        num_workers=self.num_workers,
                                        collate_fn=self.eval_collate_fn)
                acc = self.network_evaluation(test_ood_dataloader, i)
                self.logger.info("OOD timestamp = {}: \t {} is {:.3f}".format(timestamp, self.eval_metric, acc * 100.0))
                metrics.append(acc * 100.0)
        self.logger.info("\nOOD Average Metric: \t{:.3f}\nOOD Worst Metric: \t{:.3f}\nAll OOD Metrics: \t{}\n".format(np.mean(metrics), np.min(metrics), metrics))

    def network_evaluation(self, loaders, domain_indices):
        self.network.eval()
        pred_all = []
        y_all = []

        if not isinstance(domain_indices, list):
            loaders = [loaders]
            domain_indices = [domain_indices]

        for d_id, loader in zip(domain_indices, loaders):
            for _, sample in enumerate(loader):
                if len(sample) == 3:
                    x, y, _ = sample
                else:
                    x, y = sample
                x, y = prepare_data(x, y, str(self.eval_dataset))
                with torch.no_grad():
                    logits = self.network.predict(x, d_id)
                    pred = F.softmax(logits, dim=1).argmax(dim=1)
                    pred_all = list(pred_all) + pred.detach().cpu().numpy().tolist()
                    y_all = list(y_all) + y.cpu().numpy().tolist()

        pred_all = np.array(pred_all)
        y_all = np.array(y_all)
        correct = (pred_all == y_all).sum().item()
        metric = correct / float(y_all.shape[0])
        self.network.train()
        return metric

def build_hparams(args):
    hparams = {}
    keys = ["source_domains", "num_classes", "data_size", "lssae_zc_dim", "stochastic",
            "zdy_dim", "lssae_zw_dim", "zv_dim", "lssae_coeff_y", "lssae_coeff_w", "lssae_coeff_ts", "lr", 'weight_decay'] ## zdy_dim for DIVA only
    for key in keys:
        hparams[key] = getattr(args, key)

    return hparams


