import torch.optim
import time
import datetime
import numpy as np

import torch.nn.functional as F
from sklearn import metrics
from tdc import Evaluator
from torch.utils.data import DataLoader
import torch.nn as nn

from ..base_trainer import BaseTrainer
from .wrapped_network import WrappedGINetwork
from .wrapped_datasets import WrappedDataset
from ..dataloaders import InfiniteDataLoader, FastDataLoader
from ..utils import prepare_data, forward_pass, MetricLogger
from ..lssae.wrapped_network import WrappedFeature, WrappedClassifier
from .wrapped_network import WrappedGINetwork, WrappedGIFeature, TimeReluClassifier


class GI(BaseTrainer):
    def __init__(self, args, logger, dataset, network, criterion, optimizer, scheduler):
        args.defrost()
        self.extend_args(args, str(dataset))
        dataset = WrappedDataset(dataset, args.split_time)

        feature = WrappedFeature(network)
        classifier = WrappedClassifier(network)

        info = self.fake_batch_collect_info(args, feature, classifier, dataset)

        feature = WrappedGIFeature(feature, args.time_dim)

        new_classifier = TimeReluClassifier(info["feature_dim"] + args.time_append_dim, info["num_classes"], args.time_dim)

        network = WrappedGINetwork(feature, new_classifier, args.time_dim, args.time_append_dim)
        network.cuda()

        self.fake_batch_init_timerelu(network, dataset)

        optimizer, scheduler = self.reconstruct_optimizer_and_scheduler(args, network)
        super(GI, self).__init__(args, logger, dataset, network, criterion, optimizer, scheduler)
        self.train_dataset: WrappedDataset

    @staticmethod
    def extend_args(args, ds_name):
        args.delta_lr = 0.5
        args.delta_clamp = 0.5
        args.delta_steps = 20
        args.lambda_GI = 1.0

    def fake_batch_collect_info(self, args, model_func: nn.Module, cla_func: nn.Module, dataset: WrappedDataset):
        ret = {}
        dataset.mode = 0
        dataset.use_as_time_sequence()
        time = dataset.ENV[0]
        dataset.update_current_timestamp(time)

        loader = DataLoader(dataset, batch_size=64, drop_last=False)

        dl_it = iter(loader)

        x, y, t = next(dl_it)
        x, y = prepare_data(x, y, str(dataset))
        with torch.no_grad():
            feature = model_func(x)
            out = cla_func(feature)

        ret["num_classes"] = dataset.num_classes
        ret["feature_dim"] = feature.shape[-1]

        return ret

    def fake_batch_init_timerelu(self, network: WrappedGINetwork, dataset: WrappedDataset):
        # init the TimeRELU
        dataset.mode = 0
        dataset.use_as_time_sequence()
        time = dataset.ENV[0]
        dataset.update_current_timestamp(time)

        loader = DataLoader(dataset, batch_size=64, drop_last=False)

        dl_it = iter(loader)

        with torch.no_grad():
            x, y, t = next(dl_it)
            x, y = prepare_data(x, y, str(dataset))
            t = t.cuda()
            # fake forward to init the TimeRELU in network
            network(x, t)

    def reconstruct_optimizer_and_scheduler(self, args, network):
        scheduler = None
        if args.dataset == "yearbook":
            optimizer = torch.optim.Adam(network.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.dataset == "fmow":
            optimizer = torch.optim.Adam((network.parameters()), lr=args.lr, weight_decay=args.weight_decay, amsgrad=True, betas=(0.9, 0.999))
        elif args.dataset == "arxiv":
            optimizer = torch.optim.Adam(network.parameters(), lr=args.lr, weight_decay=args.weight_decay, amsgrad=True, betas=(0.9, 0.999))
        elif args.dataset == "huffpost":
            optimizer = torch.optim.Adam(network.parameters(), lr=args.lr, weight_decay=args.weight_decay, amsgrad=True, betas=(0.9, 0.999))
        elif args.dataset == "rmnist":
            optimizer = torch.optim.Adam(network.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        else:
            raise NotImplementedError(f"not implement for dataset {args.dataset}")
        return optimizer, scheduler

    def run(self):
        torch.cuda.empty_cache()
        start_time = time.time()
        self.pre_train()

        # cost too much computation resource
        # The A100 is not enough to support even the batch size is reduced to 4
        if str(self.train_dataset) in ["yearbook", "rmnist"]:
            self.gi_finetune()

        self.eval_dataset.use_as_time_sequence()

        self.evaluate_offline()
        runtime = time.time() - start_time
        runtime = runtime / 60 / 60
        self.logger.info(f'Runtime: {runtime:.2f} h\n')

    def pre_train(self):
        self.logger.info("+-----------------------------+ Pre-Train for GI +-----------------------------+")
        self.network.train()
        self.train_dataset.mode = 0
        self.train_dataset.mix_all_data()
        loader = InfiniteDataLoader(dataset=self.train_dataset,
                                    weights=None,
                                    batch_size=self.mini_batch_size,
                                    num_workers=self.num_workers,
                                    collate_fn=self.train_collate_fn)
        self.logger.info(f"self.train_dataset.len = "
                         f"{len(self.train_dataset) // self.args.mini_batch_size} x {self.args.mini_batch_size} "
                         f"= {len(self.train_dataset)} samples")

        meters = MetricLogger(delimiter="  ")
        end = time.time()
        stop_iter = self.args.epochs * (len(self.train_dataset) // self.args.mini_batch_size) - 1
        for step, (x, y, t) in enumerate(loader):
            if step >= stop_iter:
                break
            t = t.cuda()
            x, y = prepare_data(x, y, str(self.train_dataset))

            loss, logits, y = forward_pass(x, y, t, self.train_dataset, self.network, self.criterion)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            batch_time = time.time() - end
            end = time.time()

            meters.update(time=batch_time)
            eta_seconds = meters.time.global_avg * (stop_iter - step)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
            meters.update(loss=loss.item())
            if step % self.args.print_freq == 0:
                self.logger.info(
                    meters.delimiter.join(
                        [
                            "eta: {eta}",
                            f"[iter:{step}/{stop_iter}]",
                            "{meters}",
                            "max mem: {memory:.2f} GB",
                        ]
                    ).format(
                        eta=eta_string,
                        meters=str(meters),
                        memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0 / 1024.0,
                    )
                )
            if self.scheduler is not None and (step + 1) % (len(self.train_dataset) // self.args.mini_batch_size) == 0:
                self.scheduler.step()
        self.logger.info("+-----------------------------+ Pre-Train for GI End +-----------------------------+")

    def gi_finetune(self):
        self.logger.info("+-----------------------------+ Fine-tune for GI +-----------------------------+")
        self.network.train()
        self.train_dataset.mode = 0
        self.train_dataset.use_as_time_sequence()
        start_i = None if self.args.gi_start_to_finetune is None else self.args.split_time - self.args.gi_start_to_finetune + 1

        for i, timestamp in enumerate(self.train_dataset.ENV[:-1]):
            if start_i is not None and timestamp < start_i:
                continue

            if timestamp == (self.split_time + 1):
                break
            self.logger.info(f"+-----------------------------+ Fine-tune {timestamp} +-----------------------------+")

            self.train_dataset.update_current_timestamp(timestamp)
            loader = InfiniteDataLoader(dataset=self.train_dataset,
                                        weights=None,
                                        batch_size=self.args.gi_finetune_bs,
                                        num_workers=self.num_workers,
                                        collate_fn=self.train_collate_fn)

            self.logger.info(f"self.train_dataset.len = "
                             f"{len(self.train_dataset) // self.args.gi_finetune_bs} x {self.args.gi_finetune_bs} "
                             f"= {len(self.train_dataset)} samples")

            meters = MetricLogger(delimiter="  ")
            end = time.time()
            stop_iter = self.args.gi_finetune_epochs * (len(self.train_dataset) // self.args.mini_batch_size) - 1
            network_optimizer = torch.optim.Adam(self.network.parameters(), 5e-4)

            for step, (x, y, t) in enumerate(loader):
                if step >= stop_iter:
                    break
                t = t.cuda()
                t = t.view((-1, 1))
                x, y = prepare_data(x, y, str(self.train_dataset))
                delta = (torch.rand(t.size()).float() * (2 * self.args.delta_clamp) - self.args.delta_clamp).cuda()

                pred_loss, delta = adversarial_finetune(x, t, y, delta, self.network,
                                                        network_optimizer, self.criterion,
                                                        delta_lr=self.args.delta_lr, delta_clamp=self.args.delta_clamp,
                                                        delta_steps=self.args.delta_steps,
                                                        lambda_GI=self.args.lambda_GI,
                                                        writer=self.logger, step=step, string="delta_{}".format(i),
                                                        verbose=step % 20 == 0, ds_name=str(self.train_dataset))

                batch_time = time.time() - end
                end = time.time()
                meters.update(time=batch_time)
                eta_seconds = meters.time.global_avg * (stop_iter - step)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                meters.update(pred_loss=pred_loss.item())
                if step % self.args.print_freq == 0:
                    self.logger.info(
                        meters.delimiter.join(
                            [
                                "eta: {eta}",
                                f"[iter:{step}/{stop_iter}]",
                                "{meters}",
                                "max mem: {memory:.2f} GB",
                            ]
                        ).format(
                            eta=eta_string,
                            meters=str(meters),
                            memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0 / 1024.0,
                        )
                    )

    def network_evaluation(self, test_time_dataloader):
        self.network.eval()
        pred_all = []
        y_all = []
        for _, sample in enumerate(test_time_dataloader):
            x, y, t = sample
            t = t.cuda()
            x, y = prepare_data(x, y, str(self.eval_dataset))
            with torch.no_grad():
                logits = self.network(x, t)
                if self.args.dataset in ['drug']:
                    pred = logits.reshape(-1, )
                else:
                    pred = F.softmax(logits, dim=1).argmax(dim=1)
                pred_all = list(pred_all) + pred.detach().cpu().numpy().tolist()
                y_all = list(y_all) + y.cpu().numpy().tolist()

        pred_all = np.array(pred_all)
        y_all = np.array(y_all)
        correct = (pred_all == y_all).sum().item()
        metric = correct / float(y_all.shape[0])
        self.network.train()
        return metric


def forward_pass(x, y, t, dataset, network, criterion):
    logits = network(x, t)
    if str(dataset) in ['arxiv', 'fmow', 'huffpost', 'yearbook']:
        if len(y.shape) > 1:
            y = y.squeeze(1)
    loss = criterion(logits, y)
    return loss, logits, y


def adversarial_finetune(X, U, Y, delta, classifier, classifier_optimizer, classifier_loss_fn, delta_lr=0.1,
                         delta_clamp=0.15, delta_steps=10, lambda_GI=0.5, writer=None, step=None, string=None,
                         verbose=False, ds_name=""):
    if len(Y.shape) > 1:
        Y = Y.squeeze(1)

    classifier_optimizer.zero_grad()

    delta.requires_grad_(True)

    # This block of code computes delta adversarially
    d1 = delta.detach().cpu().numpy()
    for ii in range(delta_steps):
        delta = delta.clone().detach()
        delta.requires_grad_(True)
        U_grad = U.clone() - delta
        U_grad.requires_grad_(True)
        Y_pred = classifier(X, U_grad)
        if len(Y.shape) > 1 and Y.shape[1] > 1:
            Y_true = torch.argmax(Y, 1).view(-1, 1).float()

        partial_logit_pred_t = []
        if len(Y_pred.shape) < 2 or Y_pred.shape[1] < 2:
            partial_Y_pred_t = \
                torch.autograd.grad(Y_pred, U_grad, grad_outputs=torch.ones_like(Y_pred), retain_graph=True,
                                    create_graph=True)[0]
        else:
            for idx in range(Y_pred.shape[1]):
                logit = Y_pred[:, idx].view(-1, 1)
                partial_logit_pred_t.append(
                    torch.autograd.grad(logit, U_grad, grad_outputs=torch.ones_like(logit), create_graph=True)[0])

            partial_Y_pred_t = torch.cat(partial_logit_pred_t, 1)

        Y_pred = Y_pred + delta * partial_Y_pred_t

        if len(Y_pred.shape) > 1 and Y_pred.shape[1] > 1:
            Y_pred = torch.softmax(Y_pred, dim=-1)
        loss = classifier_loss_fn(Y_pred, Y).mean()
        partial_loss_delta = torch.autograd.grad(loss, delta, grad_outputs=torch.ones_like(loss), retain_graph=True)[0]
        delta = delta + delta_lr * partial_loss_delta

        if delta.size(0) > 1:
            delta[delta != delta] = 0.
            if torch.norm(partial_loss_delta) < 1e-3 * delta.size(0):
                break
        else:
            if torch.norm(partial_loss_delta) < 1e-3 or delta > delta_clamp or delta < -1 * delta_clamp:
                break
    #
    delta = delta.clamp(-1 * delta_clamp, delta_clamp).detach().clone()
    d2 = delta.detach().cpu().numpy()

    # This block of code actually optimizes our model
    U_grad = U.clone() - delta
    U_grad.requires_grad_(True)
    Y_pred = classifier(X, U_grad)

    partial_logit_pred_t = []

    if len(Y_pred.shape) < 2 or Y_pred.shape[1] < 2:
        partial_Y_pred_t = torch.autograd.grad(Y_pred, U_grad, grad_outputs=torch.ones_like(Y_pred), create_graph=True)[
            0]
    else:
        for idx in range(Y_pred.shape[1]):
            logit = Y_pred[:, idx].view(-1, 1)
            partial_logit_pred_t.append(
                torch.autograd.grad(logit, U_grad, grad_outputs=torch.ones_like(logit), retain_graph=True)[0])
        partial_Y_pred_t = torch.cat(partial_logit_pred_t, 1)

    Y_pred = Y_pred + delta * partial_Y_pred_t
    if len(Y_pred.shape) > 1 and Y_pred.shape[1] > 1:
        Y_pred = torch.softmax(Y_pred, dim=-1)
    Y_orig_pred = classifier(X, U)

    pred_loss = classifier_loss_fn(Y_pred, Y).mean() + lambda_GI * classifier_loss_fn(Y_orig_pred, Y).mean()
    pred_loss.backward()
    classifier_optimizer.step()
    return pred_loss, delta
