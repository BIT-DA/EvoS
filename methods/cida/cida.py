import datetime
import time

from ..base_trainer import BaseTrainer
from .wrapped_dataset import WrappedDataset
from copy import deepcopy
from .models import CIDAClassifier, CIDADiscriminator, CIDAYearbookFeature, CIDARotatedMNISTFeature
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F

from ..dataloaders import FastDataLoader
from ..utils import MetricLogger
import numpy as np
from sklearn.metrics import accuracy_score


label_noise_std = 0.20
use_label_noise = False
use_inverse_weighted = True
discr_thres = 999.999
normalize = True
train_discr_step_tot = 2
train_discr_step_extra = 0
slow_lrD_decay = 1


class CIDA(BaseTrainer):
    def __init__(self, args, logger, dataset, network, criterion, optimizer, scheduler):
        super(CIDA, self).__init__(args, logger, dataset, network, criterion, optimizer, scheduler)

        self.train_dataset = WrappedDataset(args, dataset)
        self.eval_dataset = deepcopy(self.train_dataset)

        if str(dataset) == "yearbook":
            self.feature = CIDAYearbookFeature(args, num_input_channels=3).cuda()
        elif str(dataset) == "rmnist":
            self.feature = CIDARotatedMNISTFeature(args, num_input_channels=1).cuda()
        else:
            raise ValueError

        self.classifier = CIDAClassifier(self.feature.output_dim, dataset.num_classes).cuda()

        self.discriminator = CIDADiscriminator(self.feature.output_dim).cuda()


        param_groups = [
            {"params": self.feature.parameters()},
            {"params": self.classifier.parameters()},
            {"params": self.discriminator.parameters()},
        ]

        self.extend_args(self.args)

        self.opt_D = optim.Adam(self.discriminator.parameters(), lr=self.args.lr)  # lr
        self.opt_non_D = optim.Adam(list(self.feature.parameters()) + list(self.classifier.parameters()),
                                    lr=self.args.lr)  # lr

        self.lr_scheduler_D = lr_scheduler.ExponentialLR(optimizer=self.opt_D, gamma=0.5 ** (
                    1 / (self.args.gamma_exp * (train_discr_step_extra + 1)) * slow_lrD_decay))

        self.lr_scheduler_non_D = lr_scheduler.ExponentialLR(optimizer=self.opt_non_D,
                                                             gamma=0.5 ** (1 / self.args.gamma_exp))

        self.scheduler = None

    @staticmethod
    def extend_args(args):
        """
        gamma_exp
        dis_lambda
        norm
        wgan
        clamp_lower
        clamp_upper
        """
        args.defrost()
        args.wgan = 'wgan'
        args.gamma_exp = 1000
        args.dis_lambda = 1.0
        args.lambda_m = 0.0
        args.clamp_lower = -0.15
        args.clamp_upper = 0.15
        args.norm = 2.0
        args.freeze()


    def run(self):
        self.train_offline()

        self.evaluate_offline()

    def train_step(self, dataloader):
        self.logger.info("-------------------start training on timestamp {}-------------------".format(
            self.train_dataset.current_time))
        self.network.train()
        loss_all = []
        meters = MetricLogger(delimiter="  ")
        end = time.time()
        self.logger.info("self.train_dataset.len = {} x {} = {} samples".format(
            self.train_dataset.__len__() // self.args.mini_batch_size, self.args.mini_batch_size,
            self.train_dataset.__len__()))

        stop_iters = self.train_dataset.__len__() // self.args.mini_batch_size
        for epoch in range(self.args.epochs):
            train(self.feature, self.classifier, self.discriminator, dataloader,
                  self.opt_D, self.opt_non_D, self.lr_scheduler_D, self.lr_scheduler_non_D,
                  epoch, self.args, True, self.train_dataset.train_list, stop_iters, self.logger)

            if (epoch + 1) % 25 == 0:
                timestamp = self.train_dataset.current_time
                self.eval_dataset.mode = 1
                self.eval_dataset.update_current_timestamp(timestamp)
                test_id_dataloader = FastDataLoader(dataset=self.eval_dataset,
                                                    batch_size=self.mini_batch_size,
                                                    num_workers=self.num_workers, collate_fn=self.eval_collate_fn)
                acc = self.network_evaluation(test_id_dataloader)
                self.logger.info(
                    "EPOCH: [{}/{}]  ID timestamp = {}: \t {} is {:.3f}".format(epoch, self.args.epochs, timestamp,
                                                                                self.eval_metric, acc * 100.0))

    def network_evaluation(self, test_time_dataloader):
        loss, acc = test_classification(self.feature, self.classifier, self.discriminator, test_time_dataloader)
        return acc


def train(encoder, predictor, discriminator, train_loader,
          opt_D, opt_non_D, lr_scheduler_D, lr_scheduler_non_D,
          epoch, args, classification, train_list, stop_iter, logger):

    models = [encoder, predictor, discriminator]
    for model in models:
        model.train()
    sum_discr_loss = 0
    sum_total_loss = 0
    sum_pred_loss = 0

    for batch_idx, data_tuple in enumerate(train_loader):
        # print(batch_idx)
        if batch_idx + 1 >= stop_iter:
            break

        data_tuple = tuple(ele.cuda() for ele in data_tuple)

        data_raw, target, domain, data, mask = data_tuple
        domain = domain.float()

        # FF encoder and predictor
        encoding = encoder(data, domain)
        prediction = predictor(encoding)
        prediction = torch.nn.functional.log_softmax(prediction)

        if use_label_noise:
            noise = (torch.randn(domain.size()).cuda() * label_noise_std).unsqueeze(1)

        # train discriminator
        train_discr_step = 0
        while args.dis_lambda > 0.0:
            train_discr_step += 1
            discr_pred_m, discr_pred_s = discriminator(encoding)
            discr_loss = gaussian_loss(discr_pred_m, discr_pred_s, domain.unsqueeze(1) / args.norm, np.mean(train_list) / args.norm, args.norm)
            for model in models:
                model.zero_grad()
            discr_loss.backward(retain_graph=True)
            opt_D.step()

            # handle extra steps to train the discr's variance branch
            if train_discr_step_extra > 0:
                cur_extra_step = 0
                while True:
                    discr_pred_m, discr_pred_s = discriminator(encoding)
                    discr_loss = gaussian_loss(discr_pred_m.detach(), discr_pred_s, domain.unsqueeze(1) / args.norm)
                    for model in models:
                        model.zero_grad()
                    discr_loss.backward(retain_graph=True)
                    opt_D.step()
                    cur_extra_step += 1
                    if cur_extra_step > train_discr_step_extra:
                        break

            if discr_loss.item() < 1.1 * discr_thres and train_discr_step >= train_discr_step_tot:
                sum_discr_loss += discr_loss.item()
                break

        # handle wgan
        if args.wgan == 'wgan':
            for p in discriminator.parameters():
                p.data.clamp_(args.clamp_lower, args.clamp_upper)

        # train encoder and predictor
        if classification:
            pred_loss = masked_cross_entropy(prediction, target, mask)
        else:
            pred_loss = masked_mse(prediction, target, mask)

        discr_pred_m, discr_pred_s = discriminator(encoding)
        ent_loss = 0

        discr_loss = gaussian_loss(discr_pred_m, discr_pred_s, domain.unsqueeze(1) / args.norm)
        total_loss = pred_loss - discr_loss * args.dis_lambda

        for model in models:
            model.zero_grad()
        total_loss.backward()
        opt_non_D.step()
        sum_pred_loss += pred_loss.item()
        sum_total_loss += total_loss.item()

    lr_scheduler_D.step()
    lr_scheduler_non_D.step()

    avg_discr_loss = sum_discr_loss / stop_iter
    avg_pred_loss = sum_pred_loss / stop_iter
    avg_total_loss = sum_total_loss / stop_iter
    log_txt = 'Train Epoch {}: avg_discr_loss = {:.5f}, avg_pred_loss = {:.3f}, avg_total_loss = {:.3f}'.format(epoch, avg_discr_loss, avg_pred_loss, avg_total_loss)
    print(log_txt)
    logger.info(log_txt)


def masked_cross_entropy(pred, label, mask):
    """ get masked cross entropy loss, for those training data padded with zeros at the end """
    temp = pred * mask.unsqueeze(1)

    label = label.long()
    loss = F.nll_loss(temp, label, reduction="sum")
    loss = loss / (mask.sum(0) + 1e-10)

    return loss


def masked_mse(pred, label, mask):
    """ get masked cross entropy loss, for those training data padded with zeros at the end """

    temp = pred * mask.unsqueeze(1)
    loss = F.mse_loss(temp, label.unsqueeze(1), reduction="sum")
    loss = loss / (mask.sum(0) + 1e-10)

    return loss


def gaussian_loss(pred_m, pred_s, label, mean=0, norm=15):
    """gaussian loss taking mean and (log) variance as input"""
    length, dim = pred_m.size()
    term1 = torch.sum((pred_m - label) ** 2 / (torch.exp(pred_s))) / length / dim

    term2 = 0.5 * torch.sum(pred_s) / length / dim

    delta = norm // 2 + 1 - torch.abs(label - mean) * norm * 1.0
    delta = delta.clone().detach()
    delta.data.clamp_(1.0, 10.0)
    term3 = 0.01 * torch.sum((1 / torch.exp(pred_s) - delta) ** 2) / length / dim #0.05

    return term1 + term2 + term3


def plain_log(filename, text):
    fp = open(filename, 'a')
    fp.write(text)
    fp.close()


def test_classification(encoder, predictor, discriminator, test_loader):
    models = [encoder, predictor, discriminator]
    for model in models:
        model.eval()
    test_loss = 0
    l_label = []
    l_true = []
    # for data, target, domain in test_loader:
    for data_tuple in test_loader:

        data_tuple = tuple(ele.cuda() for ele in data_tuple)

        data_raw, target, domain, data, mask = data_tuple
        domain = domain.float()

        encoding = encoder(data, domain)
        prediction = predictor(encoding)
        preds = torch.argmax(prediction, 1)
        l_label += list(preds.detach().cpu().numpy())
        l_true += list(target.long().clone().cpu().numpy())

    test_loss /= len(l_label)

    acc = accuracy_score(l_true, l_label)
    print('Accuracy: ', acc)

    return test_loss, acc
