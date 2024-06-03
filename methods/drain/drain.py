from ..base_trainer import BaseTrainer
from .modules import WrappedDrainNetwork
import torch.optim

from ..dataloaders import FastDataLoader
import numpy as np


class Drain(BaseTrainer):
    def __init__(self, args, logger, dataset, network, criterion, optimizer, scheduler):
        network = WrappedDrainNetwork(network, args.hidden_dim, args.latent_dim, args.num_rnn_layers, args.num_layer_to_replace, args.window_size, args.lambda_forgetting, args.dim_bottleneck_f)
        network.cuda()
        optimizer, scheduler = self.reconstruct_optimizer_and_scheduler(args, network)
        super(Drain, self).__init__(args, logger, dataset, network, criterion, optimizer, scheduler)

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

    def train_step(self, dataloader):
        super().train_step(dataloader)
        self.network.push_E()

    def evaluate_offline(self):
        self.logger.info(
            f'\n=================================== Results (Eval-Fix): Evolve Hidden for Test ===================================')
        self.evaluate_offline_implement1()

    def evaluate_offline_implement1(self):
        self.logger.info(f'Metric: {self.eval_metric}\n')
        timestamps = self.eval_dataset.ENV
        metrics = []
        for i, timestamp in enumerate(timestamps):
            if timestamp < self.split_time:
                pass
            elif timestamp == self.split_time:
                self.eval_dataset.mode = 1
                self.eval_dataset.update_current_timestamp(timestamp)
                test_id_dataloader = FastDataLoader(dataset=self.eval_dataset,
                                                    batch_size=self.mini_batch_size,
                                                    num_workers=self.num_workers, collate_fn=self.eval_collate_fn)
                id_metric = self.network_evaluation(test_id_dataloader)
                self.logger.info("Merged ID test {}: \t{:.3f}\n".format(self.eval_metric, id_metric * 100.0))
            else:
                self.eval_dataset.mode = 2
                self.eval_dataset.update_current_timestamp(timestamp)
                test_ood_dataloader = FastDataLoader(dataset=self.eval_dataset,
                                                     batch_size=self.mini_batch_size,
                                                     num_workers=self.num_workers, collate_fn=self.eval_collate_fn)
                acc = self.network_evaluation(test_ood_dataloader)
                self.network.update_hidden()
                self.network.push_E()
                self.logger.info("OOD timestamp = {}: \t {} is {:.3f}".format(timestamp, self.eval_metric, acc * 100.0))
                metrics.append(acc * 100.0)
        self.logger.info("\nOOD Average Metric: \t{:.3f}\nOOD Worst Metric: \t{:.3f}\nAll OOD Metrics: \t{}\n".format(np.mean(metrics), np.min(metrics), metrics))
