import copy
import time
import datetime
import numpy as np
import torch
import torch.nn.functional as F

from .dataloaders import FastDataLoader, InfiniteDataLoader
from .utils import prepare_data, forward_pass, get_collate_functions, MetricLogger


class BaseTrainer:
    def __init__(self, args, logger, dataset, network, criterion, optimizer, scheduler):
        self.network = network
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.logger = logger

        # Dataset settings
        self.train_dataset = dataset
        self.train_dataset.mode = 0
        self.eval_dataset = copy.deepcopy(dataset)
        self.eval_dataset.mode = 2
        self.num_classes = dataset.num_classes
        # self.num_tasks = dataset.num_tasks
        self.train_collate_fn, self.eval_collate_fn = get_collate_functions(args, self.train_dataset)

        # Training hyperparameters
        self.args = args
        self.lisa = args.lisa
        self.epochs = args.epochs
        self.mixup = args.mixup
        self.cut_mix = args.cut_mix
        self.mix_alpha = args.mix_alpha
        self.mini_batch_size = args.mini_batch_size
        self.num_workers = args.num_workers
        self.base_trainer_str = self.get_base_trainer_str()

        # Evaluation and metrics
        self.split_time = args.split_time
        self.eval_next_timestamps = args.eval_next_timestamps
        self.task_accuracies = {}
        self.worst_time_accuracies = {}
        self.best_time_accuracies = {}
        self.eval_metric = 'accuracy'

    def __str__(self):
        pass

    def get_base_trainer_str(self):
        base_trainer_str = f'epochs={self.epochs}-lr={self.args.lr}-' \
                                f'mini_batch_size={self.args.mini_batch_size}-seed={self.args.random_seed}'
        if self.args.lisa:
            base_trainer_str += f'-lisa-mix_alpha={self.mix_alpha}'
        elif self.mixup:
            base_trainer_str += f'-mixup-mix_alpha={self.mix_alpha}'
        if self.cut_mix:
            base_trainer_str += f'-cut_mix'
        if self.args.eval_fix:
            base_trainer_str += f'-eval_fix'
        else:
            base_trainer_str += f'-eval_stream'
        return base_trainer_str

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
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if step == stop_iters:
                if self.scheduler is not None:
                    self.scheduler.step()
                break
            #-----------------print log infromation------------
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

    def train_online(self):
        self.train_dataset.mode = 0
        for i, timestamp in enumerate(self.train_dataset.ENV[:-1]):
            if self.args.eval_fix and timestamp == (self.split_time + 1):
                break
            else:
                if self.args.lisa and i == self.args.lisa_start_time:
                    self.lisa = True
                self.train_dataset.update_current_timestamp(timestamp)
                if self.args.method in ['simclr', 'swav']:
                    self.train_dataset.ssl_training = True
                train_dataloader = InfiniteDataLoader(dataset=self.train_dataset, weights=None, batch_size=self.mini_batch_size,
                                                      num_workers=self.num_workers, collate_fn=self.train_collate_fn)
                self.train_step(train_dataloader)

                self.eval_dataset.mode = 1
                self.eval_dataset.update_current_timestamp(timestamp)
                test_id_dataloader = FastDataLoader(dataset=self.eval_dataset,
                                                     batch_size=self.mini_batch_size,
                                                     num_workers=self.num_workers, collate_fn=self.eval_collate_fn)
                acc = self.network_evaluation(test_id_dataloader)
                self.logger.info("ID timestamp = {}: \t {} is {:.3f}".format(timestamp, self.eval_metric, acc * 100.0))

    def train_offline(self):
        if self.args.method in ['simclr', 'swav']:
            self.train_dataset.ssl_training = True
        for i, timestamp in enumerate(self.train_dataset.ENV):
            if timestamp < self.split_time:
                self.train_dataset.mode = 0
                self.train_dataset.update_current_timestamp(timestamp)
                self.train_dataset.update_historical(i + 1)
                self.train_dataset.mode = 1
                self.train_dataset.update_current_timestamp(timestamp)
                self.train_dataset.update_historical(i + 1, data_del=True)
            elif timestamp == self.split_time:
                self.train_dataset.mode = 0
                self.train_dataset.update_current_timestamp(timestamp)
                if self.args.method in ['simclr', 'swav']:
                    self.train_dataset.ssl_training = True
                train_id_dataloader = InfiniteDataLoader(dataset=self.train_dataset, weights=None,
                                                         batch_size=self.mini_batch_size,
                                                         num_workers=self.num_workers, collate_fn=self.train_collate_fn)
                self.train_step(train_id_dataloader)
                break

    def network_evaluation(self, test_time_dataloader):
        self.network.eval()
        pred_all = []
        y_all = []
        for _, sample in enumerate(test_time_dataloader):
            if len(sample) == 3:
                x, y, _ = sample
            else:
                x, y = sample
            x, y = prepare_data(x, y, str(self.eval_dataset))
            with torch.no_grad():
                logits = self.network(x)
                pred = F.softmax(logits, dim=1).argmax(dim=1)
                pred_all = list(pred_all) + pred.detach().cpu().numpy().tolist()
                y_all = list(y_all) + y.cpu().numpy().tolist()
        pred_all = np.array(pred_all)
        y_all = np.array(y_all)
        correct = (pred_all == y_all).sum().item()
        metric = correct / float(y_all.shape[0])
        self.network.train()
        return metric

    def evaluate_stream(self, start):
        self.network.eval()
        metrics = []
        for i in range(start, min(start + self.eval_next_timestamps, len(self.eval_dataset.ENV))):
            test_time = self.eval_dataset.ENV[i]
            self.eval_dataset.mode = 2
            self.eval_dataset.update_current_timestamp(test_time)
            test_time_dataloader = FastDataLoader(dataset=self.eval_dataset, batch_size=self.mini_batch_size,
                                                  num_workers=self.num_workers, collate_fn=self.eval_collate_fn)
            metric = self.network_evaluation(test_time_dataloader)
            metrics.append(metric * 100.0)

        avg_metric, worst_metric, best_metric = np.mean(metrics), np.min(metrics), np.max(metrics)

        print(
            f'Timestamp = {start - 1}'
            f'\t Average {self.eval_metric}: {avg_metric}'
            f'\t Worst {self.eval_metric}: {worst_metric}'
            f'\t Best {self.eval_metric}: {best_metric}'
            f'\t Performance over all timestamps: {metrics}\n'
        )
        self.network.train()
        return avg_metric, worst_metric, best_metric, metrics

    def evaluate_offline(self):
        self.logger.info(f'\n=================================== Results (Eval-Fix) ===================================')
        self.logger.info(f'Metric: {self.eval_metric}\n')
        timestamps = self.eval_dataset.ENV
        metrics = []
        for i, timestamp in enumerate(timestamps):
            if timestamp < self.split_time:
                self.eval_dataset.mode = 1
                self.eval_dataset.update_current_timestamp(timestamp)
                self.eval_dataset.update_historical(i + 1, data_del=True)
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
                self.logger.info("OOD timestamp = {}: \t {} is {:.3f}".format(timestamp, self.eval_metric, acc * 100.0))
                metrics.append(acc * 100.0)
        if len(metrics) >= 2:
            self.logger.info("\nOOD Average Metric: \t{:.3f}\nOOD Worst Metric: \t{:.3f}\nAll OOD Metrics: \t{}\n".format(np.mean(metrics), np.min(metrics), metrics))

    def run_eval_fix(self):
        print('==========================================================================================')
        print("Running Eval-Fix...\n")
        if (self.args.method in ['agem', 'ewc', 'ft', 'si', 'drain', 'evos']) or self.args.online_switch:
            self.train_online()
        else:
            self.train_offline()
        self.evaluate_offline()

    def run_eval_stream(self):
        print('==========================================================================================')
        print("Running Eval-Stream...\n")
        self.train_dataset.mode = 0
        end = len(self.eval_dataset.ENV) - self.eval_next_timestamps
        for i, timestamp in enumerate(self.train_dataset.ENV[:end]):
            if self.args.lisa and i == self.args.lisa_start_time:
                self.lisa = True
            #----------train on the training set of current domain---------
            self.train_dataset.update_current_timestamp(timestamp)
            if self.args.method in ['simclr', 'swav']:
                self.train_dataset.ssl_training = True
            train_dataloader = InfiniteDataLoader(dataset=self.train_dataset, weights=None,
                                                      batch_size=self.mini_batch_size,
                                                      num_workers=self.num_workers, collate_fn=self.train_collate_fn)
            self.train_step(train_dataloader)

            # -------evaluate on the validation set of current domain-------
            self.eval_dataset.mode = 1
            self.eval_dataset.update_current_timestamp(timestamp)
            test_id_dataloader = FastDataLoader(dataset=self.eval_dataset,
                                                    batch_size=self.mini_batch_size,
                                                    num_workers=self.num_workers, collate_fn=self.eval_collate_fn)
            acc = self.network_evaluation(test_id_dataloader)
            self.logger.info("ID timestamp = {}: \t {} is {:.3f}".format(timestamp, self.eval_metric, acc * 100.0))

            # -------evaluate on the next K domains-------
            avg_metric, worst_metric, best_metric, all_metrics = self.evaluate_stream(i + 1)
            self.task_accuracies[timestamp] = avg_metric
            self.worst_time_accuracies[timestamp] = worst_metric
            self.best_time_accuracies[timestamp] = best_metric

            self.logger.info("acc of next {} domains: \t {}".format(self.eval_next_timestamps, all_metrics))
            self.logger.info("avg acc of next {} domains  : \t {:.3f}".format(self.eval_next_timestamps, avg_metric))
            self.logger.info("worst acc of next {} domains: \t {:.3f}".format(self.eval_next_timestamps, worst_metric))

        for key, value in self.task_accuracies.items():
             self.logger.info("timestamp {} : avg acc = \t {}".format(key, value))

        for key, value in self.worst_time_accuracies.items():
             self.logger.info("timestamp {} : worst acc = \t {}".format(key, value))

        self.logger.info("\naverage of avg acc list: \t {:.3f}".format(np.array(list(self.task_accuracies.values())).mean()))
        self.logger.info("average of worst acc list: \t {:.3f}".format(np.array(list(self.worst_time_accuracies.values())).mean()))

        import csv
        with open(self.args.log_dir+'/avg_acc.csv', 'w', newline='') as file:
            content = {}
            content.update({"method": self.args.method})
            content.update(self.task_accuracies)
            writer = csv.DictWriter(file, fieldnames=list(content.keys()))
            writer.writeheader()
            writer.writerow(content)
        with open(self.args.log_dir+'/worst_acc.csv', 'w', newline='') as file:
            content = {}
            content.update({"method": self.args.method})
            content.update(self.worst_time_accuracies)
            writer = csv.DictWriter(file, fieldnames=list(content.keys()))
            writer.writeheader()
            writer.writerow(content)

    def run(self):
        torch.cuda.empty_cache()
        start_time = time.time()
        if self.args.eval_fix:
            self.run_eval_fix()
        else:
            self.run_eval_stream()
        runtime = time.time() - start_time
        runtime = runtime / 60 / 60
        self.logger.info(f'Runtime: {runtime:.2f} h\n')
