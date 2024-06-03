import torch
import time
import datetime
from ..base_trainer import BaseTrainer
from ..utils import prepare_data, forward_pass, split_into_groups, MetricLogger
from ..dataloaders import FastDataLoader

class DeepCORAL(BaseTrainer):
    """
    Deep CORAL

    This algorithm was originally proposed as an unsupervised domain adaptation algorithm.

    Original paper:
        @inproceedings{sun2016deep,
          title={Deep CORAL: Correlation alignment for deep domain adaptation},
          author={Sun, Baochen and Saenko, Kate},
          booktitle={European Conference on Computer Vision},
          pages={443--450},
          year={2016},
          organization={Springer}
        }

    Code adapted from https://github.com/p-lambda/wilds/blob/main/examples/algorithms/deepCORAL.py.
    """

    def __init__(self, args, logger, dataset, network, criterion, optimizer, scheduler):
        super().__init__(args, logger, dataset, network, criterion, optimizer, scheduler)
        self.coral_lambda = args.coral_lambda

    def __str__(self):
        return f'DeepCORAL-coral_lambda={self.coral_lambda}-{self.base_trainer_str}'

    def coral_penalty(self, x, y):
        if x.dim() > 2:
            x = x.view(-1, x.size(-1))
            y = y.view(-1, y.size(-1))

        mean_x = x.mean(0, keepdim=True)
        mean_y = y.mean(0, keepdim=True)
        cent_x = x - mean_x
        cent_y = y - mean_y
        cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)
        cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)

        mean_diff = (mean_x - mean_y).pow(2).mean()
        cova_diff = (cova_x - cova_y).pow(2).mean()

        return mean_diff + cova_diff

    def train_step(self, dataloader):
        self.network.train()
        loss_all = []
        meters = MetricLogger(delimiter="  ")
        end = time.time()
        self.logger.info("self.train_dataset.len = {} x {} = {} samples".format(
            self.train_dataset.__len__() // self.args.mini_batch_size, self.args.mini_batch_size,
            self.train_dataset.__len__()))
        stop_iters = self.args.epochs * (self.train_dataset.__len__() // self.args.mini_batch_size) - 1
        for step, (x, y, g) in enumerate(dataloader):
            x, y = prepare_data(x, y, str(self.train_dataset))
            g = g.squeeze(1).cuda()
            unique_groups, group_indices, _ = split_into_groups(g)
            n_groups_per_batch = unique_groups.numel()

            classification_loss, logits, y = forward_pass(x, y, self.train_dataset, self.network, self.criterion, self.lisa, self.mixup, self.cut_mix, self.mix_alpha)
            coral_loss = torch.zeros(1).cuda()
            for i_group in range(n_groups_per_batch):
                for j_group in range(i_group + 1, n_groups_per_batch):
                    coral_loss += self.coral_penalty(logits[group_indices[i_group]].squeeze(0), logits[group_indices[j_group]].squeeze(0))
            if n_groups_per_batch > 1:
                coral_loss /= (n_groups_per_batch * (n_groups_per_batch-1) / 2)

            loss = classification_loss + self.coral_lambda * coral_loss
            loss_all.append(loss.item())
            self.optimizer.zero_grad()
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