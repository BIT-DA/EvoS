import numpy as np
import time
import datetime
import torch.utils.data

from ..base_trainer import BaseTrainer
from ..dataloaders import InfiniteDataLoader, FastDataLoader
from ..utils import prepare_data, forward_pass, MetricLogger
from ..dataloaders import FastDataLoader

class SI(BaseTrainer):
    """
    Synaptic Intelligence

    Original paper:
        @inproceedings{zenke2017continual,
            title={Continual learning through synaptic intelligence},
            author={Zenke, Friedemann and Poole, Ben and Ganguli, Surya},
            booktitle={International Conference on Machine Learning},
            pages={3987--3995},
            year={2017},
            organization={PMLR}
        }

    Code adapted from https://github.com/GMvandeVen/continual-learning.
    """
    def __init__(self, args, logger, dataset, network, criterion, optimizer, scheduler):
        super().__init__(args, logger, dataset, network, criterion, optimizer, scheduler)
        self.si_c = args.si_c            #-> hyperparam: how strong to weigh SI-loss ("regularisation strength")
        self.epsilon = args.epsilon      #-> dampening parameter: bounds 'omega' when squared parameter-change goes to 0

    def __str__(self):
        str_all = f'SI-si_c={self.si_c}-epsilon={self.epsilon}-{self.base_trainer_str}'
        return str_all

    def _device(self):
        return next(self.network.parameters()).device

    def _is_on_cuda(self):
        return next(self.network.parameters()).is_cuda

    def update_omega(self, W, epsilon):
        '''After completing training on a task, update the per-parameter regularization strength.
        [W]         <dict> estimated parameter-specific contribution to changes in total loss of completed task
        [epsilon]   <float> dampening parameter (to bound [omega] when [p_change] goes to 0)'''

        # Loop over all parameters
        for n, p in self.network.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')

            # Find/calculate new values for quadratic penalty on parameters
            p_prev = getattr(self.network, '{}_SI_prev_task'.format(n))
            p_current = p.detach().clone()
            p_change = p_current - p_prev
            omega_add = W[n] / (p_change ** 2 + epsilon)
            try:
                omega = getattr(self.network, '{}_SI_omega'.format(n))
            except AttributeError:
                omega = p.detach().clone().zero_()
            omega_new = omega + omega_add

            # Store these new values in the model
            self.network.register_buffer('{}_SI_prev_task'.format(n), p_current)
            self.network.register_buffer('{}_SI_omega'.format(n), omega_new)

    def surrogate_loss(self):
        """
        Calculate SI's surrogate loss.
        """
        try:
            losses = []
            for n, p in self.network.named_parameters():
                if p.requires_grad:
                    # Retrieve previous parameter values and their normalized path integral (i.e., omega)
                    n = n.replace('.', '__')
                    prev_values = getattr(self.network, '{}_SI_prev_task'.format(n))
                    omega = getattr(self.network, '{}_SI_omega'.format(n))
                    # Calculate SI's surrogate loss, sum over all parameters
                    losses.append((omega * (p - prev_values) ** 2).sum())
            return sum(losses)

        except AttributeError:
            # SI-loss is 0 if there is no stored omega yet
            return torch.tensor(0., device=self._device())

    def train_step(self, dataloader):
        # Prepare <dicts> to store running importance estimates and parameter-values before update
        W = {}
        p_old = {}
        for n, p in self.network.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                W[n] = p.data.clone().zero_()
                p_old[n] = p.data.clone()

        self.network.train()
        loss_all = []
        self.logger.info("-------------------start training on timestamp {}-------------------".format(self.train_dataset.current_time))
        meters = MetricLogger(delimiter="  ")
        end = time.time()
        self.logger.info("self.train_dataset.len = {} x {} = {} samples".format(self.train_dataset.__len__() // self.args.mini_batch_size, self.args.mini_batch_size, self.train_dataset.__len__()))
        stop_iters = self.args.epochs * (self.train_dataset.__len__() // self.args.mini_batch_size) - 1
        for step, (x, y) in enumerate(dataloader):
            x, y = prepare_data(x, y, str(self.train_dataset))
            loss, logits, y = forward_pass(x, y, self.train_dataset, self.network, self.criterion, self.lisa, self.mixup,
                                           self.cut_mix, self.mix_alpha)
            loss = loss + self.si_c * self.surrogate_loss()
            loss_all.append(loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if step == stop_iters:
                if self.scheduler is not None:
                    self.scheduler.step()

                # Update running parameter importance estimates in W
                for n, p in self.network.named_parameters():
                    if p.requires_grad:
                        # n = "network." + n
                        n = n.replace('.', '__')
                        if p.grad is not None:
                            W[n].add_(-p.grad * (p.detach() - p_old[n]))
                        p_old[n] = p.detach().clone()
                self.update_omega(W, self.epsilon)
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

    def train_online(self):
        # Register starting param-values (needed for "intelligent synapses").
        for n, p in self.network.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                self.network.register_buffer('{}_SI_prev_task'.format(n), p.detach().clone())

        for i, t in enumerate(self.train_dataset.ENV[:-1]):
            if self.args.eval_fix and t == (self.split_time + 1):
                break
            else:
                self.train_dataset.update_current_timestamp(t)
                train_dataloader = InfiniteDataLoader(dataset=self.train_dataset, weights=None, batch_size=self.mini_batch_size,
                                                 num_workers=self.num_workers, collate_fn=self.train_collate_fn)
                self.train_step(train_dataloader)
                self.eval_dataset.mode = 1
                self.eval_dataset.update_current_timestamp(t)
                test_id_dataloader = FastDataLoader(dataset=self.eval_dataset,
                                                     batch_size=self.mini_batch_size,
                                                     num_workers=self.num_workers, collate_fn=self.eval_collate_fn)
                acc = self.network_evaluation(test_id_dataloader)
                self.logger.info("ID timestamp = {}: \t {} is {:.3f}".format(t, self.eval_metric, acc * 100.0))

    def run_eval_stream(self):
        print('==========================================================================================')
        print("Running Eval-Stream...\n")
        self.train_dataset.mode = 0
        end = len(self.eval_dataset.ENV) - self.eval_next_timestamps

        # Register starting param-values (needed for "intelligent synapses").
        for n, p in self.network.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                self.network.register_buffer('{}_SI_prev_task'.format(n), p.detach().clone())

        for i, t in enumerate(self.train_dataset.ENV[:end]):
            self.train_dataset.update_current_timestamp(t)
            train_dataloader = InfiniteDataLoader(dataset=self.train_dataset, weights=None,
                                                      batch_size=self.mini_batch_size,
                                                      num_workers=self.num_workers, collate_fn=self.train_collate_fn)
            self.train_step(train_dataloader)

            # -------evaluate on the testing set of current domain-------
            self.eval_dataset.mode = 1
            self.eval_dataset.update_current_timestamp(t)
            test_id_dataloader = FastDataLoader(dataset=self.eval_dataset,
                                                    batch_size=self.mini_batch_size,
                                                    num_workers=self.num_workers, collate_fn=self.eval_collate_fn)
            acc = self.network_evaluation(test_id_dataloader)
            self.logger.info("ID timestamp = {}: \t {} is {:.3f}".format(t, self.eval_metric, acc * 100.0))

            # -------evaluate on the next K domains-------
            avg_metric, worst_metric, best_metric, all_metrics = self.evaluate_stream(i + 1)
            self.task_accuracies[t] = avg_metric
            self.worst_time_accuracies[t] = worst_metric
            self.best_time_accuracies[t] = best_metric

            self.logger.info("acc of next {} domains: \t {}".format(self.eval_next_timestamps, all_metrics))
            self.logger.info("avg acc of next {} domains  : \t {:.3f}".format(self.eval_next_timestamps, avg_metric))
            self.logger.info( "worst acc of next {} domains: \t {:.3f}".format(self.eval_next_timestamps, worst_metric))

        for key, value in self.task_accuracies.items():
            self.logger.info("timestamp {} : avg acc = \t {}".format(key + self.args.init_timestamp, value))

        for key, value in self.worst_time_accuracies.items():
            self.logger.info("timestamp {} : worst acc = \t {}".format(key + self.args.init_timestamp, value))

        self.logger.info("\naverage of avg acc list: \t {:.3f}".format(np.array(list(self.task_accuracies.values())).mean()))
        self.logger.info("average of worst acc list: \t {:.3f}".format(np.array(list(self.worst_time_accuracies.values())).mean()))

        import csv
        with open(self.args.log_dir + '/avg_acc.csv', 'w', newline='') as file:
            content = {}
            content.update({"method": self.args.method})
            content.update(self.task_accuracies)
            writer = csv.DictWriter(file, fieldnames=list(content.keys()))
            writer.writeheader()
            writer.writerow(content)
        with open(self.args.log_dir + '/worst_acc.csv', 'w', newline='') as file:
            content = {}
            content.update({"method": self.args.method})
            content.update(self.worst_time_accuracies)
            writer = csv.DictWriter(file, fieldnames=list(content.keys()))
            writer.writeheader()
            writer.writerow(content)
