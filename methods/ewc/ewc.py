import copy
import time
import datetime
import torch
import torch.utils.data
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torchvision import transforms

from ..base_trainer import BaseTrainer
from ..utils import prepare_data, forward_pass, MetricLogger
from ..dataloaders import FastDataLoader

class EWC(BaseTrainer):
    """
    Elastic Weight Consolidation

    Original paper:
        @article{kirkpatrick2017overcoming,
            title={Overcoming catastrophic forgetting in neural networks},
            author={Kirkpatrick, James and Pascanu, Razvan and Rabinowitz, Neil and Veness, Joel and Desjardins, Guillaume and Rusu, Andrei A and Milan, Kieran and Quan, John and Ramalho, Tiago and Grabska-Barwinska, Agnieszka and others},
            journal={Proceedings of the national academy of sciences},
            volume={114},
            number={13},
            pages={3521--3526},
            year={2017},
            publisher={National Acad Sciences}
        }

    Code adapted from https://github.com/GMvandeVen/continual-learning.
    """
    def __init__(self, args, logger, dataset, network, criterion, optimizer, scheduler):
        super().__init__(args, logger, dataset, network, criterion, optimizer, scheduler)
        self.ewc_lambda = args.ewc_lambda   #-> hyperparam: how strong to weigh EWC-loss ("regularization strength")
        self.gamma = args.gamma             #-> hyperparam (online EWC): decay-term for old tasks' contribution to quadratic term
        self.online = args.online           #-> "online" (=single quadratic term) or "offline" (=quadratic term per task) EWC
        self.fisher_n = args.fisher_n       #-> sample size for estimating FI-matrix (if "None", full pass over dataset)
        self.emp_FI = args.emp_FI           #-> if True, use provided labels to calculate FI ("empirical FI"); else predicted labels
        self.EWC_task_count = 0             #-> keeps track of number of quadratic loss terms (for "offline EWC")

    def __str__(self):
        str_all = f'EWC-lambda={self.ewc_lambda}-gamma={self.gamma}-online={self.online}-fisher_n={self.fisher_n}' \
                  f'-emp_FI={self.emp_FI}-{self.base_trainer_str}'
        return str_all

    def _device(self):
        return next(self.network.parameters()).device

    def _is_on_cuda(self):
        return next(self.network.parameters()).is_cuda

    def estimate_fisher(self):
        """
        After completing training on a task, estimate diagonal of Fisher Information matrix.
        [dataset]:          <DataSet> to be used to estimate FI-matrix
        """
        est_fisher_info = {}
        for n, p in self.network.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                est_fisher_info[n] = p.detach().clone().zero_()

        self.network.eval()

        data_loader = get_data_loader(self.train_dataset, batch_size=self.mini_batch_size, collate_fn=self.train_collate_fn)

        ind = 0
        for index, (x, y) in enumerate(data_loader):
            if self.fisher_n is not None:
                if index >= self.fisher_n:
                    break
            x, y = prepare_data(x, y, str(self.train_dataset))
            loss, output, y = forward_pass(x, y, self.train_dataset, self.network, self.criterion, self.lisa, self.mixup, self.cut_mix, self.mix_alpha)
            if self.emp_FI:
                label = torch.LongTensor([y]) if type(y) == int else y
                label = label.to(self._device())
                negloglikelihood = F.nll_loss(F.log_softmax(output, dim=1), label)
            else:
                label = output.max(1)[1]
                negloglikelihood = F.nll_loss(F.log_softmax(output, dim=1), label)

            self.network.zero_grad()
            negloglikelihood.backward()

            for n, p in self.network.named_parameters():
                if p.requires_grad:
                    n = n.replace('.', '__')
                    if p.grad is not None:
                        est_fisher_info[n] += p.grad.detach() ** 2
            ind = index

        est_fisher_info = {n: p / (ind + 1) for n, p in est_fisher_info.items()}

        for n, p in self.network.named_parameters():
            if p.requires_grad:
                n = n.replace('.', '__')
                self.network.register_buffer('{}_EWC_prev_task{}'.format(n, "" if self.online else self.EWC_task_count + 1),
                                     p.detach().clone())
                if self.online and self.EWC_task_count == 1:
                    existing_values = getattr(self.network, '{}_EWC_estimated_fisher'.format(n))
                    est_fisher_info[n] += self.gamma * existing_values
                self.network.register_buffer(
                    '{}_EWC_estimated_fisher{}'.format(n, "" if self.online else self.EWC_task_count + 1),
                    est_fisher_info[n])

        self.EWC_task_count = 1 if self.online else self.EWC_task_count + 1

        self.network.train()

    def ewc_loss(self):
        if self.EWC_task_count > 0:
            losses = []
            for task in range(1, self.EWC_task_count + 1):
                for n, p in self.network.named_parameters():
                    if p.requires_grad:
                        n = n.replace('.', '__')
                        mean = getattr(self.network, '{}_EWC_prev_task{}'.format(n, "" if self.online else task))
                        fisher = getattr(self.network, '{}_EWC_estimated_fisher{}'.format(n, "" if self.online else task))
                        fisher = self.gamma * fisher if self.online else fisher
                        losses.append((fisher * (p - mean) ** 2).sum())
            return (1. / 2) * sum(losses)
        else:
            return torch.tensor(0., device=self._device())

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
            loss, logits, y = forward_pass(x, y, self.train_dataset, self.network, self.criterion, self.lisa, self.mixup, self.cut_mix, self.mix_alpha)
            loss = loss + self.ewc_lambda * self.ewc_loss()
            loss_all.append(loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if step == stop_iters:
                if self.scheduler is not None:
                    self.scheduler.step()
                self.estimate_fisher()
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


def get_data_loader(dataset, batch_size, cuda=False, collate_fn=None, drop_last=False, augment=False):
    """
    Return <DataLoader>-object for the provided <DataSet>-object [dataset].
    """
    if augment:
        dataset_ = copy.deepcopy(dataset)
        dataset_.transform = transforms.Compose([dataset.transform, *data.AVAILABLE_TRANSFORMS['augment']])
    else:
        dataset_ = dataset

    rand_sampler = torch.utils.data.RandomSampler(dataset, replacement=True, num_samples=min(batch_size, len(dataset_)))
    return DataLoader(
        dataset_, sampler=rand_sampler,
        collate_fn=(collate_fn or default_collate), drop_last=drop_last,
        **({'num_workers': 2, 'pin_memory': True} if cuda else {})
    )