import torch
from collections import defaultdict
from collections import deque
from lightly.data import SimCLRCollateFunction, SwaVCollateFunction
from torch.autograd import Variable
from typing import Optional
from torch.optim.optimizer import Optimizer

from .lisa import lisa
from .mixup import mixup_data, mixup_criterion
from .rmnist_collate_function import rmnist_SimCLRCollateFunction, rmnist_SwaVCollateFunction






def prepare_data(x, y, dataset_name: str):
    if dataset_name in ['arxiv', 'huffpost']:
        x = x.to(dtype=torch.int64).cuda()
        if len(y.shape) > 1:
            y = y.squeeze(1).cuda()
    elif dataset_name in ['fmow', 'yearbook', 'rmnist']:
        if isinstance(x, tuple):
            x = (elt.cuda() for elt in x)
        else:
            x = x.cuda()
        if len(y.shape) > 1:
            y = y.squeeze(1).cuda()
        else:
            y = y.cuda()
    else:
        x = x.cuda()
        if len(y.shape) > 1:
            y = y.squeeze(1).cuda()
        else:
            y = y.cuda()
    return x, y


def forward_pass(x, y, dataset, network, criterion, use_lisa: bool, use_mixup: bool, cut_mix: bool, mix_alpha=2.0):
    if use_lisa:
        if str(dataset) in ['arxiv', 'huffpost']:
            x = network.model[0](x)
            sel_x, sel_y = lisa(x, y, dataset=dataset, mix_alpha=mix_alpha,
                                num_classes=dataset.num_classes, time_idx=dataset.current_time,
                                cut_mix=cut_mix, embedding=network.model[0])
            logits = network.model[1](sel_x)
        else:
            sel_x, sel_y = lisa(x, y, dataset=dataset, mix_alpha=mix_alpha,
                                num_classes=dataset.num_classes, time_idx=dataset.current_time,
                                cut_mix=cut_mix)
            logits = network(sel_x)
        y = torch.argmax(sel_y, dim=1)
        loss = criterion(logits, y)

    elif use_mixup:
        if str(dataset) in ['arxiv', 'huffpost']:
            x = network.model[0](x)
            x, y_a, y_b, lam = mixup_data(x, y, mix_alpha=mix_alpha)
            logits = network.model[1](x)
        else:
            x, y_a, y_b, lam = mixup_data(x, y, mix_alpha=mix_alpha)
            x, y_a, y_b = map(Variable, (x, y_a, y_b))
            logits = network(x)
        loss = mixup_criterion(criterion, logits, y_a, y_b, lam)

    else:
        logits = network(x)
        if str(dataset) in ['arxiv', 'fmow', 'huffpost', 'yearbook']:
            if len(y.shape) > 1:
                y = y.squeeze(1)
        loss = criterion(logits, y)

    return loss, logits, y


def split_into_groups(g):
    """
    From https://github.com/p-lambda/wilds/blob/f384c21c67ee58ab527d8868f6197e67c24764d4/wilds/common/utils.py#L40.
    Args:
        - g (Tensor): Vector of groups
    Returns:
        - groups (Tensor): Unique groups present in g
        - group_indices (list): List of Tensors, where the i-th tensor is the indices of the
                                elements of g that equal groups[i].
                                Has the same length as len(groups).
        - unique_counts (Tensor): Counts of each element in groups.
                                 Has the same length as len(groups).
    """
    unique_groups, unique_counts = torch.unique(g, sorted=False, return_counts=True)
    group_indices = []
    for group in unique_groups:
        group_indices.append(
            torch.nonzero(g == group, as_tuple=True)[0])
    return unique_groups, group_indices, unique_counts


def get_collate_functions(args, train_dataset):
    if args.method == 'simclr':
        if args.dataset == 'yearbook':
            train_collate_fn = SimCLRCollateFunction(
                input_size=train_dataset.resolution,
                vf_prob=0.5,
                rr_prob=0.5
            )
        elif args.dataset == 'rmnist':
            train_collate_fn = rmnist_SimCLRCollateFunction(
                input_size=train_dataset.resolution,
                vf_prob=0.5,
                rr_prob=0.5
            )
        else:
            train_collate_fn = SimCLRCollateFunction(
                input_size=train_dataset.resolution
            )
        eval_collate_fn = None
    elif args.method == 'swav':
        if args.dataset == 'rmnist':
            train_collate_fn = rmnist_SwaVCollateFunction()
        else:
            train_collate_fn = SwaVCollateFunction()
        eval_collate_fn = None
    else:
        train_collate_fn = None
        eval_collate_fn = None

    return train_collate_fn, eval_collate_fn



class SmoothedValue(object):
    def __init__(self, window_size=20):
        self.deque = deque(maxlen=window_size)
        self.series = []
        self.total = 0.0
        self.count = 0

    def update(self, value):
        self.deque.append(value)
        self.series.append(value)
        self.count += 1
        self.total += value

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque))
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
                    type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {:.8f} ({:.8f})".format(name, meter.series[meter.count - 1], meter.global_avg)
            )
        return self.delimiter.join(loss_str)




class StepwiseLR:
    """
    A lr_scheduler that update learning rate using the following schedule:

    .. math::
        \text{lr} = \text{init_lr} \times \text{lr_mult} \times (1+\gamma i)^{-p},

    where `i` is the iteration steps.

    Parameters:
        - **optimizer**: Optimizer
        - **init_lr** (float, optional): initial learning rate. Default: 0.01
        - **gamma** (float, optional): :math:`\gamma`. Default: 0.001
        - **decay_rate** (float, optional): :math:`p` . Default: 0.75
    """
    def __init__(self, optimizer: Optimizer, init_lr: Optional[float] = 0.01,
                 gamma: Optional[float] = 0.001, max_iters = None, decay_rate: Optional[float] = 0.75):
        self.init_lr = init_lr
        self.gamma = gamma
        self.decay_rate = decay_rate
        self.optimizer = optimizer
        self.iter_num = 0
        self.max_iters = max_iters

    def get_lr(self) -> float:
        lr = self.init_lr * (1 + self.gamma * self.iter_num) ** (-self.decay_rate)
        # lr = self.init_lr * (1 + self.gamma * (self.iter_num / self.max_iters)) ** (-self.decay_rate)
        return lr

    def step(self):
        """Increase iteration number `i` by 1 and update learning rate in `optimizer`"""
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            if 'lr_mult' not in param_group:
                param_group['lr_mult'] = 1.
            param_group['lr'] = lr * param_group['lr_mult']

        self.iter_num += 1

    def reset(self):
        self.iter_num = 0



