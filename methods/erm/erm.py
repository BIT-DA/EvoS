import os

from ..base_trainer import BaseTrainer


class ERM(BaseTrainer):
    """
    Empirical Risk Minimization
    """
    def __init__(self, args, logger, dataset, network, criterion, optimizer, scheduler):
        super().__init__(args, logger, dataset, network, criterion, optimizer, scheduler)

    def __str__(self):
        if self.args.lisa:
            return f'ERM-LISA-no-domainid-{self.base_trainer_str}'
        elif self.args.mixup:
            return f'ERM-Mixup-no-domainid-{self.base_trainer_str}'
        return f'ERM-{self.base_trainer_str}'