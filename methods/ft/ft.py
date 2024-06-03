import os

from ..base_trainer import BaseTrainer


class FT(BaseTrainer):
    """
    Fine-tuning
    """
    def __init__(self, args, logger, dataset, network, criterion, optimizer, scheduler):
        super().__init__(args, logger, dataset, network, criterion, optimizer, scheduler)
        self.args = args
        self.K = args.K

    def __str__(self):
        return f'FT-K={self.K}-{self.base_trainer_str}'
