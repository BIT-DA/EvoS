from ..base_trainer import BaseTrainer
from .trans_dataset import TransDataset
from copy import deepcopy


class CDOT(BaseTrainer):
    def __init__(self, args, logger, dataset, network, criterion, optimizer, scheduler):
        super(CDOT, self).__init__(args, logger, dataset, network, criterion, optimizer, scheduler)
        self.train_dataset = TransDataset(args, dataset)
        self.eval_dataset = deepcopy(self.train_dataset)

    def run_eval_fix(self):
        print('==========================================================================================')
        print("Running Eval-Fix...\n")
        self.train_offline()
        self.evaluate_offline()

    def __str__(self):
        return "CDOT"



