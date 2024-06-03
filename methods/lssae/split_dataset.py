import torch
from torch.utils.data import Dataset
from data.arxiv import ArXiv
from data.fmow import FMoW
from data.huffpost import HuffPost
from data.yearbook import Yearbook
from typing import Union
import numpy as np
from threading import Lock


class SubTimeDataset(Dataset):
    lock = Lock()

    def __init__(self, dataset: Union[ArXiv, FMoW, HuffPost, Yearbook], domain_idx, time):
        self.dataset = dataset
        self.time = time
        self.domain_idx = domain_idx

    def __getitem__(self, item):
        SubTimeDataset.lock.acquire()
        self.dataset.update_current_timestamp(self.time)
        data = self.dataset[item]
        SubTimeDataset.lock.release()
        return data

    @property
    def mode(self):
        return self.dataset.mode

    def __len__(self):
        SubTimeDataset.lock.acquire()
        self.dataset.update_current_timestamp(self.time)
        length = len(self.dataset)
        SubTimeDataset.lock.release()
        return length
