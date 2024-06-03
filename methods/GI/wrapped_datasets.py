import torch
from torch.utils.data import Dataset
from data.arxiv import ArXiv
from data.fmow import FMoW
from data.huffpost import HuffPost
from data.rmnist import RotatedMNIST
from data.yearbook import Yearbook
from typing import Union
import numpy as np


class WrappedDataset(Dataset):
    def __init__(self, dataset: Union[ArXiv, FMoW, HuffPost, Yearbook, RotatedMNIST], split_time):
        super().__init__()
        self.dataset = dataset
        self.min = min(dataset.ENV)
        self.max = max(dataset.ENV)
        assert self.max != self.min, "single domain!!!!"
        self.spilt_time = split_time

        self.add_time_to_ds()

        self.mix_all = False

        self.cumulate_size_info = self.compute_size_info_train()

    def mix_all_data(self):
        self.mix_all = True

    def use_as_time_sequence(self):
        self.mix_all = False

    def add_time_to_ds(self):
        for time_step in self.dataset.ENV:
            self.dataset.update_current_timestamp(time_step)
            for mode in [0, 1, 2]:
                self.dataset.mode = mode
                len_ds = len(self.dataset)
                mode_dict = self.dataset.datasets[self.dataset.current_time][self.dataset.mode]
                mode_dict.setdefault("time_step", [(time_step - self.min) / (self.max - self.min) * 0.9 + 0.1] * len_ds)

    def __getitem__(self, item):
        if self.mix_all:
            time_step, item = self.map_idx_to_time_idx(item)
            self.update_current_timestamp(time_step)

        time_step = self.dataset.datasets[self.dataset.current_time][self.dataset.mode]["time_step"][item]
        x, y = self.dataset[item]

        return x, y, torch.FloatTensor([time_step])

    def update_current_timestamp(self, time):
        self.dataset.update_current_timestamp(time)

    def compute_size_info_train(self):
        size_info = [[], [], []]
        for time_step in self.dataset.ENV:
            if time_step == self.spilt_time + 1:
                break
            self.dataset.update_current_timestamp(time_step)
            for mode in [0, 1, 2]:
                self.dataset.mode = mode
                size_info[mode].append(len(self.dataset))

        cumulate_size_info = [[], [], []]
        for mode in [0, 1, 2]:
            for i in range(len(size_info[mode]) + 1):
                cumulate_size_info[mode].append(sum(size_info[mode][0:i]))

        return cumulate_size_info

    def map_idx_to_time_idx(self, idx):
        mode = self.dataset.mode
        time_idx = None
        for i in range(len(self.cumulate_size_info[mode]) - 1):
            if self.cumulate_size_info[mode][i] <= idx < self.cumulate_size_info[mode][i + 1]:
                time_idx = i
                break

        left_offset = idx - self.cumulate_size_info[mode][time_idx]
        time_step = self.dataset.ENV[time_idx]
        return time_step, left_offset

    @property
    def mode(self):
        return self.dataset.mode

    @mode.setter
    def mode(self, value):
        self.dataset.mode = value

    @property
    def ENV(self):
        return self.dataset.ENV

    def __len__(self):
        if self.mix_all:
            return self.cumulate_size_info[self.mode][-1]
        else:
            return len(self.dataset)

    @property
    def num_tasks(self):
        return self.dataset.num_tasks

    @property
    def num_classes(self):
        return self.dataset.num_classes

    def __str__(self):
        return str(self.dataset)

    def update_historical(self, idx, data_del):
        self.dataset.update_historical(idx, False)
        time = self.dataset.ENV[idx]
        prev_time = self.dataset.ENV[idx - 1]

        self.dataset.datasets[time][self.mode]['time_step'] = np.concatenate(
            (self.dataset.datasets[prev_time][self.mode]['time_step'],
             self.dataset.datasets[time][self.mode]['time_step']),
            axis=0)

        if data_del:
            del self.dataset.datasets[prev_time]
