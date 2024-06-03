import numpy as np
import torch

from torch.utils.data import Dataset
from data.rmnist import RotatedMNIST
from data.yearbook import Yearbook
from typing import Union


class WrappedDataset(Dataset):
    def __init__(self, args, dataset: Union[Yearbook, RotatedMNIST]):
        self.args = args
        self.ENV = dataset.ENV

        train_list = []
        for i, env in enumerate(self.ENV):
            if env <= self.args.split_time:
                train_list.append(i)
            else:
                break

        self.train_list = train_list

        self.datasets = self.process_datasets(dataset)

        self.args = args
        self.num_classes = dataset.num_classes

        self.current_time = 0
        self.mini_batch_size = args.mini_batch_size
        self.mode = 0

        self.ENV = list(sorted(self.datasets.keys()))
        self.num_tasks = len(self.ENV)
        self.num_examples = {i: len(self.datasets[i][self.mode]['labels']) for i in self.ENV}
        self.ds_name = str(dataset) if str(dataset) is not None else "TransDataset"

    def process_datasets(self, dataset: Union[Yearbook, RotatedMNIST]):
        train_val_split = []
        all_data = []
        all_label = []

        ENV_range = dataset.ENV

        for env in ENV_range:
            env_data = []
            env_label = []
            dataset.update_current_timestamp(env)
            dataset.mode = 0
            len_train = len(dataset)
            for i in range(len_train):
                i_data, i_label = dataset[i]
                env_data.append(i_data)
                env_label.append(i_label)

            dataset.mode = 1
            len_val = len(dataset)
            for i in range(len_val):
                i_data, i_label = dataset[i]
                env_data.append(i_data)
                env_label.append(i_label.item())

            env_data = torch.stack(env_data, dim=0).numpy()  # (N, ...)
            env_label = torch.FloatTensor(env_label).numpy()  # (N)

            shape = env_data.shape
            all_data.append(env_data)  # (N, d)
            all_label.append(env_label)

            train_val_split.append(len_train)

        new_dataset = {}
        for env, ENV in enumerate(ENV_range):
            data_env = all_data[env]
            label_env = all_label[env]
            norm_data_env = data_env - np.mean(data_env, axis=0, keepdims=True)
            domain_env = np.array([1] * len(label_env))
            mask_env = np.array([1] * len(label_env))

            train_data = data_env[:train_val_split[env]]
            train_label = label_env[:train_val_split[env]]
            train_norm_data = norm_data_env[:train_val_split[env]]
            train_domain = np.array([env] * len(train_label))
            train_mask = np.array([1] * len(train_label))

            val_data = data_env[train_val_split[env]:]
            val_label = label_env[train_val_split[env]:]
            val_norm_data = norm_data_env[train_val_split[env]:]
            val_domain = np.array([env] * len(val_label))
            val_mask = np.array([1] * len(val_label))

            train_dict = {"row_data": train_data, "labels": train_label,
                          "data": train_norm_data, "domain": train_domain, "mask": train_mask}

            val_dict = {"row_data": val_data, "labels": val_label,
                          "data": val_norm_data, "domain": val_domain, "mask": val_mask}

            all_dict = {"row_data": data_env, "labels": label_env,
                          "data": norm_data_env, "domain": domain_env, "mask": mask_env}

            new_dataset[ENV] = {0: train_dict, 1: val_dict, 2: all_dict}

        return new_dataset

    def update_historical(self, idx, data_del=False):
        time = self.ENV[idx]
        prev_time = self.ENV[idx - 1]

        keys = ["row_data", "labels", "data", "domain", "mask"]
        for key in keys:
            self.datasets[time][self.mode][key] = \
                np.concatenate((self.datasets[prev_time][self.mode][key], self.datasets[time][self.mode][key]), axis=0)

        if data_del:
            del self.datasets[prev_time]

    def update_current_timestamp(self, time):
        self.current_time = time

    def __getitem__(self, index):
        row_data = self.datasets[self.current_time][self.mode]['row_data'][index]
        data = self.datasets[self.current_time][self.mode]['data'][index]
        label = self.datasets[self.current_time][self.mode]['labels'][index]
        domain = self.datasets[self.current_time][self.mode]['domain'][index]
        mask = self.datasets[self.current_time][self.mode]['mask'][index]
        row_data_tensor = torch.FloatTensor(row_data)
        data_tensor = torch.FloatTensor(data)

        return row_data_tensor, label, float(domain), data_tensor, mask

    def __len__(self):
        return len(self.datasets[self.current_time][self.mode]['labels'])

    def __str__(self):
        return self.ds_name


