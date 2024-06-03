import numpy as np
import torch

from torch.utils.data import Dataset
from data.rmnist import RotatedMNIST
from data.yearbook import Yearbook
from typing import Union

from .regularized_ot import RegularizedSinkhornTransportOTDA


class TransDataset(Dataset):
    def __init__(self, args, dataset: Union[Yearbook, RotatedMNIST]):
        self.args = args
        self.ENV = dataset.ENV
        self.datasets = self.transported_datasets(dataset)

        self.args = args
        self.num_classes = dataset.num_classes

        self.current_time = 0
        self.mini_batch_size = args.mini_batch_size
        self.mode = 0

        self.ENV = list(sorted(self.datasets.keys()))
        self.num_tasks = len(self.ENV)
        self.num_examples = {i: len(self.datasets[i][self.mode]['labels']) for i in self.ENV}
        self.ds_name = str(dataset) if str(dataset) is not None else "TransDataset"

    def transported_datasets(self, dataset: Union[Yearbook, RotatedMNIST]):
        train_val_split = []
        all_data = []
        all_label = []
        all_shape = []

        all_transform_data = []
        all_transform_label = []
        future_data = []
        future_label = []

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

            if env > self.args.split_time:
                future_data.extend(env_data)
                future_label.extend(env_label)

            env_data = torch.stack(env_data, dim=0).numpy()  # (N, ...)
            env_label = torch.FloatTensor(env_label).numpy()  # (N)

            shape = env_data.shape
            all_shape.append(shape)
            all_data.append(env_data.reshape(shape[0], -1))  # (N, d)
            all_label.append(env_label)

            if env <= self.args.split_time:
                all_transform_data.append(env_data.reshape(shape[0], -1)[:len_train])  # (N, d)
                all_transform_label.append(env_label[:len_train])
            else:
                if env == ENV_range[-1]:
                    future_data = torch.stack(future_data, dim=0).numpy()  # (N, ...)
                    future_label = torch.FloatTensor(future_label).numpy()  # (N)
                    all_transform_data.append(future_data.reshape(future_data.shape[0], -1))  # (N, d)
                    all_transform_label.append(future_label)

            train_val_split.append(len_train)

            all_transform_data = transform_samples_reg_otda(all_transform_data, all_transform_label)

        new_dataset = {}
        for env, ENV in enumerate(ENV_range):
            if env <= self.args.split_time:
                train_dict = {"data": all_transform_data[env][:train_val_split[env]].reshape(-1, *all_shape[env][1:]),
                              "labels": all_transform_label[env][:train_val_split[env]]}
            else:
                train_dict = {"data": all_data[env][:train_val_split[env]].reshape(-1, *all_shape[env][1:]),
                              "labels": all_label[env][:train_val_split[env]]}
            val_dict = {"data": all_data[env][train_val_split[env]:].reshape(-1, *all_shape[env][1:]),
                        "labels": all_label[env][train_val_split[env]:]}
            all_dict = {"data": all_data[env].reshape(*all_shape[env]), "labels": all_label[env]}

            new_dataset[ENV] = {0: train_dict, 1: val_dict, 2: all_dict}

        return new_dataset


    def update_historical(self, idx, data_del=False):
        time = self.ENV[idx]
        prev_time = self.ENV[idx - 1]
        self.datasets[time][self.mode]['data'] = np.concatenate(
            (self.datasets[prev_time][self.mode]['data'], self.datasets[time][self.mode]['data']), axis=0)
        self.datasets[time][self.mode]['labels'] = np.concatenate(
            (self.datasets[prev_time][self.mode]['labels'], self.datasets[time][self.mode]['labels']), axis=0)
        if data_del:
            del self.datasets[prev_time]

    def update_current_timestamp(self, time):
        self.current_time = time

    def __getitem__(self, index):
        data = self.datasets[self.current_time][self.mode]['data'][index]
        label = self.datasets[self.current_time][self.mode]['labels'][index]
        data_tensor = torch.FloatTensor(data)
        label_tensor = torch.LongTensor([label])
        return data_tensor, label_tensor

    def __len__(self):
        return len(self.datasets[self.current_time][self.mode]['labels'])

    def __str__(self):
        return self.ds_name


def transform_samples_reg_otda(X_domain, Y_domain):
    ot_sinkhorn_r = RegularizedSinkhornTransportOTDA(reg_e=0.5, max_iter=50, norm="median", verbose=False)

    for i in range(len(X_domain) - 1):
        print('-' * 100)

        gamma = []
        X_temp = X_domain[i]
        Y_temp = Y_domain[i]

        for j in range(i + 1, len(X_domain) - 1):
            print(f'Transforming domain {i} to domain {j}')
            if j == i + 1:
                ot_sinkhorn_r.fit(Xs=X_domain[j - 1] + 1e-6, ys=Y_domain[j - 1], Xt=X_domain[j] + 1e-6, yt=Y_domain[j],
                                  Xs_trans=X_temp + 1e-6, ys_trans=Y_domain[i], iteration=0)
            else:
                ot_sinkhorn_r.fit(Xs=X_domain[j - 1] + 1e-6, ys=Y_domain[j - 1], Xt=X_domain[j] + 1e-6, yt=Y_domain[j],
                                  Xs_trans=X_temp + 1e-6, ys_trans=Y_domain[i], prev_gamma=gamma, iteration=1)
            gamma = ot_sinkhorn_r.coupling_
            X_temp = ot_sinkhorn_r.transform(X_temp)
        X_domain[i] = X_temp
    return X_domain