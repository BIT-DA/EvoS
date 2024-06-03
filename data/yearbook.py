import os
import pickle
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

RESOLUTION = 32
ID_HELD_OUT = 0.1



def preprocess_group(args):
    if not os.path.isfile(os.path.join(args.data_dir, 'yearbook.pkl')):
        raise RuntimeError(
            'Dataset yearbook.pkl is not ready! Please download from https://drive.google.com/u/0/uc?id=1mPpxoX2y2oijOvW1ymiHEYd7oMu2vVRb&export=download\n, and save it as yearbook.pkl')

    print(f'Preprocessing dataset and saving to yearbook_group_{args.yearbook_group_size}.pkl')
    np.random.seed(0)
    orig_data_file = os.path.join(args.data_dir, f'yearbook.pkl')
    dataset = pickle.load(open(orig_data_file, 'rb'))
    years = list(sorted(dataset.keys()))

    group_dataset = {}
    timestamp = 0
    for y in range(years[0], years[-1], args.yearbook_group_size):
        group_dataset[timestamp] = {}
        num_train_samples, num_test_samples, num_all_samples = 0, 0, 0
        print("------------")
        for k in range(0, args.yearbook_group_size):
            print(y+k)
            train_images = dataset[y + k][0]['images']
            train_labels = dataset[y + k][0]['labels']

            test_images = dataset[y + k][1]['images']
            test_labels = dataset[y + k][1]['labels']

            all_images = dataset[y + k][2]['images']
            all_labels = dataset[y + k][2]['labels']

            num_train_samples += len(train_labels)
            num_test_samples += len(test_labels)
            num_all_samples += len(all_labels)
            if k == 0:
                new_train_images = np.array(train_images)
                new_train_labels = np.array(train_labels)

                new_test_images = np.array(test_images)
                new_test_labels = np.array(test_labels)

                new_all_images = np.array(all_images)
                new_all_labels = np.array(all_labels)
            else:
                new_train_images = np.concatenate((new_train_images, np.array(train_images)), axis=0)
                new_train_labels = np.concatenate((new_train_labels, np.array(train_labels)))

                new_test_images = np.concatenate((new_test_images, np.array(test_images)), axis=0)
                new_test_labels = np.concatenate((new_test_labels, np.array(test_labels)))

                new_all_images = np.concatenate((new_all_images, np.array(all_images)), axis=0)
                new_all_labels = np.concatenate((new_all_labels, np.array(all_labels)))
        print("------------")

        group_dataset[timestamp][0] = {}
        group_dataset[timestamp][0]['images'] = new_train_images
        group_dataset[timestamp][0]['labels'] = new_train_labels

        group_dataset[timestamp][1] = {}
        group_dataset[timestamp][1]['images'] = new_test_images
        group_dataset[timestamp][1]['labels'] = new_test_labels

        group_dataset[timestamp][2] = {}
        group_dataset[timestamp][2]['images'] = new_all_images
        group_dataset[timestamp][2]['labels'] = new_all_labels
        timestamp += 1

    preprocessed_data_file = os.path.join(args.data_dir, f'yearbook_group_{args.yearbook_group_size}.pkl')
    pickle.dump(group_dataset, open(preprocessed_data_file, 'wb'))
    np.random.seed(args.random_seed)


def preprocess(args):
    if not os.path.isfile(os.path.join(args.data_dir, 'yearbook.pkl')):
        raise RuntimeError('dataset yearbook.pkl is not yet ready! Please download from   https://drive.google.com/u/0/uc?id=1mPpxoX2y2oijOvW1ymiHEYd7oMu2vVRb&export=download   and save it as yearbook.pkl')


class YearbookBase(Dataset):
    def __init__(self, args):
        super().__init__()
        if args.yearbook_group_size is None:
            self.data_file = f'{str(self)}.pkl'
            preprocess(args)
        else:
            self.data_file = f'{str(self)}_group_{args.yearbook_group_size}.pkl'
            if not os.path.isfile(os.path.join(args.data_dir, self.data_file)):
                preprocess_group(args)

        self.datasets = pickle.load(open(os.path.join(args.data_dir, self.data_file), 'rb'))
        self.args = args
        self.num_classes = 2
        self.current_time = 0
        self.resolution = 32
        self.mini_batch_size = args.mini_batch_size
        self.mode = 0
        self.ssl_training = False

        self.ENV = list(sorted(self.datasets.keys()))
        self.num_tasks = len(self.ENV)
        self.num_examples = {i: len(self.datasets[i][self.mode]['labels']) for i in self.ENV}

        self.class_id_list = {i: {} for i in range(self.num_classes)}
        self.task_idxs = {}
        start_idx = 0
        for i in self.ENV:
            end_idx = start_idx + len(self.datasets[i][self.mode]['labels'])
            self.task_idxs[i] = {}
            self.task_idxs[i][self.mode] = [start_idx, end_idx]
            start_idx = end_idx
            for classid in range(self.num_classes):
                sel_idx = np.nonzero(self.datasets[i][self.mode]['labels'] == classid)[0]
                self.class_id_list[classid][i] = sel_idx

    def update_historical(self, idx, data_del=False):
        time = self.ENV[idx]
        prev_time = self.ENV[idx - 1]
        self.datasets[time][self.mode]['images'] = np.concatenate(
            (self.datasets[prev_time][self.mode]['images'], (self.datasets[time][self.mode]['images'])), axis=0)
        self.datasets[time][self.mode]['labels'] = np.concatenate(
            (self.datasets[prev_time][self.mode]['labels'], (self.datasets[time][self.mode]['labels'])), axis=0)
        if data_del:
            del self.datasets[prev_time]
        for classid in range(self.num_classes):
            sel_idx = np.nonzero(self.datasets[time][self.mode]['labels'] == classid)[0]
            self.class_id_list[classid][time] = sel_idx

    def update_current_timestamp(self, time):
        self.current_time = time

    def get_lisa_new_sample(self, time_idx, classid, num_sample):
        idx_all = self.class_id_list[classid][time_idx]
        sel_idx = np.random.choice(idx_all, num_sample, replace=True)
        image = self.datasets[time_idx][self.mode]['images'][sel_idx]
        label = self.datasets[time_idx][self.mode]['labels'][sel_idx]
        return torch.FloatTensor(image).permute(0, 3, 1, 2).cuda(), torch.LongTensor(label).unsqueeze(1).cuda()

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass

    def __str__(self):
        return 'yearbook'




class Yearbook(YearbookBase):
    def __init__(self, args):
        super().__init__(args=args)

    def __getitem__(self, index):
        label = self.datasets[self.current_time][self.mode]['labels'][index]
        label_tensor = torch.LongTensor([label])

        image = self.datasets[self.current_time][self.mode]['images'][index]  # image.shape=[32, 32, 3], type: numpy.ndarray, and its data has already been sacled to [0, 1] in yearbook.pkl
        if self.args.method in ['simclr', 'swav'] and self.ssl_training:
            image_tensor = torch.FloatTensor(image).permute(2, 0, 1)
            PIL_image = transforms.ToPILImage()(image_tensor)
            return PIL_image, label_tensor, ''
        else:
            image_tensor = torch.FloatTensor(image).permute(2, 0, 1)
            return image_tensor, label_tensor

    def __len__(self):
        return len(self.datasets[self.current_time][self.mode]['labels'])




class YearbookGroup(YearbookBase):
    def __init__(self, args):
        super().__init__(args=args)
        self.group_size = args.group_size
        self.num_groups = (args.split_time - args.init_timestamp + 1) - args.group_size + 1

    def __getitem__(self, index):
        if self.mode == 0:
            np.random.seed(index)
            # Select group ID
            idx = self.ENV.index(self.current_time)
            possible_groupids = [i for i in range(max(1, (idx + 1) - self.group_size + 1))]
            groupid = np.random.choice(possible_groupids)

            # Pick a time step in the sliding window
            window = np.arange(groupid, groupid + self.group_size)
            sel_time = self.ENV[np.random.choice(window)]
            start_idx, end_idx = self.task_idxs[sel_time][self.mode]

            # Pick an example in the time step
            sel_idx = np.random.choice(np.arange(start_idx, end_idx))
            image = self.datasets[self.current_time][self.mode]['images'][sel_idx]
            label = self.datasets[self.current_time][self.mode]['labels'][sel_idx]

            image_tensor = torch.FloatTensor(image).permute(2, 0, 1)
            label_tensor = torch.LongTensor([label])
            group_tensor = torch.LongTensor([groupid])

            del groupid
            del window
            del sel_time
            del start_idx
            del end_idx
            del sel_idx
            return image_tensor, label_tensor, group_tensor
        else:
            image = self.datasets[self.current_time][self.mode]['images'][index]
            label = self.datasets[self.current_time][self.mode]['labels'][index]
            image_tensor = torch.FloatTensor(image).permute(2, 0, 1)
            label_tensor = torch.LongTensor([label])
            return image_tensor, label_tensor

    def group_counts(self):
        idx = self.ENV.index(self.current_time)
        return torch.LongTensor([1 for _ in range(min(self.num_groups, idx + 1))])

    def __len__(self):
        return len(self.datasets[self.current_time][self.mode]['labels'])
