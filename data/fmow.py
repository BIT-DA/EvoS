import os
import pickle
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from wilds import get_dataset


ID_HELD_OUT = 0.1



class FMoWBase(Dataset):
    def __init__(self, args):
        super().__init__()
        self.data_file = f'{str(self)}.pkl'
        preprocess(args)

        self.datasets = pickle.load(open(os.path.join(args.data_dir, self.data_file), 'rb'))
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

        # dataset = get_dataset(dataset="fmow", root_dir=args.data_dir, download=True)  # automatically download and unzip
        dataset = get_dataset(dataset="fmow", root_dir=args.data_dir, download=False)
        # manually download fmow_v1.1.tar.gz to the folder where fmow.pkl is located and unzip it.
        # The downloading url of fmow_v1.1.tar.gz is https://worksheets.codalab.org/bundles/0xaec91eb7c9d548ebb15e1b5e60f966ab

        self.root = dataset.root
        self.args = args
        self.num_classes = 62
        self.current_time = 0
        self.num_tasks = 16
        self.ENV = [year for year in range(0, self.num_tasks)]
        self.resolution = 224
        self.mode = 0
        self.ssl_training = False

        self.class_id_list = {i: {} for i in range(self.num_classes)}
        self.task_idxs = {}
        start_idx = 0
        for year in sorted(self.datasets.keys()):
            end_idx = start_idx + len(self.datasets[year][self.mode]['labels'])
            self.task_idxs[year] = {}
            self.task_idxs[year] = [start_idx, end_idx]
            start_idx = end_idx
            for classid in range(self.num_classes):
                sel_idx = np.nonzero(self.datasets[year][self.mode]['labels'] == classid)[0]
                self.class_id_list[classid][year] = sel_idx

    def __getitem__(self, idx):
        pass

    def __len__(self):
        pass

    def __str__(self):
        return 'fmow'

    def update_historical(self, idx, data_del=False):
        time = self.ENV[idx]
        prev_time = self.ENV[idx - 1]
        self.datasets[time][self.mode]['image_idxs'] = np.concatenate(
            (self.datasets[prev_time][self.mode]['image_idxs'], self.datasets[time][self.mode]['image_idxs']), axis=0)
        self.datasets[time][self.mode]['labels'] = np.concatenate(
            (self.datasets[prev_time][self.mode]['labels'], self.datasets[time][self.mode]['labels']), axis=0)
        if data_del:
            del self.datasets[prev_time]
        for classid in range(self.num_classes):
            sel_idx = np.nonzero(self.datasets[time][self.mode]['labels'] == classid)[0]
            self.class_id_list[classid][time] = sel_idx

    def update_current_timestamp(self, time):
        self.current_time = time

    def get_input(self, idx):
        """Returns x for a given idx."""
        idx = self.datasets[self.current_time][self.mode]['image_idxs'][idx]
        img = Image.open(self.root / 'images' / f'rgb_img_{idx}.png').convert('RGB')
        return img

    def get_lisa_new_sample(self, time_idx, classid, num_sample):
        idx_all = self.class_id_list[classid][time_idx]
        sel_idx = np.random.choice(idx_all, num_sample, replace=True)
        image = torch.stack([self.transform(self.get_input(idx)) for idx in sel_idx], dim=0)
        label = self.datasets[time_idx][self.mode]['labels'][sel_idx]
        return torch.FloatTensor(image).cuda(), torch.LongTensor(label).unsqueeze(1).cuda()


class FMoW(FMoWBase):
    def __init__(self, args):
        super().__init__(args=args)

    def __getitem__(self, idx):
        label_tensor = torch.LongTensor([self.datasets[self.current_time][self.mode]['labels'][idx]])

        if self.args.method in ['simclr', 'swav'] and self.ssl_training:
            image = self.get_input(idx)
            return image, label_tensor, ''
        else:
            image_tensor = self.transform(self.get_input(idx))
            return image_tensor, label_tensor

    def __len__(self):
        return len(self.datasets[self.current_time][self.mode]['labels'])



class FMoWGroup(FMoWBase):
    def __init__(self, args):
        super().__init__(args=args)
        self.group_size = args.group_size
        self.train = True
        self.num_groups = (args.split_time - args.init_timestamp + 1) - args.group_size + 1

    def __getitem__(self, idx):
        if self.mode == 0:
            np.random.seed(idx)
            # Select group ID
            idx = self.ENV.index(self.current_time)
            possible_groupids = [i for i in range(max(1, (idx + 1) - self.group_size + 1))]
            groupid = np.random.choice(possible_groupids)

            # Pick a time step in the sliding window
            window = np.arange(groupid, groupid + self.group_size)
            sel_time = self.ENV[np.random.choice(window)]
            start_idx, end_idx = self.task_idxs[sel_time]

            # Pick an example in the time step
            sel_idx = np.random.choice(np.arange(start_idx, end_idx))
            label = self.datasets[self.current_time][self.mode]['labels'][sel_idx]
            image_tensor = self.transform(self.get_input(sel_idx))
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
            image_tensor = self.transform(self.get_input(idx))
            label = self.datasets[self.current_time][self.mode]['labels'][idx]
            label_tensor = torch.LongTensor([label])
            return image_tensor, label_tensor

    def group_counts(self):
        idx = self.ENV.index(self.current_time)
        return torch.LongTensor([1 for _ in range(min(self.num_groups, idx + 1))])

    def __len__(self):
        return len(self.datasets[self.current_time][self.mode]['labels'])




def preprocess(args):
    if not os.path.isfile(os.path.join(args.data_dir, 'fmow.pkl')):
        raise RuntimeError("dataset fmow.pkl is not yet ready! Please download from   https://drive.google.com/u/0/uc?id=1s_xtf2M5EC7vIFhNv_OulxZkNvrVwIm3&export=download   and save it as fmow.pkl")
