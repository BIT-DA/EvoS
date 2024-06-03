import torch
import os
import pickle
import numpy as np
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision.transforms.functional import rotate
from sklearn.model_selection import train_test_split

ID_HELD_OUT = 0.1



def rotate_transform(images, labels, angle):
    rotatation = transforms.Compose([
        transforms.Lambda(lambda x: rotate(x, angle, fill=(0,), interpolation=transforms.InterpolationMode.BILINEAR)),
        transforms.ToPILImage()])

    x = np.zeros((len(images), 1, 28, 28))
    for i in range(len(images)):
        x[i] = rotatation(images[i].unsqueeze(0))

    y = labels.view(-1)
    return x, y



def preprocess_orig(args, environments, rotate_angle):
    original_dataset_tr = torchvision.datasets.MNIST(args.data_dir, train=True, download=True)
    original_dataset_te = torchvision.datasets.MNIST(args.data_dir, train=False, download=True)

    original_images = torch.cat((original_dataset_tr.data, original_dataset_te.data))
    original_labels = torch.cat((original_dataset_tr.targets, original_dataset_te.targets))

    shuffle = torch.randperm(len(original_images))

    original_images = original_images[shuffle]
    original_labels = original_labels[shuffle]

    datasets = {}
    for i in range(len(environments)):
        images = original_images[i::len(environments)]
        labels = original_labels[i::len(environments)]
        datasets[i] = {}
        datasets[i][0] = {}
        datasets[i][1] = {}
        datasets[i][2] = {}

        image_in_timestamp_i, label_in_timestamp_i = rotate_transform(images, labels, environments[i] * rotate_angle)

        train_image_split_i, test_image_split_i, train_label_split_i, test_label_split_i = train_test_split(image_in_timestamp_i, label_in_timestamp_i, test_size=ID_HELD_OUT, shuffle=False)

        datasets[i][0]['images'], datasets[i][0]['labels'] = train_image_split_i, train_label_split_i
        datasets[i][1]['images'], datasets[i][1]['labels'] = test_image_split_i, test_label_split_i
        datasets[i][2]['images'], datasets[i][2]['labels'] = image_in_timestamp_i, label_in_timestamp_i
    del original_dataset_tr, original_dataset_te, original_images, original_labels, shuffle
    preprocessed_data_path = os.path.join(args.data_dir, 'rmnist.pkl')
    pickle.dump(datasets, open(preprocessed_data_path, 'wb'))



def preprocess(args, environments, rotate_angle):
    np.random.seed(0)
    if not os.path.isfile(os.path.join(args.data_dir, 'rmnist.pkl')):
        preprocess_orig(args, environments, rotate_angle)
    np.random.seed(args.random_seed)



class RotatedMNIST_Base(Dataset):
    def __init__(self, args):
        super().__init__()
        self.data_file = f'{str(self)}.pkl'

        self.rotation_angle = 10
        self.num_tasks = 9
        self.ENV = [i for i in range(0, self.num_tasks)]  # [0, 10, 20, 30, 40, 50, 60, 70, 80]

        preprocess(args, self.ENV, self.rotation_angle)

        self.datasets = pickle.load(open(os.path.join(args.data_dir, self.data_file), 'rb'))
        self.args = args
        self.num_classes = 10
        self.current_time = 0
        self.resolution = 28
        self.mode = 0
        self.ssl_training = False

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

    def __getitem__(self, idx):
        pass

    def __len__(self):
        pass

    def __str__(self):
        return 'rmnist'

    def update_historical(self, idx, data_del=False):
        time = self.ENV[idx]
        prev_time = self.ENV[idx - 1]
        self.datasets[time][self.mode]['images'] = np.concatenate(
            (self.datasets[prev_time][self.mode]['images'], self.datasets[time][self.mode]['images']), axis=0)
        self.datasets[time][self.mode]['labels'] = np.concatenate(
            (self.datasets[prev_time][self.mode]['labels'], self.datasets[time][self.mode]['labels']), axis=0)
        if data_del:
            del self.datasets[prev_time]
        for classid in range(self.num_classes):
            sel_idx = np.nonzero(self.datasets[time][self.mode]['labels'] == classid)[0]
            self.class_id_list[classid][time] = sel_idx

    def update_current_timestamp(self, time):
        self.current_time = time

    def get_lisa_new_sample(self, time_idx, classid, num_sample):
        idx_all = self.class_id_list[classid][time_idx]  # the index of samples belonging to class c in t-th domain
        if len(idx_all) == 0:
            return None, None
        sel_idx = np.random.choice(idx_all, num_sample, replace=True)
        image = self.datasets[time_idx][self.mode]['images'][sel_idx]
        label = self.datasets[time_idx][self.mode]['labels'][sel_idx]
        return torch.FloatTensor(image).cuda(), torch.LongTensor(label).unsqueeze(1).cuda()




class RotatedMNIST(RotatedMNIST_Base):
    def __init__(self, args):
        super().__init__(args=args)

    def __getitem__(self, index):
        image = self.datasets[self.current_time][self.mode]['images'][index]
        label = self.datasets[self.current_time][self.mode]['labels'][index]

        image_tensor = torch.FloatTensor(image / 255.0)
        label_tensor = torch.LongTensor([label])

        if self.args.method in ['simclr', 'swav'] and self.ssl_training:
            tensor_to_PIL = transforms.ToPILImage()
            PIL_image = tensor_to_PIL(image_tensor)
            return PIL_image, label_tensor, ''
        else:
            return image_tensor, label_tensor

    def __len__(self):
        return len(self.datasets[self.current_time][self.mode]['labels'])



class RotatedMNISTGroup(RotatedMNIST_Base):
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

            image_tensor = torch.FloatTensor(image / 255.0)
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

            image_tensor = torch.FloatTensor(image / 255.0)
            label_tensor = torch.LongTensor([label])
            return image_tensor, label_tensor

    def group_counts(self):
        idx = self.ENV.index(self.current_time)
        return torch.LongTensor([1 for _ in range(min(self.num_groups, idx + 1))])

    def __len__(self):
        return len(self.datasets[self.current_time][self.mode]['labels'])

