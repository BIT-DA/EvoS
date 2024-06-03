import numpy as np
import torch
import torch.nn as nn
import random
import os
from networks.rmnist import RotatedMNISTNetwork, RotatedMNISTNetwork_for_EvoS
from networks.article import ArticleNetwork, ArticleNetwork_for_EvoS
from networks.fmow import FMoWNetwork, FMoWNetwork_for_EvoS
from networks.yearbook import YearbookNetwork, YearbookNetwork_for_EvoS
from functools import partial


scheduler = None
group_datasets = ['coral', 'irm']
print = partial(print, flush=True)


def _rmnist_init(args):
    if args.method in group_datasets:
        from data.rmnist import RotatedMNISTGroup
        dataset = RotatedMNISTGroup(args)
    else:
        from data.rmnist import RotatedMNIST
        dataset = RotatedMNIST(args)

    scheduler = None
    criterion = nn.CrossEntropyLoss(reduction=args.reduction).cuda()
    if args.method == "evos":
        network = RotatedMNISTNetwork_for_EvoS(args, num_input_channels=1, num_classes=dataset.num_classes).cuda()
        optimizer = torch.optim.Adam(network.get_parameters(args.lr), lr=args.lr, weight_decay=args.weight_decay)
    else:
        network = RotatedMNISTNetwork(args, num_input_channels=1, num_classes=dataset.num_classes).cuda()
        optimizer = torch.optim.Adam(network.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    return dataset, criterion, network, optimizer, scheduler


def _yearbook_init(args):
    if args.method in group_datasets:
        from data.yearbook import YearbookGroup
        dataset = YearbookGroup(args)
    else:
        from data.yearbook import Yearbook
        dataset = Yearbook(args)

    scheduler = None
    criterion = nn.CrossEntropyLoss(reduction=args.reduction).cuda()
    if args.method == "evos":
        network = YearbookNetwork_for_EvoS(args, num_input_channels=3, num_classes=dataset.num_classes).cuda()
        optimizer = torch.optim.Adam(network.get_parameters(args.lr), lr=args.lr, weight_decay=args.weight_decay)
    else:
        network = YearbookNetwork(args, num_input_channels=3, num_classes=dataset.num_classes).cuda()
        optimizer = torch.optim.Adam(network.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    return dataset, criterion, network, optimizer, scheduler


def _fmow_init(args):
    if args.method in group_datasets:
        from data.fmow import FMoWGroup
        dataset = FMoWGroup(args)
    else:
        from data.fmow import FMoW
        dataset = FMoW(args)

    scheduler = None
    criterion = nn.CrossEntropyLoss(reduction=args.reduction).cuda()
    if args.method == "evos":
        network = FMoWNetwork_for_EvoS(args).cuda()
        optimizer = torch.optim.Adam((network.get_parameters(args.lr)), lr=args.lr, weight_decay=args.weight_decay, amsgrad=True, betas=(0.9, 0.999))
    else:
        network = FMoWNetwork(args).cuda()
        optimizer = torch.optim.Adam((network.parameters()), lr=args.lr, weight_decay=args.weight_decay, amsgrad=True, betas=(0.9, 0.999))
    return dataset, criterion, network, optimizer, scheduler



def _arxiv_init(args):
    if args.method in group_datasets:
        from data.arxiv import ArXivGroup
        dataset = ArXivGroup(args)
    else:
        from data.arxiv import ArXiv
        dataset = ArXiv(args)
    scheduler = None
    criterion = nn.CrossEntropyLoss(reduction=args.reduction).cuda()
    if args.method == "evos":
        network = ArticleNetwork_for_EvoS(args, num_classes=dataset.num_classes).cuda()
        optimizer = torch.optim.Adam((network.get_parameters(args.lr)), lr=args.lr, weight_decay=args.weight_decay, amsgrad=True, betas=(0.9, 0.999))
    else:
        network = ArticleNetwork(num_classes=dataset.num_classes).cuda()
        if args.method == "si" or (args.method == "erm" and args.lisa):
            optimizer = torch.optim.AdamW(network.parameters(), lr=args.lr, weight_decay=args.weight_decay, amsgrad=True, betas=(0.9, 0.999))
        else:
            optimizer = torch.optim.Adam(network.parameters(), lr=args.lr, weight_decay=args.weight_decay, amsgrad=True, betas=(0.9, 0.999))
    return dataset, criterion, network, optimizer, scheduler



def _huffpost_init(args):
    if args.method in group_datasets:
        from data.huffpost import HuffPostGroup
        dataset = HuffPostGroup(args)
    else:
        from data.huffpost import HuffPost
        dataset = HuffPost(args)

    scheduler = None
    criterion = nn.CrossEntropyLoss(reduction=args.reduction).cuda()
    if args.method == "evos":
        network = ArticleNetwork_for_EvoS(args, num_classes=dataset.num_classes).cuda()
        optimizer = torch.optim.Adam((network.get_parameters(args.lr)), lr=args.lr, weight_decay=args.weight_decay, amsgrad=True, betas=(0.9, 0.999))
    else:
        network = ArticleNetwork(num_classes=dataset.num_classes).cuda()
        optimizer = torch.optim.Adam(network.parameters(), lr=args.lr, weight_decay=args.weight_decay, amsgrad=True, betas=(0.9, 0.999))
    return dataset, criterion, network, optimizer, scheduler



def trainer_init(args):
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    return globals()[f'_{args.dataset}_init'](args)


