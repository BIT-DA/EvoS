import torch
from torch import nn
import torch.nn.functional as F

from networks.article import ArticleNetwork
from networks.fmow import FMoWNetwork
from networks.rmnist import RotatedMNISTNetwork
from networks.yearbook import YearbookNetwork


class WrappedFeature(nn.Module):
    def __init__(self, network: nn.Module):
        super().__init__()
        self.type = type(network)
        if isinstance(network, ArticleNetwork):
            self.model = network.model[0]
        elif isinstance(network, FMoWNetwork):
            self.model = network.enc
        elif isinstance(network, YearbookNetwork):
            self.model = network.enc
        elif isinstance(network, RotatedMNISTNetwork):
            self.model = network.enc
        else:
            raise NotImplementedError("Please implement your wrapped feature extractor!")

    def forward(self, input_x):
        if self.type == ArticleNetwork:
            return self.model(input_x)
        elif self.type == FMoWNetwork:
            return self.model(input_x)
        elif self.type == YearbookNetwork:
            x = self.model(input_x)
            return torch.mean(x, dim=(2, 3))
        elif self.type == RotatedMNISTNetwork:
            x = self.model(input_x)
            x = x.view(len(x), -1)
            return x


def get_out_shape_hook(model: nn.Module, input, output: torch.Tensor):
    setattr(model, "n_outputs", output.shape[1])


class WrappedClassifier(nn.Module):
    def __init__(self, network: nn.Module):
        super().__init__()
        if isinstance(network, ArticleNetwork):
            self.model = network.model[1]
        elif isinstance(network, FMoWNetwork):
            self.model = network.classifier
        elif isinstance(network, YearbookNetwork):
            self.model = network.classifier
        elif isinstance(network, RotatedMNISTNetwork):
            self.model = network.classifier
        else:
            raise NotImplementedError("Please implement your wrapped feature extractor!")

    def forward(self, input_x):
        return self.model(input_x)

