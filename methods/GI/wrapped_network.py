import torch
import torch.nn as nn
from typing import Iterable

import torch
from torch import nn
import torch.nn.functional as F
from typing import List
from torch.utils.hooks import RemovableHandle
from transformers.activations import GELUActivation

from networks.article import ArticleNetwork

from ..lssae.wrapped_network import WrappedFeature


def mul(t: Iterable[int]):
    result = 1
    for item in t:
        result *= item

    return result


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.kaiming_normal_(m.weight)
        m.bias.data.fill_(0.01)


class Time2Vec(nn.Module):
    '''
    Time encoding inspired by the Time2Vec paper
    '''

    def __init__(self, in_shape, out_shape):

        super(Time2Vec, self).__init__()
        linear_shape = out_shape // 4
        dirac_shape = 0
        sine_shape = out_shape - linear_shape - dirac_shape
        self.model_0 = nn.Linear(in_shape, linear_shape)
        self.model_1 = nn.Linear(in_shape, sine_shape)

    def forward(self, X):

        te_lin = self.model_0(X)
        te_sin = torch.sin(self.model_1(X))
        if len(te_lin.shape) == 3:
            te_lin = te_lin.squeeze(1)
        if len(te_sin.shape) == 3:
            te_sin = te_sin.squeeze(1)
        te = torch.cat((te_lin, te_sin), dim=1)
        return te


class TimeReLUCompatible(nn.Module):
    '''
    A ReLU with threshold and alpha as a function of domain indices.
    '''

    def __init__(self, time_shape, leaky=False, use_time=True, deep=False):

        super(TimeReLUCompatible, self).__init__()
        self.deep = deep  # Option for a deeper version of TReLU
        if deep:
            trelu_shape = 16
        else:
            trelu_shape = 1

        self.trelu_shape = trelu_shape
        self.leaky = leaky

        self.use_time = use_time  # Whether TReLU is active or not
        self.model_0 = nn.Linear(time_shape, trelu_shape)
        self.model_1 = None
        self.time_dim = time_shape

        if self.leaky:
            self.model_alpha_0 = nn.Linear(time_shape, trelu_shape)
            self.model_alpha_1 = None

        self.sigmoid = nn.Sigmoid()

        if self.leaky:
            self.relu = nn.LeakyReLU()
        else:
            self.relu = nn.ReLU()

        self.times = None

    def forward(self, X):
        assert self.times is not None, "please call self.set_times(times) for this forward process"
        if not self.use_time:
            return self.relu(X)
        if len(self.times.size()) == 3:
            self.times = self.times.squeeze(2)
        thresholds = self.model_1(self.relu(self.model_0(self.times)))

        # reshape to (B, ...)
        thresholds = torch.reshape(thresholds, (-1, *self.data_shape))

        if self.leaky:
            alphas = self.model_alpha_1(self.relu(self.model_alpha_0(self.times)))
            alphas = torch.reshape(alphas, (-1, *self.data_shape))
        else:
            alphas = 0.0
        if self.deep:
            X = torch.where(X > thresholds, X, alphas * (X - thresholds) + thresholds)
        else:
            X = torch.where(X > thresholds, X - thresholds, alphas * (X - thresholds))

        # set times to None
        self.times = None
        return X

    def set_times(self, times):
        self.times = times


def hook_for_init_timerelu(module: TimeReLUCompatible, input_):
    X, = input_
    if module.model_1 is not None:
        print("please delete this hook after the first run!")
        return
    shape = X.shape
    assert len(shape) >= 2, "error input size"  # (B, ...)
    data_shape = shape[1:]
    setattr(module, "data_shape", data_shape)
    model_1 = nn.Linear(module.trelu_shape, mul(data_shape))
    model_alpha_1 = nn.Linear(module.trelu_shape, mul(data_shape))

    device = module.model_0.weight.device
    model_1.to(device)
    model_alpha_1.to(device)

    setattr(module, "model_1", model_1)
    setattr(module, "model_alpha_1", model_alpha_1)


class TimeReluClassifier(nn.Module):
    def __init__(self, input_dim, out_dim, time_dim=8):
        super(TimeReluClassifier, self).__init__()
        self.fc_1 = nn.Linear(input_dim, 256)

        self.time_relu = TimeReLUCompatible(time_dim, leaky=True, use_time=True)
        self.handle = self.time_relu.register_forward_pre_hook(hook_for_init_timerelu)

        self.fc_2 = nn.Linear(256, out_dim)
        self.first_forward = True

    def forward(self, x, time_v):
        x = self.fc_1(x)

        self.time_relu.set_times(time_v)
        x = self.time_relu(x)

        out = self.fc_2(x)

        if self.first_forward:
            self.handle.remove()
            self.first_forward = False
        return out


class WrappedGIFeature(nn.Module):
    def __init__(self, feature: WrappedFeature, time_dim=8, num_replace=None):
        # network already in gup
        super(WrappedGIFeature, self).__init__()
        self.time_dim = time_dim
        self.timerelu_names, self.timerelu_handles = self.replace_relu_with_timerelu(feature, time_dim, num_replace)
        assert len(self.timerelu_names) >= 0, "no relu in the model!"
        self.feature = feature
        self.first_forward = True

    @staticmethod
    def replace_relu_with_timerelu(network: nn.Module, time_dim, num_replace=None):
        timerelu_names: List[str] = []
        timerelu_handles: List[RemovableHandle] = []
        for name, sub_module in network.named_modules(remove_duplicate=False):
            if isinstance(sub_module, nn.ReLU) or isinstance(sub_module, GELUActivation):
                timerelu_names.append(name)

        if isinstance(network, ArticleNetwork) and num_replace is not None:
            timerelu_names.reverse()
            timerelu_names = timerelu_names[0:num_replace]

        for name in timerelu_names:
            parent_name, _, relu_name = name.rpartition(".")
            parent_module = network.get_submodule(parent_name)
            time_relu = TimeReLUCompatible(time_dim, leaky=True, use_time=True).cuda()
            handle = time_relu.register_forward_pre_hook(hook_for_init_timerelu)
            timerelu_handles.append(handle)
            setattr(parent_module, relu_name, time_relu)

        return timerelu_names, timerelu_handles

    def forward(self, x, time_v):
        self.set_time_for_timerelu(time_v)

        out = self.feature(x)

        if self.first_forward:
            self.remove_hooks()
            self.first_forward = False

        return out

    def remove_hooks(self):
        for handle in self.timerelu_handles:
            handle.remove()

    def set_time_for_timerelu(self, time):
        for name in self.timerelu_names:
            timerelu: TimeReLUCompatible = self.feature.get_submodule(name)
            timerelu.set_times(time)


class WrappedGINetwork(nn.Module):
    def __init__(self, feature: WrappedGIFeature, classifier: TimeReluClassifier, time_dim=8, time_append_dim=256):
        super(WrappedGINetwork, self).__init__()
        self.time_dim = time_dim
        self.time_append_dim = time_append_dim
        self.t2v = Time2Vec(1, self.time_dim)
        self.time_fc = nn.Linear(self.time_dim, self.time_append_dim)
        self.feature = feature
        self.classifier = classifier

    def forward(self, x, t):
        time_v = self.t2v(t)
        feature = self.feature(x, time_v)
        time_append = self.time_fc(time_v)
        feature = torch.cat((feature, time_append), dim=-1)
        out = self.classifier(feature, time_v)
        return out

    def foward_encoder(self, x, t):
        time_v = self.t2v(t)
        feature = self.feature(x, time_v)
        time_append = self.time_fc(time_v)
        feature = torch.cat((feature, time_append), dim=-1)
        return feature

