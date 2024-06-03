import torch
import torch.nn as nn
from typing import Iterable
import torch.nn.functional as F

from networks.article import ArticleNetwork
from networks.fmow import FMoWNetwork
from networks.rmnist import RotatedMNISTNetwork
from networks.yearbook import YearbookNetwork


class SlidingWindow:
    def __init__(self, size):
        assert size > 0
        self.size = size
        self._data = []

    def __len__(self):
        return len(self._data)

    def push(self, _data: torch.Tensor):
        _data = _data.detach()
        if len(self) == self.size:
            self.pop()
        self._data.append(_data)

    def pop(self):
        if len(self) > 0:
            self._data.pop()

    @property
    def data(self):
        return self._data


def mul(t: Iterable[int]):
    result = 1
    for item in t:
        result *= item

    return result


class WrappedDrainNetwork(nn.Module):

    def __init__(self, network: nn.Module,
                 hidden_dim: int,
                 latent_dim: int,
                 num_rnn_layers=1,
                 num_layer_to_replace=-1,
                 window_size=-1,
                 lambda_forgetting=0.,
                 dim_bottleneck_f=None):
        super(WrappedDrainNetwork, self).__init__()
        self.num_layer_to_replace = num_layer_to_replace  # < 0 means all
        self.window_size = window_size
        self.lambda_forgetting = lambda_forgetting
        self.dim_bottleneck_f = dim_bottleneck_f

        self.sliding_window = SlidingWindow(self.window_size) if self.window_size > 0 else None

        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_rnn_layer = num_rnn_layers

        offset = self.process_network(network)

        self.network = network  # without any parameters

        self.code_dim = offset

        self.rnn = nn.LSTM(self.latent_dim, self.latent_dim, self.num_rnn_layer)

        # Transforming LSTM output to vector shape
        self.param_decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.code_dim))
        # Transforming vector to LSTM input shape
        self.param_encoder = nn.Sequential(
            nn.Linear(self.code_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.latent_dim))

        self.hidden = None
        self.E = None

        self.init_e_hidden()

    def process_network(self, network):
        offset = 0
        block_names = self.get_block_names(network, self.dim_bottleneck_f)
        for index, name in enumerate(block_names):
            if 0 < self.num_layer_to_replace <= index:
                break
            block = network.get_submodule(name)
            for sub_module in block.modules():
                offset += self.trans_param_to_buffer(sub_module, offset)

        return offset

    @staticmethod
    def get_block_names(network, dim_bottleneck_f=None):
        if isinstance(network, FMoWNetwork):
            if dim_bottleneck_f is not None:
                block_names = ['enc.0.conv0', 'enc.0.norm0', 'enc.0.relu0', 'enc.0.pool0', 'enc.0.denseblock1',
                               'enc.0.transition1', 'enc.0.denseblock2', 'enc.0.transition2', 'enc.0.denseblock3',
                               'enc.0.transition3', 'enc.0.denseblock4', 'enc.0.norm5', "enc.1", "enc.2", "enc.3",
                               "enc.4", "classifier"]  # blocks of enc(densenet101), classifier
            else:
                block_names = ['enc.0.conv0', 'enc.0.norm0', 'enc.0.relu0', 'enc.0.pool0', 'enc.0.denseblock1',
                               'enc.0.transition1', 'enc.0.denseblock2', 'enc.0.transition2', 'enc.0.denseblock3',
                               'enc.0.transition3', 'enc.0.denseblock4', 'enc.0.norm5', "enc.1", "enc.2", "enc.3",
                               "classifier"]  # blocks of enc(densenet101), classifier
        elif isinstance(network, YearbookNetwork):
            block_names = [""]  # network itself
        elif isinstance(network, ArticleNetwork):
            block_names = ["model.0.embeddings", "model.0.transformer.layer.0", "model.0.transformer.layer.1",
                           "model.0.transformer.layer.2", "model.0.transformer.layer.3",
                           "model.0.transformer.layer.4" "model.0.transformer.layer.5", "model.1"]
        elif isinstance(network, RotatedMNISTNetwork):
            block_names = [""]
        else:
            raise NotImplementedError
        block_names.reverse()
        return block_names

    @staticmethod
    def trans_param_to_buffer(module: nn.Module, offset: int):
        module.has_been_transformed = True
        names = []
        shapes = []
        # get the name and shape of params of the current module
        for name, param in module.named_parameters(recurse=False):
            names.append(name)
            shapes.append(param.shape)

        for name, shape in zip(names, shapes):
            # delete all parameters and register these name as buffer (maybe simple attribute)
            delattr(module, name)
            module.register_buffer(name, torch.randn(shape))

        module.transformed_names = names
        module.shapes_for_names = shapes
        module.offset = offset

        if len(names) == 0:
            return 0
        else:
            num_params = 0
            for shape in shapes:
                num_params += mul(shape)
            return num_params

    def reconstruct(self, decoded_params: torch.Tensor):
        decoded_params = self.skip_connection(decoded_params)
        for sub_module in self.network.modules():
            self.reconstruct_module(sub_module, decoded_params)

    def skip_connection(self, decoded_params: torch.Tensor):
        history = self.lambda_forgetting * sum(self.sliding_window.data)
        return decoded_params + history

    @staticmethod
    def reconstruct_module(module: nn.Module, decoded_params: torch.Tensor):
        if not hasattr(module, "has_been_transformed"):
            return

        offset = module.offset
        all_names = module.transformed_names
        all_shapes = module.shapes_for_names

        local_offset = 0

        for name, shape in zip(all_names, all_shapes):
            value = torch.reshape(decoded_params[offset + local_offset:offset + local_offset + mul(shape)], shape)
            setattr(module, name, value)
            local_offset += mul(shape)

    def init_e_hidden(self):
        init_c, init_h = [], []
        for _ in range(self.num_rnn_layer):
            init_c.append(torch.tanh(torch.randn(1, self.latent_dim)))
            init_h.append(torch.tanh(torch.randn(1, self.latent_dim)))

        self.hidden = (torch.stack(init_c, dim=0).cuda(), torch.stack(init_h, dim=0).cuda())

        self.E = torch.randn((1, self.code_dim)).cuda()

    def forward(self, x):
        lstm_input = self.param_encoder(self.E)

        lstm_out, hidden = self.rnn(lstm_input.unsqueeze(0), self.hidden)

        new_E = self.param_decoder(lstm_out.squeeze(0))

        self.reconstruct(new_E.view(-1))

        if self.training:
            self.E = torch.detach(new_E)
            for item in hidden:
                item.detach_()

            self.hidden = hidden

        prediction = self.network(x)

        return prediction

    def foward_encoder(self, x):
        lstm_input = self.param_encoder(self.E)

        lstm_out, hidden = self.rnn(lstm_input.unsqueeze(0), self.hidden)

        new_E = self.param_decoder(lstm_out.squeeze(0))

        self.reconstruct(new_E.view(-1))

        if isinstance(self.network, ArticleNetwork):
            encoder = self.network.model[0]
            fea = encoder(x)
        elif isinstance(self.network, FMoWNetwork):
            encoder = self.network.enc
            fea = encoder(x)
        elif isinstance(self.network, YearbookNetwork):
            encoder = self.network.enc
            x = encoder(x)
            fea = torch.mean(x, dim=(2, 3))
        elif isinstance(self.network, RotatedMNISTNetwork):
            encoder = self.network.enc
            x = encoder(x)
            fea = x.view(len(x), -1)
        else:
            raise NotImplementedError("Please implement your wrapped feature extractor!")

        return fea

    def push_E(self):
        if self.E is not None:
            self.sliding_window.push(self.E.view(-1))

    def update_hidden(self):
        with torch.no_grad():
            lstm_input = self.param_encoder(self.E)

            lstm_out, hidden = self.rnn(lstm_input.unsqueeze(0), self.hidden)

            new_E = self.param_decoder(lstm_out.squeeze(0))

            self.E = torch.detach(new_E)
            for item in hidden:
                item.detach_()
            self.hidden = hidden

