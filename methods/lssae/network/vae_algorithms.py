import pdb

import torch
from torch.autograd import Variable

from .submodules import *
from .cla_func import *
from .loss_func import *


class AbstractAutoencoder(nn.Module):
    def __init__(self, model_func, cla_func, hparams):
        super().__init__()
        self.model_func = model_func
        self.cla_func = cla_func
        self.hparams = hparams
        self.feature_dim = model_func.n_outputs
        self.data_size = hparams['data_size']
        self.num_classes = hparams['num_classes']
        self.seen_domains = hparams['source_domains']

        self.lssae_zc_dim = hparams['lssae_zc_dim']
        self.lssae_zw_dim = hparams['lssae_zw_dim']
        self.zv_dim = hparams['zv_dim']

        self.recon_criterion = nn.MSELoss(reduction='sum')
        self.criterion = nn.CrossEntropyLoss()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 1)
            elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

    def _get_decoder_func(self):
        if len(self.data_size) > 2:
            if self.data_size[-1] == 28:
                decoder_class = CovDecoder28x28
            elif self.data_size[-1] == 84:
                decoder_class = CovDecoder84x84
            elif self.data_size[-1] == 32:
                decoder_class = CovDecoder32x32
            elif self.data_size[-1] == 224:
                decoder_class = CovDecoder224x224
            else:
                raise ValueError('Don\'t support shape:{}'.format(self.hparams['data_size']))
        else:
            decoder_class = LinearDecoder
        return decoder_class

    @abstractmethod
    def _build(self):
        pass

    @abstractmethod
    def update(self, minibatches, unlabeled=False):
        """
        Perform one update step, given a list of (x, y) tuples for all
        environments.
        Admits an optional list of unlabeled minibatches from the test domains,
        when task is domain_adaptation.
        """
        pass

    @abstractmethod
    def predict(self, x, *args, **kwargs):
        pass

    def calc_recon_loss(self, recon_x, x):
        recon_loss = self.recon_criterion(recon_x, x)
        recon_loss = recon_loss.sum()
        return recon_loss

    def update_scheduler(self):
        pass


class LSSAE(AbstractAutoencoder):
    """
    Implementation of LSSAE.
    """

    def __init__(self, model_func, cla_func, hparams):
        super(LSSAE, self).__init__(model_func, cla_func, hparams)
        self.factorised = True

        self.aux_loss_multiplier_y = hparams['lssae_coeff_y']
        self.ts_multiplier = hparams['lssae_coeff_ts']
        self.lssae_coeff_w = hparams['lssae_coeff_w']

        self._build()
        self._init()

    def _build(self):
        # Static env components
        self.static_prior = GaussianModule(self.lssae_zc_dim)
        self.dynamic_w_prior = ProbabilisticSingleLayerLSTM(input_dim=self.lssae_zw_dim,
                                                            hidden_dim=2 * self.lssae_zw_dim,
                                                            stochastic=self.hparams['stochastic'])

        self.dynamic_v_prior = ProbabilisticCatSingleLayer(input_dim=self.zv_dim,
                                                           hidden_dim=2 * self.zv_dim,
                                                           stochastic=self.hparams['stochastic'])

        self.static_encoder = StaticProbabilisticEncoder(self.model_func,
                                                         self.lssae_zc_dim,
                                                         stochastic=self.hparams['stochastic'])
        self.dynamic_w_encoder = DynamicProbabilisticEncoder(copy.deepcopy(self.model_func),
                                                             self.lssae_zw_dim,
                                                             self.lssae_zc_dim,
                                                             factorised=self.factorised,
                                                             stochastic=self.hparams['stochastic'])
        self.dynamic_v_encoder = DynamicCatEncoder(self.zv_dim,
                                                   self.lssae_zc_dim,
                                                   factorised=self.factorised,
                                                   stochastic=self.hparams['stochastic'])

        self.decoder = self._get_decoder_func()(self.lssae_zc_dim + self.lssae_zw_dim, self.data_size)

        # fix dimension mismatch
        self.category_cla_func = SingleLayerClassifier(self.zv_dim + self.lssae_zc_dim, self.num_classes)

        self.opt = torch.optim.Adam([{'params': self.static_encoder.parameters()},
                                     {'params': self.category_cla_func.parameters()},
                                     {'params': self.decoder.parameters()},
                                     {'params': self.dynamic_w_encoder.parameters(), 'lr': 1.0 * self.hparams["lr"]},
                                     {'params': self.dynamic_v_encoder.parameters(), 'lr': 1.0 * self.hparams["lr"]},
                                     {'params': self.dynamic_w_prior.parameters(), 'lr': 1.0 * self.hparams["lr"]},
                                     {'params': self.dynamic_v_prior.parameters(), 'lr': 1.0 * self.hparams["lr"]}
                                     ],
                                    lr=self.hparams["lr"],
                                    weight_decay=self.hparams['weight_decay'])

    @staticmethod
    def gen_dynamic_prior(prior_net, prior_latent_dim, domains, batch_size=1, stochastic=False):
        z_out, z_out_value = None, None
        hx = Variable(prior_net.h0.detach().clone(), requires_grad=True)
        cx = Variable(prior_net.c0.detach().clone(), requires_grad=True)

        init_prior = torch.zeros([2 * prior_latent_dim if stochastic else prior_latent_dim]).cuda()

        z_t = Variable(init_prior.detach().clone(), requires_grad=True).unsqueeze(0)

        for _ in range(domains):
            z_t, hx, cx = prior_net(z_t, hx, cx.detach().clone())

            if z_out is None:
                # If z_out is none it means z_t is z_1, hence store it in the format [batch_size, 1, z_dim]
                z_out = z_t.unsqueeze(1)
                z_out_value = prior_net.sampling(batch_size)
            else:
                z_out = torch.cat((z_out, z_t.unsqueeze(1)), dim=1)
                z_out_value = torch.cat((z_out_value, prior_net.sampling(batch_size)), dim=1)
        return z_out, z_out_value

    def update(self, minibatches, unlabeled=None):
        """
        :param minibatches: list
        :param unlabeled:
        :return:
        """
        all_x = torch.stack([x for x, y in minibatches])  # [source_domains, batch_size, data_size]
        all_y = torch.stack([y for x, y in minibatches])  # [source_domains, batch_size]

        domains, batch_size = all_x.shape[:2]

        all_x = torch.transpose(all_x, 0, 1)  # [batch_size, source_domains, data_size]
        all_y = torch.transpose(all_y, 0, 1)  # [batch_size, source_domains]

        # ------------------------------ Covariant shift  -------------------------------
        static_qx_latent_variables = self.static_encoder(all_x)  # [batch_size, lssae_zc_dim*2]
        dynamic_qw_latent_variables = self.dynamic_w_encoder(all_x, None)  # [batch_size, source_domains, lssae_zw_dim*2]
        dynamic_pw_latent_variables, _ = self.gen_dynamic_prior(self.dynamic_w_prior, self.lssae_zw_dim, domains, batch_size,
                                                                self.hparams['stochastic'])  # [1, source_domains, lssae_zw_dim*2]

        zc = self.static_encoder.sampling()
        zw = self.dynamic_w_encoder.sampling()
        recon_x = self.decoder(torch.cat([zc, zw], dim=1))
        all_x = all_x.contiguous().view(batch_size * domains, *all_x.shape[2:])
        CE_x = self.calc_recon_loss(recon_x, all_x)
        CE_x = CE_x / (domains * batch_size * all_x.shape[2] * all_x.shape[3])

        # Distribution loss
        # kld on zc
        static_kld = -1.0 * torch.sum(1 + static_qx_latent_variables[:, self.lssae_zc_dim:] -
                                      torch.pow(static_qx_latent_variables[:, :self.lssae_zc_dim], 2) -
                                      torch.exp(static_qx_latent_variables[:, self.lssae_zc_dim:]))
        static_kld = static_kld / (domains * batch_size * self.lssae_zc_dim)

        # kld on zw
        dynamic_qw_mu, dynamic_qw_log_sigma = dynamic_qw_latent_variables[:, :, :self.lssae_zw_dim], \
                                              dynamic_qw_latent_variables[:, :, self.lssae_zw_dim:]
        dynamic_pw_mu, dynamic_pw_log_sigma = dynamic_pw_latent_variables[:, :, :self.lssae_zw_dim], \
                                              dynamic_pw_latent_variables[:, :, self.lssae_zw_dim:]
        dynamic_qw_sigma = torch.exp(dynamic_qw_log_sigma)
        dynamic_pw_sigma = torch.exp(dynamic_pw_log_sigma)

        dynamic_w_kld = 1.0 * torch.sum(dynamic_pw_log_sigma - dynamic_qw_log_sigma + ((dynamic_qw_sigma + torch.pow(dynamic_qw_mu - dynamic_pw_mu, 2)) / dynamic_pw_sigma) - 1)
        dynamic_w_kld = self.lssae_coeff_w * dynamic_w_kld / (domains * batch_size * self.lssae_zw_dim)

        # ------------------------------ Concept shift  -------------------------------
        all_y = all_y.contiguous().view(-1)
        one_hot_y = one_hot(all_y, self.num_classes, all_y.device)
        one_hot_y = one_hot_y.view(batch_size, domains, -1)
        dynamic_qv_latent_variables = self.dynamic_v_encoder(one_hot_y, None)

        dynamic_pv_latent_variables, _ = self.gen_dynamic_prior(self.dynamic_v_prior, self.zv_dim, domains, batch_size,
                                                                False)  # [1, source_domains, zv_dim]

        # recon y
        zv = self.dynamic_v_encoder.sampling()
        zv.view(batch_size, domains, -1)
        recon_y = self.category_cla_func(torch.cat([zv, zc], dim=1))
        CE_y = self.aux_loss_multiplier_y * self.criterion(recon_y, all_y)

        # kld on zv
        dynamic_v_kld = torch.sum(torch.softmax(dynamic_qv_latent_variables, dim=-1) *
                                  (torch.log_softmax(dynamic_qv_latent_variables, dim=-1) -
                                   torch.log_softmax(dynamic_pv_latent_variables, dim=-1)))
        dynamic_v_kld = dynamic_v_kld / (domains * batch_size * self.zv_dim)

        # temporal smooth constrain on prior_dynamic_latent_variables
        ts_w_loss = self.ts_multiplier * temporal_smooth_loss(dynamic_qw_latent_variables)
        ts_v_loss = self.ts_multiplier * temporal_smooth_loss(dynamic_qv_latent_variables)

        total_loss = (CE_x + static_kld + dynamic_w_kld + dynamic_v_kld) + CE_y + ts_w_loss + ts_v_loss

        self.opt.zero_grad()
        total_loss.backward()
        self.opt.step()

        str_log = 'Total loss:{:.3f}, recon_x_loss:{:.3f}, CE_y:{:.3f}, static_loss:{:.6f}, dynamic_w_loss:{:.3f}, dynamic_v_loss:{:.6f}, TS_W_loss:{:.3f}, TS_V_loss:{:.6f}'.format(total_loss, CE_x, CE_y, static_kld, dynamic_w_kld, dynamic_v_kld, ts_w_loss, ts_v_loss)
        return CE_y, recon_y, all_y, str_log

    def predict(self, x, domain_idx, *args, **kwargs):
        _ = self.static_encoder(x.unsqueeze(1))
        zc = self.static_encoder.latent_space.base_dist.loc
        _, zv_prob = self.gen_dynamic_prior(self.dynamic_v_prior, self.zv_dim, domain_idx+1, x.size(0), False)  # [1, source_domains, zv_dim]
        zv = zv_prob[:, -1, :]
        y_logit = self.category_cla_func(torch.cat([zv, zc], dim=1))
        return y_logit

    def foward_encoder(self, x):
        _ = self.static_encoder(x.unsqueeze(1))
        zc = self.static_encoder.latent_space.base_dist.loc
        return zc

    def reconstruct_for_test(self, minibatches, generative=False):
        """
        :param minibatches:
        :param generative: True or False
        :return:
        """
        all_x = torch.stack([x for x, y in minibatches])  # [source_domains, batch_size, 2]
        all_y = torch.stack([y for x, y in minibatches])  # [source_domains, batch_size]
        domains, batch_size = all_x.shape[:2]

        all_x = torch.transpose(all_x, 0, 1)  # [batch_size, source_domains, 2]
        all_y = torch.transpose(all_y, 0, 1)  # [batch_size, source_domains]

        static_qx_latent_variables = self.static_encoder(all_x)  # [48, 40]
        _ = self.dynamic_w_encoder(all_x, static_qx_latent_variables[:, : self.lssae_zc_dim])  # [48, 15, 40]
        _, zw = self.gen_dynamic_prior(self.dynamic_w_prior, self.lssae_zw_dim, domains, batch_size)  # [1, 15, 40]

        all_y = all_y.contiguous().view(-1)

        domain_idx = torch.arange(domains).to(all_x.device)
        domain_idx = domain_idx.unsqueeze(0).expand(batch_size, -1)
        domain_idx = domain_idx.contiguous().view(-1)

        if generative:
            # sample from static gaussian
            zc = self.static_encoder.sampling(batch_size)
            zw = zw.contiguous().view(batch_size, domains, -1)
            zw = zw[:, 0, :]
            zw = zw.expand(batch_size * domains, -1)
        else:
            zc = self.static_encoder.sampling()
            zw = self.dynamic_w_encoder.sampling()  # [720, 20]
        recon_x = self.decoder(torch.cat([zc, zw], dim=1))

        return recon_x, all_y, domain_idx


def one_hot(indices, depth, device=None):
    """
    Returns a one-hot tensor.
    This is a PyTorch equivalent of Tensorflow's tf.one_hot.
    Parameters:
      indices:  a (n_batch, m) Tensor or (m) Tensor.
      depth: a scalar. Represents the depth of the one hot dimension.
    Returns: a (n_batch, m, depth) Tensor or (m, depth) Tensor.
    """
    if device is None:
        encoded_indicies = torch.zeros(indices.size() + torch.Size([depth]))
    else:
        encoded_indicies = torch.zeros(indices.size() + torch.Size([depth])).to(device)

    index = indices.view(indices.size() + torch.Size([1]))
    encoded_indicies = encoded_indicies.scatter_(1, index, 1)

    return encoded_indicies


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, out_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)