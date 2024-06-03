import torch
import torch.nn as nn
from torch.autograd import Variable
from einops import rearrange
from lightly.models.modules import SwaVProjectionHead, SwaVPrototypes
from lightly.models.modules.heads import SimCLRProjectionHead
from .submodules import Attention, DomainDiscriminator, DomainAdversarialLoss


class YearbookNetwork(nn.Module):
    def __init__(self, args, num_input_channels, num_classes, ssl_training=False):
        super(YearbookNetwork, self).__init__()
        self.args = args
        self.enc = nn.Sequential(self.conv_block(num_input_channels, 32), self.conv_block(32, 32),
                                 self.conv_block(32, 32), self.conv_block(32, 32))
        self.hid_dim = 32
        self.classifier = nn.Linear(32, num_classes)

        # SimCLR: projection head
        if self.args.method == 'simclr':
            self.projection_head = SimCLRProjectionHead(input_dim=32, hidden_dim=32, output_dim=128)
        # SwaV: projection head and prototypes
        elif self.args.method == 'swav':
            self.projection_head = SwaVProjectionHead(input_dim=32, hidden_dim=32, output_dim=128)
            self.prototypes = SwaVPrototypes(input_dim=128, n_prototypes=32)
        self.ssl_training = ssl_training

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        x = self.enc(x)
        x = torch.mean(x, dim=(2, 3))
        if self.args.method == 'simclr' and self.ssl_training:
            return self.projection_head(x)
        elif self.args.method == 'swav' and self.ssl_training:
            x = self.projection_head(x)
            x = nn.functional.normalize(x, dim=1, p=2)
            return self.prototypes(x)
        else:
            return self.classifier(x)





class YearbookNetwork_for_EvoS(nn.Module):
    def __init__(self, args, num_input_channels, num_classes):
        super(YearbookNetwork_for_EvoS, self).__init__()
        self.args = args
        self.enc = nn.Sequential(self.conv_block(num_input_channels, 32), self.conv_block(32, 32),
                                 self.conv_block(32, 32), self.conv_block(32, 32))
        self.feature_dim = 32
        self.classifier = nn.Linear(self.feature_dim, num_classes)

        self.attention_dict = {}
        if args.dim_head is None:
            self.attention_dict[1] = Attention(dim=2 * self.feature_dim, heads=args.num_head, dim_head=2 * self.feature_dim // args.num_head).cuda()
        else:
            self.attention_dict[1] = Attention(dim=2 * self.feature_dim, heads=args.num_head, dim_head=args.dim_head).cuda()

        self.conv1d_dict = {}
        for k in range(2, args.scale + 1):
            self.conv1d_dict[k] = nn.Unfold(kernel_size=(1, k), stride=1).cuda()  # overlapping window
            if args.dim_head is None:
                self.attention_dict[k] = Attention(dim=k * 2 * self.feature_dim, heads=k * args.num_head,
                                                   dim_head=k * 2 * self.feature_dim // (k * args.num_head)).cuda()
            else:
                self.attention_dict[k] = Attention(dim=k * 2 * self.feature_dim, heads=k * args.num_head,
                                                       dim_head=args.dim_head).cuda()

        self.init_pool = {0: nn.Parameter(torch.zeros([1, 2 * self.feature_dim], requires_grad=True)),
                          1: nn.Parameter(torch.zeros([1, 2 * self.feature_dim], requires_grad=True))}
        self.memory_pool = {}

        self.domain_discriminator = DomainDiscriminator(in_feature=self.feature_dim, hidden_size=self.args.hidden_discriminator).cuda()
        self.domain_adv = DomainAdversarialLoss(self.domain_discriminator).cuda()
        self.eps = 1e-6

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    def memorize(self, timestamp, mean_logStd_t=None):
        with torch.no_grad():
            t = timestamp - self.args.init_timestamp
            if t <= 1:
                self.memory_pool[t] = Variable(self.init_pool[t].detach().clone(), requires_grad=False).cuda()
            else:
                self.memory_pool[t] = Variable(mean_logStd_t.detach().clone(), requires_grad=False).cuda()

    def get_previous_mean_logStd(self, timestamp):
        t = timestamp - self.args.init_timestamp
        if t < 2:
            raise RuntimeError(f' Timestamp {timestamp} does not require previous statistics!')
        else:
            previous_mean_logStd = None
            lower_bound = 0
            if self.args.memory_pool is not None:
                lower_bound = t - self.args.memory_pool if t - self.args.memory_pool > 0 else 0

            for i in range(lower_bound, t):
                mean_logStd_i = self.memory_pool[i]
                if previous_mean_logStd is None:
                    previous_mean_logStd = mean_logStd_i
                else:
                    previous_mean_logStd = torch.cat((previous_mean_logStd, mean_logStd_i), dim=0)
            return previous_mean_logStd.cuda()

    def foward_for_FeatureDistritbuion(self, previous_mean_logStd):
        future_out = None
        num_previous_domain = previous_mean_logStd.size(0)
        s = 0
        loss_consistency = None
        for k in range(1, self.args.scale + 1):
            if k == 1:
                out_k = self.attention_dict[k](previous_mean_logStd) # out_k.shape: [1, t-1, 2*feature_dim]
                future_out = torch.mean(out_k, dim=1)
                s += 1
            else:
                if (num_previous_domain < k + 1) or (self.args.scale > self.args.split_time - self.args.init_timestamp):
                    pass
                else:   # concatenate version
                    rearranged_input = previous_mean_logStd.view(1, previous_mean_logStd.shape[0], 1, previous_mean_logStd.shape[1])
                    patches = self.conv1d_dict[k](rearrange(rearranged_input, 'b w h c -> b c h w'))
                    patches = torch.reshape(patches, (1, previous_mean_logStd.shape[1], k*1, patches.shape[-1]))  # patches.shape:[1, d, k, slices]
                    patches = patches.permute(0, 3, 2, 1)    # to shape:[1, slices, k, d]
                    patches = rearrange(patches, 'b l k d -> b l (k d)')
                    concate_patches = patches.squeeze(0)   # to [l, kd]

                    out_k = self.attention_dict[k](concate_patches)       # out_k.shape: [1, l, kd]
                    last_out_k = torch.mean(out_k, dim=1)
                    last_out_k = last_out_k.view(1, k, 2*self.feature_dim)

                    future_out = future_out + last_out_k[:, -1, :]

                    temp_loss_consistency = torch.norm(previous_mean_logStd[-(k - 1):, :] - last_out_k.squeeze(0)[:k-1, :], p=2, dim=1)
                    if loss_consistency is None:
                        loss_consistency = torch.mean(temp_loss_consistency)
                    else:
                        loss_consistency += torch.mean(temp_loss_consistency)
                    s += 1
        future_out = future_out / s
        return future_out, loss_consistency

    def foward_encoder(self, x):
        f = self.enc(x)
        f = torch.mean(f, dim=(2, 3))
        return f

    def foward_classifier(self, normalized_f):
        logits = self.classifier(normalized_f)
        return logits

    def forward_evaluate(self, x, mean, std):
        f = self.enc(x)
        f = torch.mean(f, dim=(2, 3))
        f = (f - mean) / (std + self.eps)
        logits = self.classifier(f)
        return logits

    def forward_domain_discriminator(self, f, previous_mean_logStd):
        b, d = f.size()
        l = previous_mean_logStd.size(0)

        for i in range(0, l):
            previous_mean_i = previous_mean_logStd[i, :self.feature_dim]
            previous_logStd_i = previous_mean_logStd[i, self.feature_dim:]
            std_i = torch.exp(previous_logStd_i)

            if self.args.truncate is not None:
                w = torch.empty(b, 1).cuda()
                nn.init.trunc_normal_(w, mean=0.0, std=1.0, a=-self.args.truncate, b=self.args.truncate)
            else:
                distri = torch.distributions.Normal(loc=torch.tensor([0.0]), scale=torch.tensor([1.0]))
                w = distri.sample((b,)).cuda()

            temp_features_i = previous_mean_i.view(1, d).expand(b, d) + w.expand(b, d) * std_i.view(1, d).expand(b, d)

            if i == 0:
                previous_f = temp_features_i.detach()
            else:
                previous_f = torch.cat((previous_f, temp_features_i.detach()), dim=0)
        adv_loss = self.domain_adv(previous_f, f)
        return adv_loss

    def rest_discriminator_lr(self, max_iters):
        self.domain_adv.reset(max_iters=max_iters)

    def get_parameters(self, lr):
        params_list = []
        for k in range(1, self.args.scale + 1):
            params_list.append({"params": self.attention_dict[k].parameters(), 'lr': 1 * lr})
        params_list.extend([
                {"params": self.enc.parameters(), 'lr': 1 * lr},
                {"params": self.classifier.parameters(), 'lr': 1 * lr},
                {"params": self.init_pool[0], 'lr': 1 * lr},
                {"params": self.init_pool[1], 'lr': 1 * lr},
                {"params": self.domain_discriminator.parameters(), 'lr': 1 * lr},
            ]
        )
        return params_list






