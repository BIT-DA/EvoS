import torch
import torch.nn as nn
import torch.nn.functional as F


class CIDAYearbookFeature(nn.Module):
    def __init__(self, args, num_input_channels):
        super(CIDAYearbookFeature, self).__init__()
        self.args = args
        self.enc = nn.Sequential(self.conv_block(num_input_channels, 32), self.conv_block(32, 32),
                                 self.conv_block(32, 32), self.conv_block(32, 32))
        self.output_dim = 32

        self.domain_fc = nn.Linear(1, 32)
        self.combine_fc = nn.Linear(64, 32)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    def forward(self, x, domain):
        x = self.enc(x)
        x = torch.mean(x, dim=(2, 3))

        if len(domain.shape) == 1:
            domain = domain.unsqueeze(1)

        domain_latent = self.domain_fc(domain)
        fea = self.combine_fc(torch.cat((x, domain_latent), dim=-1))

        return fea


class CIDARotatedMNISTFeature(nn.Module):
    def __init__(self, args, num_input_channels):
        super(CIDARotatedMNISTFeature, self).__init__()
        self.args = args

        self.conv1 = nn.Conv2d(num_input_channels, 64, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 128, 3, 1, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, 1, padding=1)

        self.bn0 = nn.GroupNorm(8, 64)
        self.bn1 = nn.GroupNorm(8, 128)
        self.bn2 = nn.GroupNorm(8, 128)
        self.bn3 = nn.GroupNorm(8, 128)

        self.relu = nn.ReLU()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.enc = nn.Sequential(self.conv1, self.relu, self.bn0, self.conv2, self.relu, self.bn1,
                                 self.conv3, self.relu, self.bn2, self.conv4, self.relu, self.bn3,
                                 self.avgpool)
        self.output_dim = 128

        self.domain_fc = nn.Linear(1, 32)
        self.combine_fc = nn.Linear(128 + 32, 128)

    def forward(self, x, domain):
        x = self.enc(x)
        x = x.view(len(x), -1)  # [b, 128]

        if len(domain.shape) == 1:
            domain = domain.unsqueeze(1)

        domain_latent = self.domain_fc(domain)
        fea = self.combine_fc(torch.cat((x, domain_latent), dim=-1))

        return fea


class CIDAClassifier(nn.Module):
    def __init__(self, in_feature, num_classes):
        super().__init__()
        self.in_feature = in_feature
        self.num_classes = num_classes

        self.fc = nn.Linear(in_feature, num_classes)

    def forward(self, x):
        return self.fc(x)


class CIDADiscriminator(nn.Module):
    def __init__(self, in_feature, hidden_feature=32):
        super().__init__()
        self.in_feature = in_feature
        self.hidden_feature = hidden_feature
        self.drop_x = nn.Dropout(0.3)

        self.proj_m = nn.Linear(in_feature, hidden_feature)
        self.bn_m = nn.BatchNorm1d(hidden_feature)
        self.drop_m = nn.Dropout(0.3)

        self.proj_s = nn.Linear(in_feature, hidden_feature)
        self.bn_s = nn.BatchNorm1d(hidden_feature)
        self.drop_s = nn.Dropout(0.3)

        self.fc_m = nn.Linear(hidden_feature, 1)
        self.fc_s = nn.Linear(hidden_feature, 1)

    def forward(self, x):
        x = self.drop_x(x)

        x_m = F.relu(self.bn_m(self.proj_m(x)))
        x_m = self.drop_m(x_m)

        x_s = F.relu(self.bn_s(self.proj_s(x)))
        x_s = self.drop_s(x_s)

        x_m = self.fc_m(x_m)
        x_s = self.fc_s(x_s)  # log sigma^2

        return x_m, x_s

