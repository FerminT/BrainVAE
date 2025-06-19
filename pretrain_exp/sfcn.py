import torch.nn as nn
import torch.nn.functional as F
from torch import sigmoid

class SFCN(nn.Module):
    def __init__(self, channel_number=[32, 64, 128, 256, 256, 64], output_dim=75, dropout=True):
        super(SFCN, self).__init__()
        n_layer = len(channel_number)
        self.feature_extractor = nn.Sequential()
        for i in range(n_layer):
            if i == 0:
                in_channel = 1
            else:
                in_channel = channel_number[i - 1]
            out_channel = channel_number[i]
            if i < n_layer - 1:
                self.feature_extractor.add_module('conv_%d' % i,
                                                  self.conv_layer(in_channel,
                                                                  out_channel,
                                                                  maxpool=True,
                                                                  kernel_size=3,
                                                                  padding=1))
            else:
                self.feature_extractor.add_module('conv_%d' % i,
                                                  self.conv_layer(in_channel,
                                                                  out_channel,
                                                                  maxpool=False,
                                                                  kernel_size=1,
                                                                  padding=0))
        self.classifier = nn.Sequential()
        avg_shape = [5, 6, 5]
        self.classifier.add_module('average_pool', nn.AvgPool3d(avg_shape))
        if dropout is True:
            self.classifier.add_module('dropout', nn.Dropout(0.5))
        i = n_layer
        in_channel = channel_number[-1]
        out_channel = output_dim
        self.classifier.add_module('conv_%d' % i,
                                   nn.Conv3d(in_channel, out_channel, padding=0, kernel_size=1))

    @staticmethod
    def conv_layer(in_channel, out_channel, maxpool=True, kernel_size=3, padding=0, maxpool_stride=2):
        if maxpool is True:
            layer = nn.Sequential(
                nn.Conv3d(in_channel, out_channel, padding=padding, kernel_size=kernel_size),
                nn.BatchNorm3d(out_channel),
                nn.MaxPool3d(2, stride=maxpool_stride),
                nn.ReLU(),
            )
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channel, out_channel, padding=padding, kernel_size=kernel_size),
                nn.BatchNorm3d(out_channel),
                nn.ReLU()
            )
        return layer

    def forward(self, x):
        out = list()
        x_f = self.feature_extractor(x)
        x = self.classifier(x_f)
        x = F.log_softmax(x, dim=1)
        # x = F.sigmoid(x)
        # x = F.relu(x)
        out.append(x)
        return out


class SFCN_TL(nn.Module):
    def __init__(self, pretrained_model, num_classes):
        super(SFCN_TL, self).__init__()
        in_channels = pretrained_model.classifier[-1].in_channels
        self.feature_extractor = pretrained_model.feature_extractor
        self.classifier = nn.Sequential()
        self.classifier.add_module('average_pool', nn.AvgPool3d([5, 6, 5]))
        self.classifier.add_module('dropout', nn.Dropout(0.5))
        self.classifier.add_module('flatten', nn.Flatten())
        self.classifier.add_module('fc', nn.Linear(in_features=in_channels, out_features=num_classes))
        # self.classifier.add_module('conv_6', torch.nn.Conv3d(in_channels, num_classes, padding=0, kernel_size=1))

    def forward(self, x):
        x_f = self.feature_extractor(x)
        x = self.classifier(x_f)
        # x = F.log_softmax(x, dim=1)
        x = sigmoid(x)
        return x