""" Each encoder should have following attributes and methods and be inherited from `_base.EncoderMixin`

Attributes:

    _out_channels (list of int): specify number of channels for each encoder feature tensor
    _depth (int): specify number of stages in decoder (in other words number of downsampling operations)
    _in_channels (int): default number of input channels in first Conv2d layer for encoder (usually 3)

Methods:

    forward(self, x: torch.Tensor)
        produce list of features of different spatial resolutions, each feature is a 4D torch.tensor of
        shape NCHW (features should be sorted in descending order according to spatial resolution, starting
        with resolution same as input `x` tensor).

        Input: `x` with shape (1, 3, 64, 64)
        Output: [f0, f1, f2, f3, f4, f5] - features with corresponding shapes
                [(1, 3, 64, 64), (1, 64, 32, 32), (1, 128, 16, 16), (1, 256, 8, 8),
                (1, 512, 4, 4), (1, 1024, 2, 2)] (C - dim may differ)

        also should support number of features according to specified depth, e.g. if depth = 5,
        number of feature tensors = 6 (one with same resolution as input and 5 downsampled),
        depth = 3 -> number of feature tensors = 4 (one with same resolution as input and 3 downsampled).
"""
import pdb
from copy import deepcopy

import torch.nn as nn

from .simple_res_conv import SimpleResConv
from pretrainedmodels.models.torchvision_models import pretrained_settings

from ._base import EncoderMixin


class SimpleResNetEncoder(SimpleResConv, EncoderMixin):
    def __init__(self, out_channels, depth=5, **kwargs):
        super().__init__(**kwargs)
        self._depth = depth
        self._out_channels = out_channels
        self._in_channels = 3

    def get_stages(self):
        return [
            nn.Sequential(self.conv0_0),
            nn.Sequential(self.pool, self.conv1_0),
            nn.Sequential(self.pool, self.conv2_0),
            nn.Sequential(self.pool, self.conv3_0),
            nn.Sequential(self.pool, self.conv4_0),
            self.pool,
        ]

    def forward(self, x):
        stages = self.get_stages()

        features = []
        for i in range(self._depth + 1):
            x = stages[i](x)
            features.append(x)

        return features

n1 = 24
simple_resnet_encoders = {
    "simple_res_conv": {
        "encoder": SimpleResNetEncoder,
        "pretrained_settings": None,
        "params": {
            "out_channels": (n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16, n1 * 16),
        }
    },
}
