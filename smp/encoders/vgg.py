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

import torch.nn as nn
from torchvision.models.vgg import VGG
from torchvision.models.vgg import make_layers
from pretrainedmodels.models.torchvision_models import pretrained_settings

from ._base import EncoderMixin
from typing import List, Union, cast

# fmt: off
cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'sim1': [16, 16, 'M', 32, 32, 'M', 64, 64, 64, 'M', 128, 128, 128, 'M'],
    'sim2': [16, 16, 'M', 32, 32, 'M', 64, 64, 'M', 128, 128, 'M', 256, 256, 'M'],
}
# fmt: on


class VGGEncoder(VGG, EncoderMixin):
    def __init__(self, out_channels, config, batch_norm=False, begin=0, depth=5, **kwargs):
        super().__init__(make_layers(config, batch_norm=batch_norm))
        self._out_channels = out_channels
        self._begin = begin
        self._depth = depth
        self._in_channels = 3
        del self.classifier

    def make_dilated(self, stage_list, dilation_list):
        raise ValueError("'VGG' models do not support dilated mode due to Max Pooling"
                         " operations for downsampling!")

    def get_stages(self):
        stages = []
        stage_modules = []
        for module in self.features:
            if isinstance(module, nn.MaxPool2d):
                stages.append(nn.Sequential(*stage_modules))
                stage_modules = []
            stage_modules.append(module)
        stages.append(nn.Sequential(*stage_modules))
        return stages

    def forward(self, x):
        stages = self.get_stages()

        features = []
        for i in range(self._begin, self._depth + 1):
            x = stages[i](x)
            features.append(x)

        return features

    def load_state_dict(self, state_dict, **kwargs):
        keys = list(state_dict.keys())
        for k in keys:
            if k.startswith("classifier"):
                state_dict.pop(k)
        super().load_state_dict(state_dict, **kwargs)


class VirtualVGGEncoder(nn.Module, EncoderMixin):
    def __init__(self, out_channels, config, batch_norm=False, **kwargs):
        super().__init__()
        self.features = self.make_layers(cfg=config, batch_norm=batch_norm)
        self._out_channels = out_channels

    def make_layers(self, cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
        layers: List[nn.Module] = []
        expand_ratio = 1
        in_channels = 3
        for v in cfg:
            if v == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                expand_ratio = 3
            else:
                v = cast(int, v)
                in_channels = in_channels * expand_ratio
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
                expand_ratio = 1
        return nn.Sequential(*layers)

    def make_dilated(self, stage_list, dilation_list):
        raise ValueError("'VGG' models do not support dilated mode due to Max Pooling"
                         " operations for downsampling!")


vgg_encoders = {
    "vgg11": {
        "encoder": VGGEncoder,
        "pretrained_settings": pretrained_settings["vgg11"],
        "params": {
            "out_channels": (64, 128, 256, 512, 512, 512),
            "config": cfg["A"],
            "batch_norm": False,
        },
    },
    "vgg11_bn": {
        "encoder": VGGEncoder,
        "pretrained_settings": pretrained_settings["vgg11_bn"],
        "params": {
            "out_channels": (64, 128, 256, 512, 512, 512),
            "config": cfg["A"],
            "batch_norm": True,
        },
    },
    "vgg13": {
        "encoder": VGGEncoder,
        "pretrained_settings": pretrained_settings["vgg13"],
        "params": {
            "out_channels": (64, 128, 256, 512, 512, 512),
            "config": cfg["B"],
            "batch_norm": False,
        },
    },
    "vgg13_bn": {
        "encoder": VGGEncoder,
        "pretrained_settings": pretrained_settings["vgg13_bn"],
        "params": {
            "out_channels": (64, 128, 256, 512, 512, 512),
            "config": cfg["B"],
            "batch_norm": True,
        },
    },
    "vgg16": {
        "encoder": VGGEncoder,
        "pretrained_settings": pretrained_settings["vgg16"],
        "params": {
            "out_channels": (64, 128, 256, 512, 512, 512),
            "config": cfg["D"],
            "batch_norm": False,
        },
    },
    "virtual_vgg16_bn": {
        "encoder": VirtualVGGEncoder,
        "pretrained_settings": pretrained_settings["vgg16_bn"],
        "params": {
            "out_channels": (128, 256, 512, 512, 512),
            "config": [128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
            "batch_norm": True,
        },
    },
    "vgg16_bn": {
        "encoder": VGGEncoder,
        "pretrained_settings": pretrained_settings["vgg16_bn"],
        "params": {
            "out_channels": (64, 128, 256, 512, 512, 512),
            "config": cfg["D"],
            "batch_norm": True,
        },
    },
    "vgg19": {
        "encoder": VGGEncoder,
        "pretrained_settings": pretrained_settings["vgg19"],
        "params": {
            "out_channels": (64, 128, 256, 512, 512, 512),
            "config": cfg["E"],
            "batch_norm": False,
        },
    },
    "vgg19_bn": {
        "encoder": VGGEncoder,
        "pretrained_settings": pretrained_settings["vgg19_bn"],
        "params": {
            "out_channels": (64, 128, 256, 512, 512, 512),
            "config": cfg["E"],
            "batch_norm": True,
        },
    },
    "vgg_sim1": {
        "encoder": VGGEncoder,
        "pretrained_settings": None,
        "params": {
            "out_channels": (16, 32, 64, 128, 128),
            "config": cfg["sim1"],
            "batch_norm": True,
        },
    },
    "vgg_sim2": {
        "encoder": VGGEncoder,
        "pretrained_settings": None,
        "params": {
            "out_channels": (16, 32, 64, 128, 256, 256),
            "config": cfg["sim2"],
            "batch_norm": True,
        },
    },
}
