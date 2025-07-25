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
from copy import deepcopy

import torch.nn as nn

from .hrnet_backbone import HighResolutionNet as HRNet
from pretrainedmodels.models.torchvision_models import pretrained_settings

from ._base import EncoderMixin
from . import _utils as utils


class HRNetEncoder(nn.Module, EncoderMixin):
    def __init__(self, out_channels, depth):
        super().__init__()
        self._depth = depth
        self.model = HRNet(out_channels)
        self._out_channels = out_channels
        del self.model.incre_modules
        del self.model.downsamp_modules
        del self.model.final_layer
        del self.model.classifier

    def set_in_channels(self, in_channels):
        """Change first convolution channels"""
        if in_channels == 3:
            return

        self._in_channels = in_channels
        if self._out_channels[0] == 3:
            self._out_channels = tuple([in_channels] + list(self._out_channels)[1:])

        utils.patch_first_conv(model=self.model, in_channels=in_channels)

    def forward(self, x):
        features = [x]
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        features.append(x)
        x = self.model.conv2(x)
        x = self.model.bn2(x)
        x = self.model.relu(x)
        x = self.model.layer1(x)
        features.append(x)

        x_list = []
        for i in range(2):
            if self.model.transition1[i] is not None:
                x_list.append(self.model.transition1[i](x))
            else:
                x_list.append(x)
        # features.append(x_list)
        y_list = self.model.stage2(x_list)
        features.append(y_list)

        x_list = []
        for i in range(3):
            if self.model.transition2[i] is not None:
                if i < 2:
                    x_list.append(self.model.transition2[i](y_list[i]))
                else:
                    x_list.append(self.model.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        # features.append(x_list)
        y_list = self.model.stage3(x_list)
        features.append(y_list)

        x_list = []
        for i in range(4):
            if self.model.transition3[i] is not None:
                if i < 3:
                    x_list.append(self.model.transition3[i](y_list[i]))
                else:
                    x_list.append(self.model.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        # features.append(x_list)
        y_list = self.model.stage4(x_list)
        features.append(y_list)

        return features

    def load_state_dict(self, state_dict, **kwargs):
        for k in list(state_dict.keys()):
            if k.startswith('incre_modules'):
                state_dict.pop(k)
            if k.startswith('downsamp_modules'):
                state_dict.pop(k)
            if k.startswith('final_layer'):
                state_dict.pop(k)
            if k.startswith('classifier'):
                state_dict.pop(k)

        self.model.load_state_dict(state_dict, strict=True)


new_settings = {
    "hrnet18": {
        "imagenet": 'https://opr0mq.dm.files.1drv.com/y4mIoWpP2n-LUohHHANpC0jrOixm1FZgO2OsUtP2DwIozH5RsoYVyv_De5wDgR6XuQmirMV3C0AljLeB-zQXevfLlnQpcNeJlT9Q8LwNYDwh3TsECkMTWXCUn3vDGJWpCxQcQWKONr5VQWO1hLEKPeJbbSZ6tgbWwJHgHF7592HY7ilmGe39o5BhHz7P9QqMYLBts6V7QGoaKrr0PL3wvvR4w',
    },
    "hrnet32": {
        'imagenet': 'https://opr74a.dm.files.1drv.com/y4mKOuRSNGQQlp6wm_a9bF-UEQwp6a10xFCLhm4bqjDu6aSNW9yhDRM7qyx0vK0WTh42gEaniUVm3h7pg0H-W0yJff5qQtoAX7Zze4vOsqjoIthp-FW3nlfMD0-gcJi8IiVrMWqVOw2N3MbCud6uQQrTaEAvAdNjtjMpym1JghN-F060rSQKmgtq5R-wJe185IyW4-_c5_ItbhYpCyLxdqdEQ'
    },
    "hrnet48": {
        'imagenet': 'https://optgaw.dm.files.1drv.com/y4mWNpya38VArcDInoPaL7GfPMgcop92G6YRkabO1QTSWkCbo7djk8BFZ6LK_KHHIYE8wqeSAChU58NVFOZEvqFaoz392OgcyBrq_f8XGkusQep_oQsuQ7DPQCUrdLwyze_NlsyDGWot0L9agkQ-M_SfNr10ETlCF5R7BdKDZdupmcMXZc-IE3Ysw1bVHdOH4l-XEbEKFAi6ivPUbeqlYkRMQ'
    }
}

pretrained_settings = deepcopy(pretrained_settings)
for model_name, sources in new_settings.items():
    if model_name not in pretrained_settings:
        pretrained_settings[model_name] = {}

    for source_name, source_url in sources.items():
        pretrained_settings[model_name][source_name] = {
            "url": source_url,
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'num_classes': 1000
        }

hrnet_encoders = {
    "hrnet18": {
        "encoder": HRNetEncoder,
        "pretrained_settings": pretrained_settings["hrnet18"],
        "params": {
            'out_channels': [18, 36, 72, 144],
        }
    },
    "hrnet32": {
        "encoder": HRNetEncoder,
        "pretrained_settings": pretrained_settings["hrnet32"],
        "params": {
            'out_channels': [32, 64, 128, 256],
        }
    },
    "hrnet48": {
        "encoder": HRNetEncoder,
        "pretrained_settings": pretrained_settings["hrnet48"],
        "params": {
            'out_channels': [48, 96, 192, 384],
        }
    },
}
