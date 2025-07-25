from typing import List, Optional

import torch.nn as nn

from ..base import initialization as init
from ..base import SegmentationHead
from ..encoders import get_encoder
from .decoder import VGGDecoder


class SegNet(nn.Module):
    def __init__(
            self,
            encoder_name: str = "vgg16_bn",
            encoder_depth: int = 5,
            encoder_weights: Optional[str] = "imagenet",
            in_channels: int = 3,
            classes: int = 1,
            upsampling: int = 2,
    ):
        super(SegNet, self).__init__()
        self.encoder = get_encoder(
            name=encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights
        )
        self.encoder_depth = encoder_depth
        self.decoder = VGGDecoder(encoder=self.encoder)

        self.MaxPooling_indices = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.MaxUnPooling = nn.MaxUnpool2d(kernel_size=2, stride=2)

        self.seg_head = SegmentationHead(
            in_channels=64,
            out_channels=classes,
            kernel_size=1,
            upsampling=upsampling,
        )

        self.initialize()

    def initialize(self):
        init.initialize_decoder(self.decoder)
        init.initialize_head(self.seg_head)

    def get_stage(self, model, del_pooling=True):
        stages = []
        stage_modules = []
        for module in model.features:
            if isinstance(module, nn.MaxPool2d) or isinstance(module, nn.MaxUnpool2d):
                if len(stage_modules) != 0:
                    stages.append(nn.Sequential(*stage_modules))
                    stage_modules = []
                if del_pooling is True:
                    continue
            if isinstance(module, nn.AdaptiveAvgPool2d):
                continue
            stage_modules.append(module)
        stages.append(nn.Sequential(*stage_modules))
        return stages

    def feature_down(self, x):
        pooling_indices = []
        for i in range(self.encoder_depth):
            stage = self.encoder_stages[i]
            x = stage(x)
            x, idx = self.MaxPooling_indices(x)
            pooling_indices.append(idx)

        return x, pooling_indices

    def feature_up(self, x, indices):
        for stage in self.decoder_stages:
            idx = indices.pop()
            x = self.MaxUnPooling(x, idx)
            x = stage(x)
        return x

    def forward(self, x, labels=None):
        self.encoder_stages = self.get_stage(self.encoder)
        self.decoder_stages = self.get_stage(self.decoder)

        x, indices = self.feature_down(x)
        x = self.feature_up(x, indices)
        logits = self.seg_head(x)

        feat_dict = {
            'feats': x,
        }
        logits_dict = {
            'logits': logits,
        }

        return feat_dict, logits_dict