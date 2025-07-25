from copy import deepcopy

import torch.nn as nn
import torch.nn.functional as F

from .swin_transformer_backbone import SwinTransformer
from .swin_transformerv2_backbone import SwinTransformerV2
from pretrainedmodels.models.torchvision_models import pretrained_settings


class SwinEncoder(SwinTransformer):
    def __init__(self, out_channels, depth, **kwargs):
        super().__init__(**kwargs)
        self._depth = depth
        self._out_channels = out_channels
        self._in_channels = 3

        del self.avgpool
        del self.head

    def set_in_channels(self, in_channels):
        """Change first convolution channels"""
        if in_channels == 3:
            return

        self._in_channels = in_channels
        if self._out_channels[0] == 3:
            self._out_channels = tuple([in_channels] + list(self._out_channels)[1:])

        from .swin_transformer_backbone import PatchEmbed
        self.patch_embed = PatchEmbed(in_chans=in_channels)

    def forward(self, x):
        features = [x]
        x = self.patch_embed(x)
        features.append(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)
            features.append(x)

        return features

    def load_state_dict(self, state_dict, **kwargs):
        state_dict = state_dict['model']
        if 'head.weight' in state_dict:
            state_dict.pop("head.weight")
        if 'head.bias' in state_dict:
            state_dict.pop("head.bias")
        super().load_state_dict(state_dict, **kwargs)


class SwinEncoderV2(SwinTransformerV2):
    def __init__(self, out_channels, depth, **kwargs):
        super().__init__(**kwargs)
        self._depth = depth
        self._out_channels = out_channels
        self.out_channels = self._out_channels[: self._depth + 1]
        self._in_channels = 3

        del self.avgpool
        del self.head

    def set_in_channels(self, in_channels):
        """Change first convolution channels"""
        if in_channels == 3:
            return

        self._in_channels = in_channels
        if self._out_channels[0] == 3:
            self._out_channels = tuple([in_channels] + list(self._out_channels)[1:])

        from .swin_transformerv2_backbone import PatchEmbed
        self.patch_embed = PatchEmbed(in_chans=in_channels)

    def linear2patch(self, x):
        b, l, c = x.size()
        h = int(l ** 0.5)
        assert h ** 2 == l
        x_out = x.view(b, -1, h, c).permute(0, 3, 1, 2).contiguous()
        return x_out

    def patch2linear(self, x):
        b, c, w, h = x.size()
        x_out = x.view(b, c, w * h).permute(0, 2, 1).contiguous()
        return x_out

    def get_stages(self):
        return nn.ModuleList([
            nn.Sequential(
                self.patch_embed,
                self.pos_drop
            ),
            *self.layers
        ])

    def forward(self, x):
        features = [x]
        x = self.patch_embed(x)
        features.append(self.linear2patch(x))
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)
            features.append(self.linear2patch(x))

        return features

    def load_state_dict(self, state_dict, **kwargs):
        state_dict = state_dict['model']
        new_state_dict = {}
        local_state = super().state_dict()
        miss_keys = set()
        for k, v in local_state.items():
            if k in state_dict and state_dict[k].size() == local_state[k].size():
                new_state_dict[k] = v
            else:
                miss_keys.add(k)
        super().load_state_dict(new_state_dict, strict=False)
        print('miss keys: {}'.format(miss_keys))


new_settings = {
    "swin_t": {
        "imagenet": 'https://github.com/SwinTransformer/storage/releases/download/v1.0.8/swin_tiny_patch4_window7_224_22k.pth',
    },
    "swin_s": {
        'imagenet': 'https://github.com/SwinTransformer/storage/releases/download/v1.0.8/swin_small_patch4_window7_224_22k.pth'
    },
    "swin_b": {
        'imagenet': 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth'
    },
    "swin_l": {
        'imagenet': 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22k.pth'
    },
    "swinv2_s_w8": {
        'imagenet': 'https://github.com/SwinTransformer/storage/releases/download/v2.0.0/swinv2_small_patch4_window8_256.pth'
    },
    "swinv2_s_w16": {
        'imagenet': 'https://github.com/SwinTransformer/storage/releases/download/v2.0.0/swinv2_small_patch4_window16_256.pth'
    },
    "swinv2_b_w8": {
        'imagenet': 'https://github.com/SwinTransformer/storage/releases/download/v2.0.0/swinv2_base_patch4_window8_256.pth'
    },
    "swinv2_b_w16": {
        'imagenet': 'https://github.com/SwinTransformer/storage/releases/download/v2.0.0/swinv2_base_patch4_window16_256.pth'
    },
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

swin_encoders = {
    "swin_t": {
        "encoder": SwinEncoder,
        "pretrained_settings": pretrained_settings["swin_t"],
        "params": {
            "out_channels": (3, 48, 96, 192, 384, 768),
            'patch_size': 4,
            'embed_dim': 96,
            'depths': [2, 2, 6, 2],
            'num_heads': [3, 6, 12, 24],
            'window_size': 7,
            'drop_path_rate': 0.1,
        },
    },
    "swin_s": {
        "encoder": SwinEncoder,
        "pretrained_settings": pretrained_settings["swin_s"],
        "params": {
            "out_channels": (3, 48, 96, 192, 384, 768),
            'patch_size': 4,
            'embed_dim': 96,
            'depths': [2, 2, 18, 2],
            'num_heads': [3, 6, 12, 24],
            'window_size': 8,
            'drop_path_rate': 0.3,
        },
    },
    "swin_b": {
        "encoder": SwinEncoder,
        "pretrained_settings": pretrained_settings["swin_b"],
        "params": {
            "out_channels": (3, 48, 128, 256, 512, 1024),
            'patch_size': 4,
            'embed_dim': 128,
            'depths': [2, 2, 18, 2],
            'num_heads': [4, 8, 16, 32],
            'window_size': 7,
            'drop_path_rate': 0.5,
        },
    },
    "swinv2_t_w8": {
        "encoder": SwinEncoderV2,
        "pretrained_settings": pretrained_settings["swinv2_s_w8"],
        "params": {
            "out_channels": (3, 96, 96, 192, 384, 768),
            'patch_size': 4,
            'embed_dim': 96,
            'depths': [2, 2, 6, 2],
            'num_heads': [3, 6, 12, 24],
            'window_size': 8,
            'drop_path_rate': 0.2,
        },
    },
    "swinv2_s_w8": {
        "encoder": SwinEncoderV2,
        "pretrained_settings": pretrained_settings["swinv2_s_w8"],
        "params": {
            "out_channels": (3, 96, 96, 192, 384, 768),
            'patch_size': 4,
            'embed_dim': 96,
            'depths': [2, 2, 18, 2],
            'num_heads': [3, 6, 12, 24],
            'window_size': 8,
            'drop_path_rate': 0.3,
        },
    },
    "swinv2_s_w16": {
        "encoder": SwinEncoderV2,
        "pretrained_settings": pretrained_settings["swinv2_s_w16"],
        "params": {
            "out_channels": (3, 96, 96, 192, 384, 768),
            'patch_size': 4,
            'embed_dim': 96,
            'depths': [2, 2, 18, 2],
            'num_heads': [3, 6, 12, 24],
            'window_size': 16,
            'drop_path_rate': 0.3,
        },
    },
}
