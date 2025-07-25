import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

import smp
from smp.encoders import get_encoder
from smp.base import initialization as init
from smp.base import SegmentationHead
from .vmamba.vmamba import VSSBlock, Permute

current_module = sys.modules[__name__]


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Concat(nn.Module):
    def __init__(self, in_channels, **kwargs):
        super().__init__()
        self.conv = smp.base.Conv2dReLU(in_channels * 2, in_channels, 3, 1)

    def forward(self, m1, m2):
        fuse = self.conv(torch.cat([m1, m2], dim=1))

        return fuse


class Add(nn.Module):
    def __init__(self, in_channels, **kwargs):
        super().__init__()
        pass

    def forward(self, m1, m2):
        return m1 + m2


class DiffShareMamba(nn.Module):
    def __init__(self, in_channels, channel_first=False, return_feat=False, norm_layer=nn.LayerNorm,
                 ssm_act_layer=nn.SiLU,
                 mlp_act_layer=nn.GELU, **kwargs):
        super().__init__()
        self.return_feat = return_feat
        self.share_st_block = nn.Sequential(*[
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=in_channels, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                     ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'],
                     ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
                     ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'],
                     ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
                     forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], mlp_act_layer=mlp_act_layer,
                     mlp_drop_rate=kwargs['mlp_drop_rate'],
                     gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        ])
        self.share_conv = smp.base.Conv2dReLU(2 * in_channels, in_channels, 3, 1)
        self.diff_st_block = nn.Sequential(*[
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=in_channels, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                     ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'],
                     ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
                     ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'],
                     ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
                     forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], mlp_act_layer=mlp_act_layer,
                     mlp_drop_rate=kwargs['mlp_drop_rate'],
                     gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        ])
        self.smooth_layer = ResBlock(in_channels=in_channels, out_channels=in_channels, stride=1)

    def forward(self, m1, m2):
        share_m1 = self.share_st_block(m1)
        share_m2 = self.share_st_block(m2)
        share_m = self.share_conv(torch.cat([share_m1, share_m2], dim=1))

        diff_m = m1 - m2
        diff_m = self.diff_st_block(diff_m)

        fuse = self.smooth_layer(share_m + diff_m)

        if self.return_feat:
            return share_m, diff_m, fuse
        return fuse


class LinkMamba(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        from .vmamba.config import vssm_block_config as kwargs
        clean_kwargs = {k: v for k, v in kwargs.items() if k not in ['norm_layer', 'ssm_act_layer', 'mlp_act_layer']}
        self.cfg = cfg
        self.multi_seg = self.cfg.get('multi_seg', False)
        decoder_channels = self.cfg.decoder_channels
        self.opt_encoder = get_encoder(
            name=self.cfg.opt_encoder_name,
            in_channels=self.cfg.opt_in_channels,
            depth=self.cfg.opt_encoder_depth,
            weights=self.cfg.opt_encoder_weights,
            custom_weights=self.cfg.custom_weights if 'custom_weights' in self.cfg else None
        )
        self.sar_encoder = get_encoder(
            name=self.cfg.sar_encoder_name,
            in_channels=self.cfg.sar_in_channels,
            depth=self.cfg.sar_encoder_depth,
            weights=self.cfg.sar_encoder_weights,
            custom_weights=self.cfg.custom_weights if 'custom_weights' in self.cfg else None
        )
        out_channels = list(self.opt_encoder.out_channels)

        self.decoder = MTMambaFusionDecoder_smooth(
            encoder_dims=out_channels,
            decoder_channels=cfg.decoder_channels,
            fuse_method=cfg.decoder_fuse_method,
            channel_first=False,
            norm_layer=nn.LayerNorm,
            ssm_act_layer=nn.SiLU,
            mlp_act_layer=nn.GELU,
            **clean_kwargs
        )
        self.seg_head = SegmentationHead(
            in_channels=decoder_channels, out_channels=self.cfg.classes,
            kernel_size=1, upsampling=self.cfg.upsampling,
        )
        if self.multi_seg:
            self.opt_seg_head = SegmentationHead(
                in_channels=decoder_channels, out_channels=self.cfg.classes,
                kernel_size=1, upsampling=self.cfg.upsampling,
            )
            self.sar_seg_head = SegmentationHead(
                in_channels=decoder_channels, out_channels=self.cfg.classes,
                kernel_size=1, upsampling=self.cfg.upsampling,
            )

        self.initialize()

    def initialize(self):
        init.initialize_decoder(self.decoder)
        init.initialize_head(self.seg_head)
        if self.multi_seg:
            init.initialize_head(self.opt_seg_head)
            init.initialize_head(self.sar_seg_head)

    def forward(self, opt_x, sar_x, labels=None):
        opt_stages = self.opt_encoder.get_stages()[1:]
        sar_stages = self.sar_encoder.get_stages()[1:]
        opt_features, sar_features = [], []
        for i, (opt_stage, sar_stage) in enumerate(zip(opt_stages, sar_stages)):
            opt_x = opt_stage(opt_x)
            sar_x = sar_stage(sar_x)
            opt_features.append(opt_x)
            sar_features.append(sar_x)

        if self.multi_seg:
            decoder_feats, opt_decoder_feats, sar_decoder_feats = self.decoder(opt_features, sar_features)
            logits = self.seg_head(decoder_feats[-1])
            opt_logits = self.opt_seg_head(opt_decoder_feats[-1])
            sar_logits = self.sar_seg_head(sar_decoder_feats[-1])
            feat_dict = {
                'opt_encoder_feats': opt_features,
                'sar_encoder_feats': sar_features,
                'decoder_feats': decoder_feats,
                'opt_decoder_feats': opt_decoder_feats,
                'sar_decoder_feats': sar_decoder_feats,
            }
            logits_dict = {
                'logits': logits,
                'opt_logits': opt_logits,
                'sar_logits': sar_logits
            }
        else:
            decoder_feats = self.decoder(opt_features, sar_features)
            logits = self.seg_head(decoder_feats)
            feat_dict = {
                'decoder_feats': decoder_feats
            }
            logits_dict = {
                'logits': logits
            }

        return feat_dict, logits_dict


class MTMambaFusionDecoder_smooth(nn.Module):
    def __init__(self, encoder_dims, decoder_channels, fuse_method, channel_first, norm_layer,
                 ssm_act_layer, mlp_act_layer, **kwargs):
        super().__init__()
        fuse_method = getattr(current_module, fuse_method)

        self.opt_proj_layer5 = conv1x1(encoder_dims[-1], decoder_channels)
        self.opt_proj_layer4 = conv1x1(encoder_dims[-2], decoder_channels)
        self.opt_proj_layer3 = conv1x1(encoder_dims[-3], decoder_channels)
        self.opt_proj_layer2 = conv1x1(encoder_dims[-4], decoder_channels)
        self.opt_proj_layer1 = conv1x1(encoder_dims[-5], decoder_channels)

        self.sar_proj_layer5 = conv1x1(encoder_dims[-1], decoder_channels)
        self.sar_proj_layer4 = conv1x1(encoder_dims[-2], decoder_channels)
        self.sar_proj_layer3 = conv1x1(encoder_dims[-3], decoder_channels)
        self.sar_proj_layer2 = conv1x1(encoder_dims[-4], decoder_channels)
        self.sar_proj_layer1 = conv1x1(encoder_dims[-5], decoder_channels)

        self.fuse_layer5 = fuse_method(decoder_channels, return_feat=True, **kwargs)
        self.fuse_layer4 = fuse_method(decoder_channels, return_feat=True, **kwargs)
        self.fuse_layer3 = fuse_method(decoder_channels, return_feat=True, **kwargs)
        self.fuse_layer2 = fuse_method(decoder_channels, return_feat=True, **kwargs)
        self.fuse_layer1 = fuse_method(decoder_channels, return_feat=True, **kwargs)

        self.st_block_5 = nn.Sequential(*[
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=decoder_channels, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                     ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'],
                     ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
                     ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'],
                     ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
                     forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], mlp_act_layer=mlp_act_layer,
                     mlp_drop_rate=kwargs['mlp_drop_rate'],
                     gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        ])
        self.st_block_4 = nn.Sequential(*[
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=decoder_channels, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                     ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'],
                     ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
                     ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'],
                     ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
                     forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], mlp_act_layer=mlp_act_layer,
                     mlp_drop_rate=kwargs['mlp_drop_rate'],
                     gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        ])
        self.st_block_3 = nn.Sequential(*[
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=decoder_channels, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                     ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'],
                     ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
                     ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'],
                     ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
                     forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], mlp_act_layer=mlp_act_layer,
                     mlp_drop_rate=kwargs['mlp_drop_rate'],
                     gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        ])
        self.st_block_2 = nn.Sequential(*[
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=decoder_channels, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                     ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'],
                     ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
                     ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'],
                     ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
                     forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], mlp_act_layer=mlp_act_layer,
                     mlp_drop_rate=kwargs['mlp_drop_rate'],
                     gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        ])
        self.st_block_1 = nn.Sequential(*[
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=decoder_channels, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                     ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'],
                     ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
                     ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'],
                     ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
                     forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'],
                     mlp_act_layer=mlp_act_layer,
                     mlp_drop_rate=kwargs['mlp_drop_rate'],
                     gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        ])

        self.opt_fuse_layer_5 = ResBlock(in_channels=decoder_channels, out_channels=decoder_channels, stride=1)
        self.sar_fuse_layer_5 = ResBlock(in_channels=decoder_channels, out_channels=decoder_channels, stride=1)
        self.opt_fuse_layer_4 = ResBlock(in_channels=decoder_channels, out_channels=decoder_channels, stride=1)
        self.sar_fuse_layer_4 = ResBlock(in_channels=decoder_channels, out_channels=decoder_channels, stride=1)
        self.opt_fuse_layer_3 = ResBlock(in_channels=decoder_channels, out_channels=decoder_channels, stride=1)
        self.sar_fuse_layer_3 = ResBlock(in_channels=decoder_channels, out_channels=decoder_channels, stride=1)
        self.opt_fuse_layer_2 = ResBlock(in_channels=decoder_channels, out_channels=decoder_channels, stride=1)
        self.sar_fuse_layer_2 = ResBlock(in_channels=decoder_channels, out_channels=decoder_channels, stride=1)
        self.opt_fuse_layer_1 = ResBlock(in_channels=decoder_channels, out_channels=decoder_channels, stride=1)
        self.sar_fuse_layer_1 = ResBlock(in_channels=decoder_channels, out_channels=decoder_channels, stride=1)

        # Smooth layer
        self.smooth_layer_41 = ResBlock(in_channels=decoder_channels, out_channels=decoder_channels, stride=1)
        self.smooth_layer_42 = ResBlock(in_channels=decoder_channels, out_channels=decoder_channels, stride=1)
        self.smooth_layer_43 = ResBlock(in_channels=decoder_channels, out_channels=decoder_channels, stride=1)

        self.smooth_layer_31 = ResBlock(in_channels=decoder_channels, out_channels=decoder_channels, stride=1)
        self.smooth_layer_32 = ResBlock(in_channels=decoder_channels, out_channels=decoder_channels, stride=1)
        self.smooth_layer_33 = ResBlock(in_channels=decoder_channels, out_channels=decoder_channels, stride=1)

        self.smooth_layer_21 = ResBlock(in_channels=decoder_channels, out_channels=decoder_channels, stride=1)
        self.smooth_layer_22 = ResBlock(in_channels=decoder_channels, out_channels=decoder_channels, stride=1)
        self.smooth_layer_23 = ResBlock(in_channels=decoder_channels, out_channels=decoder_channels, stride=1)

        self.smooth_layer_11 = ResBlock(in_channels=decoder_channels, out_channels=decoder_channels, stride=1)
        self.smooth_layer_12 = ResBlock(in_channels=decoder_channels, out_channels=decoder_channels, stride=1)
        self.smooth_layer_13 = ResBlock(in_channels=decoder_channels, out_channels=decoder_channels, stride=1)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear') + y

    def forward(self, opt_features, sar_features):
        *others, opt_feat_1, opt_feat_2, opt_feat_3, opt_feat_4, opt_feat_5 = opt_features
        *others, sar_feat_1, sar_feat_2, sar_feat_3, sar_feat_4, sar_feat_5 = sar_features

        fuse_feats, opt_feats, sar_feats = [], [], []

        opt_feat_1, sar_feat_1 = self.opt_proj_layer1(opt_feat_1), self.sar_proj_layer1(sar_feat_1)
        opt_feat_2, sar_feat_2 = self.opt_proj_layer2(opt_feat_2), self.sar_proj_layer2(sar_feat_2)
        opt_feat_3, sar_feat_3 = self.opt_proj_layer3(opt_feat_3), self.sar_proj_layer3(sar_feat_3)
        opt_feat_4, sar_feat_4 = self.opt_proj_layer4(opt_feat_4), self.sar_proj_layer4(sar_feat_4)
        opt_feat_5, sar_feat_5 = self.opt_proj_layer5(opt_feat_5), self.sar_proj_layer5(sar_feat_5)

        s5, d5, p5 = self.fuse_layer5(opt_feat_5, sar_feat_5)
        p5 = self.st_block_5(p5)
        opt_5, sar_5 = self.opt_fuse_layer_5(opt_feat_5 + d5), self.sar_fuse_layer_5(sar_feat_5 + d5)
        fuse_feats.append(p5)
        opt_feats.append(opt_5)
        sar_feats.append(sar_5)

        s4, d4, p4 = self.fuse_layer4(opt_feat_4, sar_feat_4)
        p4 = self.st_block_4(p4)
        opt_4, sar_4 = self.opt_fuse_layer_4(opt_feat_4 + d4), self.sar_fuse_layer_4(sar_feat_4 + d4)
        p4 = self._upsample_add(p5, p4)
        opt_4 = self._upsample_add(opt_5, opt_4)
        sar_4 = self._upsample_add(sar_5, sar_4)
        p4 = self.smooth_layer_41(p4)
        opt_4 = self.smooth_layer_42(opt_4)
        sar_4 = self.smooth_layer_43(sar_4)
        fuse_feats.append(p4)
        opt_feats.append(opt_4)
        sar_feats.append(sar_4)

        s3, d3, p3 = self.fuse_layer3(opt_feat_3, sar_feat_3)
        p3 = self.st_block_3(p3)
        opt_3, sar_3 = self.opt_fuse_layer_3(opt_feat_3 + d3), self.sar_fuse_layer_3(sar_feat_3 + d3)
        p3 = self._upsample_add(p4, p3)
        opt_3 = self._upsample_add(opt_4, opt_3)
        sar_3 = self._upsample_add(sar_4, sar_3)
        p3 = self.smooth_layer_31(p3)
        opt_3 = self.smooth_layer_32(opt_3)
        sar_3 = self.smooth_layer_33(sar_3)
        fuse_feats.append(p3)
        opt_feats.append(opt_3)
        sar_feats.append(sar_3)

        s2, d2, p2 = self.fuse_layer2(opt_feat_2, sar_feat_2)
        p2 = self.st_block_2(p2)
        opt_2, sar_2 = self.opt_fuse_layer_2(opt_feat_2 + d2), self.sar_fuse_layer_2(sar_feat_2 + d2)
        p2 = self._upsample_add(p3, p2)
        opt_2 = self._upsample_add(opt_3, opt_2)
        sar_2 = self._upsample_add(sar_3, sar_2)
        p2 = self.smooth_layer_21(p2)
        opt_2 = self.smooth_layer_22(opt_2)
        sar_2 = self.smooth_layer_23(sar_2)
        fuse_feats.append(p2)
        opt_feats.append(opt_2)
        sar_feats.append(sar_2)

        s1, d1, p1 = self.fuse_layer1(opt_feat_1, sar_feat_1)
        p1 = self.st_block_1(p1)
        opt_1, sar_1 = self.opt_fuse_layer_1(opt_feat_1 + d1), self.sar_fuse_layer_1(sar_feat_1 + d1)
        p1 = self._upsample_add(p2, p1)
        opt_1 = self._upsample_add(opt_2, opt_1)
        sar_1 = self._upsample_add(sar_2, sar_1)
        p1 = self.smooth_layer_11(p1)
        opt_1 = self.smooth_layer_12(opt_1)
        sar_1 = self.smooth_layer_13(sar_1)
        fuse_feats.append(p1)
        opt_feats.append(opt_1)
        sar_feats.append(sar_1)

        return fuse_feats, opt_feats, sar_feats


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
