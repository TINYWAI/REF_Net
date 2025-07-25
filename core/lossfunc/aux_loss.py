import torch
from torch import nn
# import torchvision.transforms.functional as trF
import torch.nn.functional as F


class AuxCrossEntropyLoss2d(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        ignore_label = cfg.get('IGNORE')
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_label)
        self.keys = cfg.get('KEYS')
        self.aux_weight = cfg.get('WEIGHT', 1.0)
        self.use_dice = cfg.get('USE_DICE', False)

    def forward(self, feat_dict, logits_dict, targets):
        loss = 0
        for key in self.keys:
            logits = logits_dict[key]
            B, C, H, W = logits.size()
            targets_ = targets.clone().unsqueeze(1).float()
            targets_ = F.interpolate(targets_, size=(H, W), mode='nearest')
            targets_ = targets_.squeeze(1).long()
            loss += self.ce_loss(logits, targets_)
            if self.use_dice:
                loss += self.dice_loss(logits, targets_)

        return self.aux_weight * loss
