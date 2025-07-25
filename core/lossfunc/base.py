import torch
from torch import nn
import torch.nn.functional as F
from smp.losses import DiceLoss


class CrossEntropyLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        if cfg.WEIGHT is not None:
            weight = torch.Tensor(cfg.WEIGHT).cuda()
        else:
            weight = None
        ignore_index = cfg.IGNORE
        self.ce_loss = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
        self.use_dice = cfg.get('USE_DICE', False)
        if self.use_dice:
            self.dice_loss = DiceLoss(mode='multiclass', from_logits=True, ignore_index=ignore_index)

    def forward(self, feat_dict, logits_dict, y):
        logits = logits_dict['logits']
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        loss = self.ce_loss(logits, y)
        if self.use_dice:
            loss += self.dice_loss(logits, y)

        return loss