import math
import torch
from torch.optim.lr_scheduler import MultiStepLR, _LRScheduler


class WarmupMultiStepLR(MultiStepLR):
    def __init__(self, optimizer, milestones, gamma=0.1, warmup_factor=1.0 / 3,
                 warmup_iters=500, last_epoch=-1):
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        super().__init__(optimizer, milestones, gamma, last_epoch)

    def get_lr(self):
        if self.last_epoch <= self.warmup_iters:
            alpha = self.last_epoch / self.warmup_iters
            warmup_factor = self.warmup_factor * (1 - alpha) + alpha
            # print(self.base_lrs[0]*warmup_factor)
            return [lr * warmup_factor for lr in self.base_lrs]
        else:
            lr = super().get_lr()
        return lr


class WarmupCosineLR(_LRScheduler):
    def __init__(self, optimizer, T_max, warmup_factor=1.0 / 3, warmup_iters=500,
                 eta_min=0, last_epoch=-1):
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.T_max, self.eta_min = T_max, eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch <= self.warmup_iters:
            alpha = self.last_epoch / self.warmup_iters
            warmup_factor = self.warmup_factor * (1 - alpha) + alpha
            return [lr * warmup_factor for lr in self.base_lrs]
        else:
            return [self.eta_min + (base_lr - self.eta_min) *
                    (1 + math.cos(
                        math.pi * (self.last_epoch - self.warmup_iters) / (self.T_max - self.warmup_iters))) / 2
                    for base_lr in self.base_lrs]


class IterPolyLRScheduler(_LRScheduler):
    """Polynomial learning rate decay until step reach to max_decay_step

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        max_iter: after this step, we stop decreasing learning rate
        min_lr: scheduler stoping learning rate decay, value of learning rate must be this value
        power: The power of the polynomial.
    """

    def __init__(self, optimizer, max_iter, min_lr, power=0.9, cur_iter=-1):
        if max_iter <= 1.:
            raise ValueError('max_iter should be greater than 1.')
        self.max_iter = max_iter
        self.min_lr = min_lr
        self.power = power
        self.cur_iter = cur_iter
        super().__init__(optimizer)

    def get_lr(self):
        if self.cur_iter > self.max_iter:
            return [self.min_lr for _ in self.base_lrs]

        return [(base_lr - self.min_lr) *
                ((1 - self.cur_iter / self.max_iter) ** self.power) +
                self.min_lr for base_lr in self.base_lrs]

    def step(self, step=None):
        if step is None:
            step = self.cur_iter + 1
        self.cur_iter = step if step != 0 else 1
        if self.cur_iter <= self.max_iter:
            decay_lrs = [(base_lr - self.min_lr) *
                         ((1 - self.cur_iter / self.max_iter) ** self.power) +
                         self.min_lr for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, decay_lrs):
                param_group['lr'] = lr


class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        min_lr_mul: target learning rate = base lr * min_lr_mul
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(self, optimizer, total_epoch, min_lr_mul=0.1, after_scheduler=None):
        self.min_lr_mul = min_lr_mul
        if self.min_lr_mul > 1. or self.min_lr_mul < 0.:
            raise ValueError('min_lr_mul should be [0., 1.]')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = self.base_lrs
                    self.finished = True
                return self.after_scheduler.get_lr()
            else:
                return self.base_lrs
        else:
            return [base_lr * (self.min_lr_mul + (1. - self.min_lr_mul) * (self.last_epoch / float(self.total_epoch)))
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if self.finished and self.after_scheduler:
            return self.after_scheduler.step(epoch - self.total_epoch)
        else:
            return super(GradualWarmupScheduler, self).step(epoch)


class IterLRScheduler(object):
    def __init__(self, optimizer, milestones, lr_mults, latest_iter=-1):
        assert len(milestones) == len(lr_mults), "{} vs {}".format(milestones, lr_mults)
        self.milestones = milestones
        self.lr_mults = lr_mults
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer
        for i, group in enumerate(optimizer.param_groups):
            if 'lr' not in group:
                raise KeyError("param 'lr' is not specified "
                               "in param_groups[{}] when resuming an optimizer".format(i))
        self.latest_iter = latest_iter

    def _get_lr(self):
        try:
            pos = self.milestones.index(self.latest_iter)
        except ValueError:
            return list(map(lambda group: group['lr'], self.optimizer.param_groups))
        except:
            raise Exception('wtf?')
        return list(map(lambda group: group['lr'] * self.lr_mults[pos], self.optimizer.param_groups))

    def get_lr(self):
        return list(map(lambda group: group['lr'], self.optimizer.param_groups))

    def step(self, this_iter=None):
        if this_iter is None:
            this_iter = self.latest_iter + 1
        self.latest_iter = this_iter
        for param_group, lr in zip(self.optimizer.param_groups, self._get_lr()):
            param_group['lr'] = lr
            param_group['lr'] = lr


if __name__ == '__main__':
    optim = WarmupPolyLR()
