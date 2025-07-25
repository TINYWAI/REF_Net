import os
import logging
import shutil
import numpy as np
import torch
from collections import OrderedDict
import torch.nn as nn
import pdb
import re


def create_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def create_logger(name, log_file, level=logging.INFO):
    l = logging.getLogger(name)
    formatter = logging.Formatter('[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s')
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    l.setLevel(level)
    l.addHandler(fh)
    l.addHandler(sh)
    return l


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.all = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.all = self.sum / 3600


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    target_all = target.view(1, -1).expand_as(pred)
    # all
    correct = pred.eq(target_all)

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def save_state(state, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    model_path = '{}/checkpoints/iter_{}_checkpoint.pth.tar'.format(save_path, state['step'])
    latest_path = '{}/checkpoints/latest_checkpoint.pth.tar'.format(save_path)
    torch.save(state, model_path)
    shutil.copyfile(model_path, latest_path)


def save_add_loss_state(state, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    model_path = '{}/checkpoints/iter_{}_{}_checkpoint.pth.tar'.format(save_path, state['step'], state['add_loss_name'])
    latest_path = '{}/checkpoints/latest_{}_checkpoint.pth.tar'.format(save_path, state['add_loss_name'])
    torch.save(state, model_path)
    shutil.copyfile(model_path, latest_path)


def load_add_loss_state(path, model):
    def map_func(storage, location):
        return storage.cuda()

    if os.path.isfile(path):
        checkpoint = torch.load(path, map_location=map_func)
    else:
        assert True, "=> no checkpoint found at '{}'".format(path)
    model.load_state_dict(checkpoint['state_dict'])


def load_state(path, model, logger=None, latest_flag=True, optimizer=None):
    # pdb.set_trace()
    def map_func(storage, location):
        return storage.cuda()

    if os.path.isfile(path):
        checkpoint = torch.load(path, map_location='cpu')
        if 'state_dict' not in checkpoint and 'step' not in checkpoint:
            checkpoint = {'state_dict': checkpoint, 'step': -1}
    elif latest_flag and not os.path.isfile(path):
        latest_file = '{}/checkpoints/latest_checkpoint.pth.tar'.format(path)
        checkpoint = torch.load(latest_file, map_location='cpu')
    else:
        assert True, "=> no checkpoint found at '{}'".format(path)
    ckpt_keys = set(checkpoint['state_dict'].keys())
    own_keys = set(model.state_dict().keys())
    missing_keys = own_keys - ckpt_keys
    for k in missing_keys:
        if logger != None:
            logger.info('caution: missing keys from checkpoint {}: {}'.format(path, k))
        else:
            print('caution: missing keys from checkpoint {}: {}'.format(path, k))

    copying_layers = {}
    ignoring_layers = {}
    for key in own_keys:
        if key not in ckpt_keys:
            continue
        if checkpoint['state_dict'][key].shape == model.state_dict()[key].shape:
            copying_layers[key] = checkpoint['state_dict'][key]
        else:
            ignoring_layers[key] = checkpoint['state_dict'][key]
            if logger != None:
                logger.info('caution: shape mismatched keys from checkpoint {}: {}'.format(path, key))
            else:
                print('caution: shape mismatched keys from checkpoint {}: {}'.format(path, key))

    model.load_state_dict(copying_layers, strict=False)
    eval_iteration = checkpoint['step']
    if optimizer != None:
        optimizer.load_state_dict(checkpoint['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
        if logger != None:
            logger.info("=> also loaded optimizer from checkpoint '{}' (iter {})".format(path, eval_iteration))
        else:
            print("=> also loaded optimizer from checkpoint '{}' (iter {})".format(path, eval_iteration))
    return eval_iteration


def load_optimizer(optimizer, ckpt):
    optimizer.load_state_dict(ckpt)
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda()


def load_imgnet_models(model, path, logger):
    def map_func(storage, location):
        return storage.cuda()

    if os.path.isfile(path):
        state_dict = torch.load(path, map_location=map_func)
        if 'densenet' in path:
            pattern = re.compile(
                r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')

            for key in list(state_dict.keys()):
                res = pattern.match(key)
                if res:
                    new_key = res.group(1) + res.group(2)
                    state_dict[new_key] = state_dict[key]
                    del state_dict[key]
        mapped_state_dict = OrderedDict()
        for key, value in state_dict.items():
            # print(key)
            mapped_key = key
            mapped_state_dict[mapped_key] = value
            if 'running_var' in key:
                mapped_state_dict[key.replace('running_var', 'num_batches_tracked')] = torch.zeros(1)
        if 'vgg16' in path:
            mapped_state_dict['fc0.0.weight'] = mapped_state_dict['classifier.0.weight']
            mapped_state_dict['fc0.0.bias'] = mapped_state_dict['classifier.0.bias']
            mapped_state_dict['fc1.0.weight'] = mapped_state_dict['classifier.3.weight']
            mapped_state_dict['fc1.0.bias'] = mapped_state_dict['classifier.3.bias']
        elif 'alexnet' in path:
            mapped_state_dict['fc0.0.weight'] = mapped_state_dict['classifier.1.weight']
            mapped_state_dict['fc0.0.bias'] = mapped_state_dict['classifier.1.bias']
            mapped_state_dict['fc1.0.weight'] = mapped_state_dict['classifier.4.weight']
            mapped_state_dict['fc1.0.bias'] = mapped_state_dict['classifier.4.bias']

        state_dict = mapped_state_dict
    else:
        assert True, "=> no checkpoint found at '{}'".format(path)

    ckpt_keys = set(state_dict.keys())
    own_keys = set(model.state_dict().keys())
    missing_keys = own_keys - ckpt_keys
    # pdb.set_trace()
    for k in missing_keys:
        logger.info('caution: missing keys from checkpoint {}: {}'.format(path, k))

    model.load_state_dict(state_dict, False)


def param_groups(model):
    conv_weight_group = []
    conv_bias_group = []
    bn_group = []
    feature_weight_group = []
    feature_bias_group = []
    classification_fc_group = []

    normal_group = []
    arranged_names = set()

    for name, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            bn_group.append(m.weight)
            bn_group.append(m.bias)
            arranged_names.add(name + '.weight')
            arranged_names.add(name + '.bias')
        elif isinstance(m, nn.Conv2d):
            conv_weight_group.append(m.weight)
            if m.bias is not None:
                conv_bias_group.append(m.bias)
            arranged_names.add(name + '.weight')
            arranged_names.add(name + '.bias')
        elif isinstance(m, nn.Linear):
            if m.out_features == model.num_classes:
                classification_fc_group.append(m.weight)
                if m.bias is not None:
                    classification_fc_group.append(m.bias)
            else:
                feature_weight_group.append(m.weight)
                if m.bias is not None:
                    feature_bias_group.append(m.bias)

            arranged_names.add(name + '.weight')
            arranged_names.add(name + '.bias')

    for name, param in model.named_parameters():
        if name in arranged_names:
            continue
        else:
            normal_group.append(param)

    return conv_weight_group, conv_bias_group, bn_group, \
        feature_weight_group, feature_bias_group, classification_fc_group, \
        normal_group
