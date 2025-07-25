import os
import sys
import threading

this_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
lib_path = os.path.join('../', this_dir)
if lib_path not in sys.path:
    sys.path.insert(0, lib_path)
import time
import yaml
import shutil
import logging
import argparse
import numpy as np
from tqdm import tqdm
from skimage import io
from copy import deepcopy
from easydict import EasyDict
from tensorboardX import SummaryWriter

import torch
import torch.optim
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

import core.lossfunc as lossfunc
import core.models as models
import datasets.datasets_catalog as dc
from core.utils.mm_dataset import MM2Dataset
from core.utils.lc_metric import lc_batch_compute_metric, LCMetricAverageMeter
from core.utils.sampler_utils import set_seed_, mm2_collate, TrainIterationSampler
from core.utils.misc import create_dirs, create_logger, AverageMeter, save_state, load_state
from core.utils.lr_scheduler import IterLRScheduler, IterPolyLRScheduler, WarmupCosineLR

parser = argparse.ArgumentParser(description='PyTorch Multimodal Segmentation Training')
parser.add_argument('--config', default='cfgs/paper/wos_bs10_CMFNet_paper_set_vgg16.yaml')
parser.add_argument('--resume_opt', action='store_true')
parser.add_argument('--latest_flag', action='store_true')

args = parser.parse_args()
with open(args.config) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    cfg = EasyDict(config)
    file_name = os.path.splitext(os.path.basename(args.config))[0]
    arch = file_name.split('_')[-1]
    cfg.TRAIN.CKPT = os.path.join(cfg.TRAIN.CKPT, arch, file_name)

device_ids = cfg.TRAIN.DEVICE_IDS
torch.cuda.set_device(device_ids[0])
set_seed_(cfg.TRAIN.SEED)
thread_max_num = threading.Semaphore(4)


def resize_labels(input_labels, h, w):
    labels = deepcopy(input_labels)
    if len(labels.shape) == 3:
        labels = labels.unsqueeze(1).float().clone()
    labels = torch.nn.functional.interpolate(labels,
                                             (h, w), mode='nearest')
    labels = labels.squeeze(1).long()
    return labels


def main():
    create_dirs('{}/events'.format(cfg.TRAIN.CKPT))
    create_dirs('{}/checkpoints'.format(cfg.TRAIN.CKPT))
    create_dirs('{}/val_results'.format(cfg.TRAIN.CKPT))
    create_dirs('{}/logs'.format(cfg.TRAIN.CKPT))
    if os.path.exists(cfg.TRAIN.CKPT) and not args.resume_opt and os.listdir(
            '{}/checkpoints'.format(cfg.TRAIN.CKPT)) != []:
        print('exp exists!\r\n')
        sys.exit(0)
    logger_path = '{}/logs/log.txt'.format(cfg.TRAIN.CKPT)
    if os.path.exists(logger_path):
        logger_path = '{}/logs/log_resume.txt'.format(cfg.TRAIN.CKPT)
    logger = create_logger('global_logger', logger_path)
    logger.info('{}'.format(cfg))
    global tfb_logger
    tfb_logger = SummaryWriter('{}/events'.format(cfg.TRAIN.CKPT))
    shutil.copyfile(args.config, os.path.join(cfg.TRAIN.CKPT, args.config.split('/')[-1]))

    model = models.__dict__[cfg.MODEL.type](cfg=cfg.MODEL)
    logger.info(
        "=> creating model: \n{}".format(model))
    # logger.info(
    #     stat(model, (cfg.MODEL.opt_in_channels, *cfg.AUG.INPUT_SIZE), (cfg.MODEL.sar_in_channels, *cfg.AUG.INPUT_SIZE))
    # )

    train_dataset = MM2Dataset(cfg=cfg, mode='train')
    logger.info('train_set num: {}'.format(len(train_dataset)))
    if cfg.TRAIN.get('VAL') is True:
        val_dataset = MM2Dataset(cfg=cfg, mode='val')
        logger.info('val_set num: {}'.format(len(val_dataset)))
    if cfg.TRAIN.get('TEST') is True:
        test_dataset = MM2Dataset(cfg=cfg, mode='test')
        logger.info('test_set num: {}'.format(len(test_dataset)))

    init_lr = cfg.SOLVER.BASE_LR
    params = [
        {'params': model.parameters()},
    ]

    if cfg.SOLVER.OPTIM == 'SGD':
        optimizer = torch.optim.SGD(params, init_lr,
                                    momentum=cfg.SOLVER.MOMENTUM,
                                    weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    elif cfg.SOLVER.OPTIM == 'Adam':
        optimizer = torch.optim.Adam(params, init_lr,
                                     weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    elif cfg.SOLVER.OPTIM == 'AdamW':
        optimizer = torch.optim.AdamW(params, init_lr,
                                      weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    else:
        raise NotImplementedError()

    logger.info(optimizer)
    latest_iter = -1
    # optionally resume from a checkpoint
    if cfg.TRAIN.get('FINETUNE') is True:
        ckpt = torch.load(cfg.TRAIN.FINETUNE_PATH, map_location='cpu')['state_dict']
        ckpt_ = {}
        for k, v in ckpt.items():
            if cfg.MODEL.opt_in_channels != cfg.MODEL.sar_in_channels and k == 'opt_encoder.model.conv1.weight': continue
            k = k.replace('opt_encoder', 'sar_encoder')
            ckpt_[k] = v
        model.load_state_dict(ckpt_, strict=False)
        logger.info('load finetune model: {}'.format(cfg.TRAIN.FINETUNE_PATH))

    if args.resume_opt:
        if cfg.TRAIN.LOAD_PATH != '' and args.latest_flag is False:
            logger.info('=> loading model: {}\n'.format(cfg.TRAIN.LOAD_PATH))
            latest_iter = load_state(cfg.TRAIN.LOAD_PATH, model, logger, latest_flag=False, optimizer=optimizer)
        elif args.latest_flag and cfg.TRAIN.LOAD_PATH == '':
            logger.info('=> loading latest saved model\n')
            latest_iter = load_state(cfg.TRAIN.CKPT, model, logger, latest_flag=True, optimizer=optimizer)
        else:
            assert True, 'wrong resume option'

    model = torch.nn.DataParallel(model.cuda(), device_ids=device_ids)

    criterion = lossfunc.__dict__[cfg.MAIN_LOSS.TYPE](cfg=cfg.MAIN_LOSS).cuda()
    extra_criterion = None
    if 'EXTRA_LOSS' in cfg:
        extra_criterion = lossfunc.__dict__[cfg.EXTRA_LOSS.TYPE](cfg=cfg.EXTRA_LOSS).cuda()

    train_sampler = TrainIterationSampler(dataset=train_dataset, total_iter=cfg.SOLVER.MAX_ITER,
                                          batch_size=train_dataset.batch_size, last_iter=latest_iter)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=train_dataset.batch_size,
        shuffle=False,
        num_workers=cfg.TRAIN.WORKERS, pin_memory=False, sampler=train_sampler, collate_fn=mm2_collate)
    if cfg.TRAIN.get('VAL') is True:
        val_loader = torch.utils.data.DataLoader(
            dataset=val_dataset,
            batch_size=val_dataset.batch_size, shuffle=False, drop_last=False,
            num_workers=cfg.TRAIN.WORKERS, pin_memory=False, collate_fn=mm2_collate)
    else:
        val_loader = None
    if cfg.TRAIN.get('TEST') is True:
        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=test_dataset.batch_size, shuffle=False, drop_last=False,
            num_workers=cfg.TRAIN.WORKERS, pin_memory=False, collate_fn=mm2_collate)
    else:
        test_loader = None

    if cfg.SOLVER.TYPE == 'IterLRScheduler':
        lr_scheduler = IterLRScheduler(optimizer, cfg.SOLVER.LR_STEPS, cfg.SOLVER.LR_MULTS, latest_iter=latest_iter)
    elif cfg.SOLVER.TYPE == 'IterPolyLRScheduler':
        lr_scheduler = IterPolyLRScheduler(optimizer, cfg.SOLVER.MAX_ITER, cfg.SOLVER.MIN_LR, power=cfg.SOLVER.POWER,
                                           cur_iter=latest_iter)
    elif cfg.SOLVER.TYPE == 'WarmupCosineLR':
        lr_scheduler = WarmupCosineLR(optimizer, cfg.SOLVER.MAX_ITER, warmup_iters=cfg.SOLVER.WARMUP_STEPS,
                                      last_epoch=latest_iter)
    else:
        raise NotImplementedError()
    train_val(cfg, train_loader, val_loader, test_loader, model, criterion, extra_criterion, optimizer, lr_scheduler,
              latest_iter + 1)
    tfb_logger.close()


def train_val(cfg, train_loader, val_loader, test_loader, model, criterion, extra_criterion, optimizer, lr_scheduler,
              start_iter):
    logger = logging.getLogger('global_logger')
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    add_losses = AverageMeter()
    oas = AverageMeter()
    mious = AverageMeter()
    moas = AverageMeter()
    kappas = AverageMeter()
    # switch to train mode
    model.train()
    end = time.time()
    for i, data in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        input_A = data[0].cuda()
        input_B = data[1].cuda()
        target = data[2].long().cuda()

        curr_step = start_iter + i
        lr_scheduler.step(curr_step)
        current_lr = lr_scheduler.get_lr()[0]

        # compute output
        feat_dict, logits_dict = model(input_A, input_B, target)
        main_loss = criterion(feat_dict, logits_dict, target)
        main_loss = main_loss.float()

        add_loss = None
        if extra_criterion is not None:
            add_loss = extra_criterion(feat_dict, logits_dict, target)
            add_loss = add_loss.float()
            add_losses.update(add_loss.item())
        loss = main_loss if add_loss is None else main_loss + add_loss

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = logits_dict['logits']
        if isinstance(output, (list, tuple)):
            output = output[0]
        output = output.float()

        # ['TP', 'TN', 'FP', 'FN', 'Pcc', 'Kappa', 'Pr', 'Re', 'F1']
        batch_val = lc_batch_compute_metric(output, target, ignore_channels=cfg.MAIN_LOSS.IGNORE)
        losses.update(loss.item())
        oas.update(batch_val['OA'].item())
        moas.update(batch_val['mOA'].item())
        mious.update(batch_val['mIoU'].item())
        kappas.update(batch_val['Kappa'].item())
        # measure elapsed time
        batch_time.update(time.time() - end)

        if curr_step % cfg.SOLVER.PRINT_FREQ == 0:
            tfb_logger.add_scalar('train_{}/loss'.format(cfg.TRAIN.MODAL), losses.val, curr_step)
            tfb_logger.add_scalar('train_{}/OA'.format(cfg.TRAIN.MODAL), oas.val, curr_step)
            # tfb_logger.add_scalar('train_{}/mOA'.format(cfg.TRAIN.MODAL), moas.val, curr_step)
            tfb_logger.add_scalar('train_{}/mIoU'.format(cfg.TRAIN.MODAL), mious.val, curr_step)
            tfb_logger.add_scalar('train_{}/Kappa'.format(cfg.TRAIN.MODAL), kappas.val, curr_step)
            tfb_logger.add_scalar('lr', current_lr, curr_step)

            logger.info('Cfg: {cfg} |'
                        'Iter: [{0}/{1}] |'
                        'Time {batch_time.avg:.3f}({batch_time.val:.3f}) |'
                        'Data {data_time.avg:.3f}({data_time.val:.3f}) |'
                        'Loss {loss.avg:.4f}({loss.val:.4f}) |'
                        '{extra_loss_info}'
                        'OA {oas.avg:.3f}({oas.val:.3f}) |'
                        'mOA {moas.avg:.3f}({moas.val:.3f}) |'
                        'mIoU {mious.avg:.3f}({mious.val:.3f}) |'
                        'Kappa {kappas.avg:.3f}({kappas.val:.3f}) |'
                        'LR {lr:.6f} |'
                        'Total {batch_time.all:.2f}hrs |'.format(
                curr_step, len(train_loader) + start_iter,
                dataset_name=cfg.TRAIN.DATASETS,
                batch_time=batch_time, data_time=data_time,
                loss=losses, extra_loss_info='Extra_loss {add_losses.avg:.4f}({add_losses.val:.4f}) |'.format(
                    add_losses=add_losses) if add_loss is not None else '',
                oas=oas, moas=moas, kappas=kappas, mious=mious,
                lr=current_lr, cfg=os.path.basename(args.config)))

        if (curr_step + 1) % cfg.SOLVER.SNAPSHOT == 0:
            save_state({
                'step': curr_step + 1,
                'dataset_name': cfg.TRAIN.DATASETS,
                'type': cfg.MODEL.type,
                'opt_backbone': cfg.MODEL.opt_encoder_name if 'opt_encoder_name' in cfg.MODEL else None,
                'sar_backbone': cfg.MODEL.sar_encoder_name if 'sar_encoder_name' in cfg.MODEL else None,
                'state_dict': model.module.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, cfg.TRAIN.CKPT)

            eval_iteration = curr_step + 1
            if cfg.TRAIN.get('VAL') is True:
                val(cfg, eval_iteration, logger, val_loader, model, mode='val')
            if cfg.TRAIN.get('TEST') is True:
                val(cfg, eval_iteration, logger, test_loader, model, mode='test')

        end = time.time()


def val(cfg, eval_iteration, logger, loader, model, mode):
    val_metric = LCMetricAverageMeter(
        class_names=list(dc.get_vis_colors(cfg.TRAIN.DATASETS).keys()),
        ignore_channels=cfg.MAIN_LOSS.IGNORE,
    )
    logger.info('iter {}: {} opt_sar ...'.format(eval_iteration, mode))

    model.eval()
    pbar = tqdm(total=len(loader.dataset))
    with torch.no_grad():
        for i, data in enumerate(loader):
            input_A = data[0].cuda()
            input_B = data[1].cuda()
            target = data[2].long().cuda()
            # compute output
            feat_dict, logits_dict = model(input_A, input_B)
            output = logits_dict['logits']
            if isinstance(output, (list, tuple)):
                output = output[0]

            output = output.float()
            current_batch_size = input_A.shape[0]
            pbar.update(current_batch_size)

            val_metric.update(output, target)
        pbar.close()

        tb = val_metric.summary_all()
        logger.info(tb)

        tfb_logger.add_scalar('{}_opt_sar/OA'.format(mode), val_metric.OA, eval_iteration)
        tfb_logger.add_scalar('{}_opt_sar/mF1'.format(mode), val_metric.mF1, eval_iteration)
        tfb_logger.add_scalar('{}_opt_sar/mIoU'.format(mode), val_metric.mIoU, eval_iteration)
        tfb_logger.add_scalar('{}_opt_sar/Kappa'.format(mode), val_metric.Kappa, eval_iteration)

    model.train()


def post_process_work(param):
    # file_name = os.path.basename(test_dataset.metas[index + j][2])
    # pred = pred_maps[j]
    file_name, pred, eval_iteration, state, output, target = param
    pred = pred.astype(np.uint8)
    save_path = '{}/val_results/pred_{}/{}'.format(cfg.TRAIN.CKPT, eval_iteration, file_name)
    if cfg.VAL.SAVE_PRED is True:
        io.imsave(save_path, pred, check_contrast=False)

    if cfg.VAL.SAVE_VIS is True:
        pred_color = label_map_color(pred)
        save_color_path = '{}/val_results/pred_color_{}/{}'.format(cfg.TRAIN.CKPT, eval_iteration, file_name)
        io.imsave(save_color_path, pred_color, check_contrast=False)

    batch_val = lc_batch_compute_metric(output, target, ignore_channels=cfg.MAIN_LOSS.IGNORE)

    with open('{}/val_results/patch_results_{}_{}.csv'.format(cfg.TRAIN.CKPT, state, eval_iteration), 'a') as fout:
        patch_msg = '{}, {:.4f}, {:.4f}, {:.4f}, '.format(file_name,
                                                          batch_val['mIoU'].item(),
                                                          batch_val['OA'].item(),
                                                          batch_val['mOA'].item())
        patch_msg = patch_msg + ', '

        for acc in batch_val['OAs']:
            patch_msg = patch_msg + '{:.4f}, '.format(acc)
        patch_msg = patch_msg + '\r\n'
        fout.write(patch_msg)


def label_map_color(masked_pred, cls_color_map=dc.get_vis_colors(cfg.TRAIN.DATASETS)):
    cm = np.array(list(cls_color_map.values())).astype(np.uint8)
    color_img = cm[masked_pred]
    return color_img


if __name__ == '__main__':
    main()
