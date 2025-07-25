import os
import logging
import shutil
import numpy as np
import torch
from collections import OrderedDict
import torch.nn as nn
import pdb
import re
import torch.nn.functional as F
from smp.utils.functional import _take_channels
from scipy import sparse
import prettytable as pt
import pandas as pd

eps = 1e-7


def compute_iou_per_class(confusion_matrix):
    """
    Args:
        confusion_matrix: numpy array [num_classes, num_classes] row - gt, col - pred
    Returns:
        iou_per_class: float32 [num_classes, ]
    """
    sum_over_row = np.sum(confusion_matrix, axis=1)
    sum_over_col = np.sum(confusion_matrix, axis=0)
    diag = np.diag(confusion_matrix)
    denominator = sum_over_row + sum_over_col - diag

    iou_per_class = diag / (denominator + 1e-7)

    return iou_per_class


def compute_f1_per_class(confusion_matrix, beta=1.0):
    sum_over_row = np.sum(confusion_matrix, axis=1)
    sum_over_col = np.sum(confusion_matrix, axis=0)
    diag = np.diag(confusion_matrix)
    recall_per_class = diag / (sum_over_row + eps)
    precision_per_class = diag / (sum_over_col + eps)
    F1_per_class = (1 + beta ** 2) * precision_per_class * recall_per_class / (
            (beta ** 2) * precision_per_class + recall_per_class + eps)
    return F1_per_class


def compute_kappa_score(confusion_matrix):
    confusion_matrix = confusion_matrix.astype(np.float32)
    n_classes = confusion_matrix.shape[0]
    sum0 = confusion_matrix.sum(axis=0)
    sum1 = confusion_matrix.sum(axis=1)
    expected = np.outer(sum0, sum1) / (np.sum(sum0) + eps)
    w_mat = np.ones([n_classes, n_classes])
    w_mat.flat[:: n_classes + 1] = 0
    k = np.sum(w_mat * confusion_matrix) / (np.sum(w_mat * expected) + eps)

    return 1. - k


def compute_OA(confusion_matrix):
    diag = np.diag(confusion_matrix)
    OA = np.sum(diag) / (np.sum(confusion_matrix) + eps)
    return OA


def compute_OAs(confusion_matrix):
    diag = np.diag(confusion_matrix)
    # sum_over_row = np.sum(confusion_matrix, axis=0)
    sum_over_row = np.sum(confusion_matrix, axis=1)
    recall_per_class = diag / (sum_over_row + eps)
    return recall_per_class


def lc_batch_compute_metric(output, target, ignore_channels=None):
    # generate one-hot gt
    num_classes = output.shape[1]
    _total = sparse.coo_matrix((num_classes, num_classes), dtype=np.float32)
    output = output.max(1)[1]
    if ignore_channels is not None:
        output = output[target != ignore_channels]
        target = target[target != ignore_channels]
    else:
        output = output.view(-1)
        target = target.view(-1)
    v = np.ones_like(output.cpu().numpy())
    cm = sparse.coo_matrix((v, (target.cpu().numpy(), output.cpu().numpy())), shape=(num_classes, num_classes),
                           dtype=np.float32)
    _total += cm
    dense_cm = _total.toarray()
    valid_gt = np.unique(target.detach().cpu())
    num_gt = len(valid_gt)
    if ignore_channels in valid_gt:
        num_gt = num_gt - 1

    if ignore_channels == 0:
        dense_cm = dense_cm[1:, 1:]
    elif ignore_channels > 0:
        raise NotImplementedError()

    F1s = np.round(compute_f1_per_class(dense_cm, beta=1.0), 5)
    IoUs = np.round(compute_iou_per_class(dense_cm), 5)
    kappa = np.round(compute_kappa_score(dense_cm), 5)
    OA = np.round(compute_OA(dense_cm), 5)
    OAs = np.round(compute_OAs(dense_cm), 5)

    if ignore_channels == 0:
        mF1 = F1s.sum() / num_gt
        mIoU = IoUs.sum() / num_gt
        mOA = OAs.sum() / num_gt
    elif ignore_channels is None or ignore_channels < 0:
        mF1 = F1s.sum() / num_classes
        mIoU = IoUs.sum() / num_classes
        mOA = OAs.sum() / num_classes
    else:
        raise NotImplementedError()
    keys = ['mOA', 'OA', 'mIoU', 'IoUs', 'Kappa', 'OAs', 'F1', 'mF1']
    values = [mOA, OA, mIoU, IoUs, kappa, OAs, F1s, mF1]
    return dict(zip(keys, values))


class LCMetricAverageMeter(object):
    def __init__(self, class_names, ignore_channels=None):
        self.class_names = class_names
        self.num_classes = len(class_names)
        self._total = sparse.coo_matrix((self.num_classes, self.num_classes), dtype=np.float32)
        self.ignore_cls_id = ignore_channels

    def get_dense_cm(self):
        dense_cm = self._total.toarray()
        if self.ignore_cls_id is None or self.ignore_cls_id < 0:
            output = dense_cm
        elif self.ignore_cls_id == 0:
            output = dense_cm[1:, 1:]
        elif self.ignore_cls_id > 0:
            raise NotImplementedError()
        else:
            raise NotImplementedError()
        return output

    def update(self, output, target):
        # generate one-hot gt
        num_classes = output.shape[1]
        output = output.max(1)[1]

        if self.ignore_cls_id is not None:
            output = output[target != self.ignore_cls_id]
            target = target[target != self.ignore_cls_id]
        else:
            output = output.view(-1)
            target = target.view(-1)
        v = np.ones_like(output.cpu().numpy())
        cm = sparse.coo_matrix((v, (target.cpu().numpy(), output.cpu().numpy())), shape=(num_classes, num_classes),
                               dtype=np.float32)
        self._total += cm

    def cm_value(self):
        dense_cm = self.get_dense_cm()
        row_sum = np.sum(dense_cm, axis=1)
        dense_cm /= row_sum[:, None]
        return dense_cm

    def compute_iou_per_class(self, confusion_matrix):
        """
        Args:
            confusion_matrix: numpy array [num_classes, num_classes] row - gt, col - pred
        Returns:
            iou_per_class: float32 [num_classes, ]
        """
        sum_over_col = np.sum(confusion_matrix, axis=0)
        sum_over_row = np.sum(confusion_matrix, axis=1)
        diag = np.diag(confusion_matrix)
        denominator = sum_over_row + sum_over_col - diag

        iou_per_class = diag / (denominator + eps)

        return iou_per_class

    def compute_total(self, confusion_matrix):
        diag = np.diag(confusion_matrix)
        sum_over_col = np.sum(confusion_matrix, axis=0)
        sum_over_row = np.sum(confusion_matrix, axis=1)
        TP = np.sum(diag)
        FN = np.sum(sum_over_row) - TP
        FP = np.sum(sum_over_col) - TP
        return TP, FN, FP

    def compute_recall_per_class(self, confusion_matrix):
        sum_over_row = np.sum(confusion_matrix, axis=1)
        diag = np.diag(confusion_matrix)
        recall_per_class = diag / (sum_over_row + eps)
        return recall_per_class

    def compute_precision_per_class(self, confusion_matrix):
        sum_over_col = np.sum(confusion_matrix, axis=0)
        diag = np.diag(confusion_matrix)
        precision_per_class = diag / (sum_over_col + eps)
        return precision_per_class

    def compute_overall_accuracy(self, confusion_matrix):
        diag = np.diag(confusion_matrix)
        return np.sum(diag) / (np.sum(confusion_matrix) + eps)

    def compute_F_measure_per_class(self, confusion_matrix, beta=1.0):
        precision_per_class = self.compute_precision_per_class(confusion_matrix)
        recall_per_class = self.compute_recall_per_class(confusion_matrix)
        F1_per_class = (1 + beta ** 2) * precision_per_class * recall_per_class / (
                (beta ** 2) * precision_per_class + recall_per_class + eps)

        return F1_per_class

    def cohen_kappa_score(self, cm_th):
        cm_th = cm_th.astype(np.float32)
        n_classes = cm_th.shape[0]
        sum0 = cm_th.sum(axis=0)
        sum1 = cm_th.sum(axis=1)
        expected = np.outer(sum0, sum1) / (np.sum(sum0) + eps)
        w_mat = np.ones([n_classes, n_classes])
        w_mat.flat[:: n_classes + 1] = 0
        k = np.sum(w_mat * cm_th) / (np.sum(w_mat * expected) + eps)
        return 1. - k

    def summary_all(self, dec=5):
        dense_cm = self.get_dense_cm()

        iou_per_class = np.round(self.compute_iou_per_class(dense_cm), dec)
        miou = np.round(iou_per_class.mean(), dec)
        F1_per_class = np.round(self.compute_F_measure_per_class(dense_cm, beta=1.0), dec)
        mF1 = np.round(F1_per_class.mean(), dec)
        overall_accuracy = np.round(self.compute_overall_accuracy(dense_cm), dec)
        kappa = np.round(self.cohen_kappa_score(dense_cm), dec)

        precision_per_class = np.round(self.compute_precision_per_class(dense_cm), dec)
        mprec = np.round(precision_per_class.mean(), dec)
        recall_per_class = np.round(self.compute_recall_per_class(dense_cm), dec)
        mrecall = np.round(recall_per_class.mean(), dec)
        TP, FN, FP = np.round(self.compute_total(dense_cm), dec)
        precision_total = TP / (TP + FP)
        recall_total = TP / (TP + FN)
        recall_total = np.round(recall_total, dec)
        f1_total = 2 * TP / (2 * TP + FP + FN)
        f1_total = np.round(f1_total, dec)
        iou_total = TP / (TP + FP + FN)
        iou_total = np.round(iou_total, dec)

        tb = pt.PrettyTable()
        if self.class_names:
            tb.field_names = ['name', 'class', 'iou', 'f1', 'precision', 'recall']
            idxs = np.arange(self.num_classes)

            if self.ignore_cls_id is None or self.ignore_cls_id < 0:
                class_names = self.class_names
                idxs = idxs
            elif self.ignore_cls_id == 0:
                class_names = self.class_names[1:]
                idxs = idxs[1:]
            else:
                raise NotImplementedError()

            assert len(class_names) == len(iou_per_class)

            for idx, class_name, iou, f1, precision, recall in zip(idxs, class_names, iou_per_class, F1_per_class,
                                                                   precision_per_class, recall_per_class):
                tb.add_row([class_name, idx, iou, f1, precision, recall])

            tb.add_row(['', 'mean', miou, mF1, mprec, mrecall])
            tb.add_row(['', 'OA', overall_accuracy, '-', '-', '-'])
            tb.add_row(['', 'Kappa', kappa, '-', '-', '-'])

        else:
            tb.field_names = ['class', 'iou', 'f1', 'precision', 'recall']
            for idx, (iou, f1, precision, recall) in enumerate(
                    zip(iou_per_class, F1_per_class, precision_per_class, recall_per_class)):
                tb.add_row([idx, iou, f1, precision, recall])

            tb.add_row(['mean', miou, mF1, mprec, mrecall])
            tb.add_row(['OA', overall_accuracy, '-', '-', '-'])
            tb.add_row(['', 'Kappa', kappa, '-', '-', '-'])

        if self.num_classes == 2:
            self.Kappa = kappa
            self.OA = overall_accuracy
            self.mIoU = iou_per_class[1]
            self.mF1 = F1_per_class[1]
            self.mOA = recall_per_class[1]
            self.tb = tb
            self.OAs = recall_per_class
        else:
            self.Kappa = kappa
            self.OA = overall_accuracy
            self.mIoU = miou
            self.mF1 = mF1
            self.mOA = mrecall
            self.tb = tb
            self.OAs = recall_per_class

        return tb

    def prettytable_to_dataframe(self, tb: pt.PrettyTable):
        heads = tb.field_names
        data = tb._rows
        df = pd.DataFrame(data, columns=heads)
        return df

    def prettytable_to_csv(self, csv_file: str):
        df = self.prettytable_to_dataframe(self.tb)
        df.to_csv(csv_file)

    def summary_all_to_csv(self, csv_file: str, dec=5):
        tb = self.summary_all(dec)
        df = self.prettytable_to_dataframe(tb)
        df.to_csv(csv_file)

    def save_cm(self, path):
        np.save(path, self._total.toarray().astype(float))
