import torch
from sklearn.metrics import jaccard_score
import numpy as np


def dice_loss(pred, target, epsilon=1e-7, use_sigmoid=True):
    pred = pred.contiguous()
    if use_sigmoid:
        pred = torch.nn.Sigmoid()(pred)
    target = target.contiguous()
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    loss = (1 - ((2. * intersection + epsilon) / (
            pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + epsilon)))
    return loss.mean()

def tversky_loss(pred, target, beta=0.5, epsilon=1e-7, use_sigmoid=True):
    pred = pred.contiguous()
    if use_sigmoid:
        pred = torch.nn.Sigmoid()(pred)
    target = target.contiguous()
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    loss = (1 - ((intersection + epsilon) / (
            intersection + beta * (pred.sum(dim=2).sum(dim=2) - intersection) + (1 - beta) * (
            target.sum(dim=2).sum(dim=2) - intersection) + epsilon)))
    return loss.mean()

def dice_coeff(pred, target, threshold=0.5, epsilon=1e-6, use_sigmoid=True):
    # make sure the tensors are align in memory and convert to probabilities if needed
    pred = pred.contiguous()
    if use_sigmoid:
        pred = torch.nn.Sigmoid()(pred)
    target = target.contiguous()
    pred = (pred > threshold).float()
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    dice = (2. * intersection + epsilon) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + epsilon)
    return dice.mean()

def Cls_coeff(pred, target, use_softmax=True):
    if use_softmax:
        pred = torch.nn.Softmax(dim=1)(pred)
    score = pred[0][int(target.item())]
    # print(pred[0], target.item(), score)
    return score.item()

def ioU(pred, target, threshold=0.5, epsilon=1e-6, use_sigmoid=True):
    pred = pred.contiguous()
    if use_sigmoid:
        pred = torch.nn.Sigmoid()(pred)
    target = target.contiguous()
    pred = (pred > threshold).float()
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    iou = (intersection + epsilon)/(pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) - intersection + epsilon)

    return iou.mean()

def TN_IOU(pred, target, threshold = 0.5, use_sigmoid=True):
    pred = pred.contiguous()
    if use_sigmoid:
        pred =torch.nn.Sigmoid()(pred)
    target = target.contiguous()
    pred = (pred > threshold).float()
    target = target.detach().cpu().clone().numpy().flatten()
    pred = pred.detach().cpu().clone().numpy().flatten()
    IOU = jaccard_score(target, pred)

    return IOU

def F1_metric(pred, target, use_softmax=True):
    if use_softmax:
        pred = torch.nn.Softmax(dim=1)(pred)

    T0_P0= 0
    T1_P0= 0
    T1_P1= 0
    T0_P1= 0

    target_num = int(target.item())
    if pred[0][0] > pred[0][1]:
        pred_num = 0
    else:
        pred_num = 1

    if target_num==0 and pred_num==0:
        T0_P0 += 1
    elif target_num==0 and pred_num==1:
        T0_P1 += 1
    elif target_num==1 and pred_num==1:
        T1_P1 += 1
    elif target_num==1 and pred_num==0:
        T1_P0 += 1

    return T0_P0, T0_P1, T1_P0, T1_P1

def F1_score(T0_P0, T0_P1, T1_P0, T1_P1, epsilon = 1e-6):
    precision0 = T0_P0/(T0_P0 + T1_P0 + epsilon)
    precision1 = T1_P1/(T1_P1 + T0_P1 + epsilon)
    recall0 = T0_P0 / (T0_P0 + T0_P1 + epsilon)
    recall1 = T1_P1 / (T1_P1 + T1_P0 + epsilon)

    avg_precision = (precision0 + precision1)/2
    avg_recall = (recall0 + recall1) / 2

    F1 = 2*avg_precision*avg_recall/(avg_precision+avg_recall)

    return F1

def classification_correct(mask, pred, threshold=0.5, use_sigmoid =True):
    pred = pred.contiguous()
    if use_sigmoid:
        pred = torch.nn.Sigmoid()(pred)
    mask = mask.contiguous()
    pred = (pred > threshold).float()
    if mask[0, 0, :, :].sum() > mask[0, 1, :, :].sum():
        cate = 0
    else:
        cate = 1

    if pred[0, 0, :, :].sum() > pred[0, 1, :, :].sum():
        pred_cate = 0
    else:
        pred_cate = 1

    if pred_cate == cate:
        score = 1
    else:
        score = 0

    return score


def class_Metric(mask, pred):
    if mask[0, 0, :, :].sum() > mask[0, 1, :, :].sum():
        cate = 0
    else:
        cate = 1

    score = classification_correct(mask, pred)

    if cate == 0:
        class_dice = dice_coeff(pred[:, 0, :, :].unsqueeze(dim=1), mask[:, 0, :, :].unsqueeze(dim=1))
        nodule_dice = dice_coeff(pred[:, 0, :, :].unsqueeze(dim=1), mask[:, 0, :, :].unsqueeze(dim=1)+mask[:, 1, :, :].unsqueeze(dim=1))
        nodule_IOU = ioU(pred[:, 0, :, :].unsqueeze(dim=1), mask[:, 0, :, :].unsqueeze(dim=1)+mask[:, 1, :, :].unsqueeze(dim=1))

    else:
        class_dice = dice_coeff(pred[:, 1, :, :].unsqueeze(dim=1), mask[:, 1, :, :].unsqueeze(dim=1))
        nodule_dice = dice_coeff(pred[:, 1, :, :].unsqueeze(dim=1), mask[:, 0, :, :].unsqueeze(dim=1)+mask[:, 1, :, :].unsqueeze(dim=1))
        nodule_IOU = ioU(pred[:, 1, :, :].unsqueeze(dim=1), mask[:, 0, :, :].unsqueeze(dim=1)+mask[:, 1, :, :].unsqueeze(dim=1))


    if score == 0:
        class_dice = 0

    return score, class_dice, nodule_dice, nodule_IOU, cate