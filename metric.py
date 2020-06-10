import torch

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
    return score

def ioU(pred, target, threshold=0.5, epsilon=1e-6, use_sigmoid=True):
    pred = pred.contiguous()
    if use_sigmoid:
        pred = torch.nn.Sigmoid()(pred)
    target = target.contiguous()
    pred = (pred > threshold).float()
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    iou = intersection/(pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) - intersection + epsilon)

    return iou.mean()

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