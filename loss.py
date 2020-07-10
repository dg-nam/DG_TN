from metric import *
import torch.nn as nn

class Criterion_DG(nn.Module):
    def __init__(self, base_criterion=torch.nn.BCEWithLogitsLoss, sub_criterion=tversky_loss, beta=0.5):
        super(Criterion_DG, self).__init__()
        self.base_criterion = base_criterion()
        self.sub_criterion = sub_criterion
        self.beta = beta

    def forward(self, preds, target):
        if isinstance(target, tuple) or isinstance(target, list):
            gt = target[0].float().cuda()
            if len(gt.shape) == 3:
                gt = torch.unsqueeze(gt, axis=1)
            target[1] = target[1].float().cuda()
            target[1] = torch.unsqueeze(target[1], axis=1)
            if (isinstance(preds, tuple) or isinstance(preds, list)):
                for i in range(len(preds)):
                    preds[i] = preds[i] * target[1]
            else:
                preds = preds * target[1]
        else:
            gt = target.float().cuda()
            if len(gt.shape) == 3:
                gt = torch.unsqueeze(gt, axis=1)
        if (isinstance(preds, tuple) or isinstance(preds, list)) and len(preds) == 2:
            main_loss = self.base_criterion(preds[0], gt) + 0.4 * self.base_criterion(preds[1], gt)
            sub_loss = self.sub_criterion(preds[0], gt) + 0.4 * self.sub_criterion(preds[1], gt)
        elif (isinstance(preds, tuple) or isinstance(preds, list)):
            main_loss = 0
            sub_loss = 0
            for i in range(len(preds)):
                weight = 0.4**i
                main_loss += weight * self.base_criterion(preds[i], gt)
                sub_loss += weight * self.sub_criterion(preds[i], gt)
        else:
            main_loss = self.base_criterion(preds, gt)
            sub_loss = self.sub_criterion(preds, gt)

        return main_loss + sub_loss
