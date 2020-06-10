from tqdm import tqdm
from metric import *

class Evaluator(object):
    def __init__(self, val_loader, model, device=None):
        self.val_loader = val_loader
        self.model = model
        self.device = device

    def __len__(self): return len(self.val_loader)

    def evaluate(self):
        with torch.no_grad():
            Seg_score = 0
            IOU = 0
            Cls_score = 0
            M0=0
            M1=0
            M2=0
            M3=0
            val_loader = iter(self.val_loader)
            for idx in tqdm(range(len(val_loader)), desc=f'Validation', ncols=120):
                batch = val_loader.next()
                Image = batch['img']
                Mask = batch['mask']
                CATE = batch['CATE']
                Image = Image.to(device=self.device)
                Mask = Mask.to(device=self.device)
                Mask = torch.unsqueeze(Mask, dim=1)
                CATE = CATE.to(device=self.device)

                pred_seg, pred_cls = self.model(Image)

                Seg_score += dice_coeff(pred_seg, Mask)
                IOU += ioU(pred_seg, Mask)
                Cls_score += Cls_coeff(pred_cls, CATE)
                a, b, c, d = F1_metric(pred_cls, CATE)
                M0 += a
                M1 += b
                M2 += c
                M3 += d
        Seg_score = Seg_score/len(val_loader)
        IOU = IOU/len(val_loader)
        Cls_score = Cls_score/len(val_loader)
        F1 = F1_score(M0, M1, M2, M3)
        return Seg_score, IOU, Cls_score, F1







