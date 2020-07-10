from TN_Data import *
import models
import torch.optim as optim
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from loss import *
import argparse
from metric import *
from efficientnet_pytorch import EfficientNet


def save_model(model_name, model, optimizer, scheduler):
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
    }
    torch.save(state, os.path.join(model_name + '.pth'))
    print('model saved')

def load_model(model_name, model, optimizer=None, scheduler=None):
    state = torch.load(os.path.join(model_name + '.pth'))
    model.load_state_dict(state['model'])
    if optimizer is not None:
        optimizer.load_state_dict(state['optimizer'])
    if scheduler is not None:
        scheduler.load_state_dict(state['scheduler'])
    print('model loaded')

def seg_evaluate(model, val_loader, device):
    seg_score = 0
    IOU_score = 0
    val_data_loader = iter(val_loader)
    with torch.no_grad():
        for idx in tqdm(range(len(val_loader)), desc=f'validation', ncols=120):
            batch = val_data_loader.next()
            Image = batch['img']
            Mask = batch['mask']
            Cate = batch['CATE']

            Image = Image.to(device=device)
            Mask = Mask.to(device=device)

            pred = model(Image)
            if isinstance(pred, list):
                pred = pred[0]

            seg_score += dice_coeff(pred, Mask)
            IOU_score += TN_IOU(pred, Mask)

        Seg_score = seg_score/len(val_loader)
        IOU_score = IOU_score/len(val_loader)

    return Seg_score, IOU_score

def cls_evaluate(model, seg_model, val_loader, device):
    val_data_loader = iter(val_loader)
    M0 = 0
    M1 = 0
    M2 = 0
    M3 = 0
    correct = 0
    with torch.no_grad():
        for idx in tqdm(range(len(val_loader)), desc=f'validation', ncols=120):
            batch = val_data_loader.next()
            Image = batch['img']
            Cate = batch['CATE']
            Image = Image.to(device=device)
            if config.GT is False:
                mask = seg_model(Image)
                if isinstance(mask, list):
                    mask = mask[0]
            else:
                Mask = batch['mask']

                mask = Mask.type(torch.FloatTensor).to(device=device)

            Cate = Cate.to(device=device)

            input = torch.cat([Image, Image, mask], dim=1)

            pred = model(input)

            if pred[0][0] < pred[0][1]:
                pred_cate = 1
            else:
                pred_cate = 0

            if pred_cate == Cate.item():
                correct += 1

            a, b, c, d = F1_metric(pred, Cate)
            M0 += a
            M1 += b
            M2 += c
            M3 += d

        F1 = F1_score(M0, M1, M2, M3)

    return F1, correct


def main():

    train_set = TN_Dataset(image_path, mask_path, csv_file, 'train')
    val_set = TN_Dataset(image_path, mask_path, csv_file, 'val')

    train_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True)
    val_loader = data.DataLoader(val_set, batch_size=1, shuffle=False, drop_last=True)

    seg_model = models.__dict__[config.model_seg](n_ch=1, n_classes=1)

    seg_optimizer = optim.Adam(seg_model.parameters(), lr=lr)
    seg_optimizer.zero_grad()

    seg_lr_policy = torch.optim.lr_scheduler.ExponentialLR(seg_optimizer, 0.978)

    seg_model.train()
    seg_model.to(device=device)

    criterion_Seg = Criterion_DG()

    criterion_Seg.to(device=device)

    cudnn.benchmark = True

    torch.cuda.empty_cache()

    global_step = 0
    best_seg_score = 0
    best_IOU_score = 0
    if load_seg_model is None:
        for epoch in range(num_epochs):
            dataloader_model = iter(train_loader)
            epoch_loss = 0
            with tqdm(total=len(train_loader)*batch_size, unit='img', ncols=120) as pbar:
                for step in range(len(train_loader)):
                    batch = dataloader_model.next()
                    Image = batch['img']
                    Mask = batch['mask']
                    Cate = batch['CATE']

                    Image = Image.to(device=device)
                    Mask = Mask.to(device=device)


                    seg_optimizer.zero_grad()

                    pred_seg = seg_model(Image)
                    Seg_loss = criterion_Seg(pred_seg, Mask)
                    loss = Seg_loss
                    loss.backward()
                    seg_optimizer.step()

                    epoch_loss += loss.item()
                    pbar.set_postfix({'loss': epoch_loss/(step+1)})
                    pbar.update(batch_size)
                    pbar.set_description(f'Epoch {epoch + 1}/{num_epochs}' "[Step %d/%d]" % (step + 1, len(train_loader)))
                    global_step += 1
                    # if step == 5:
                    #     break

            seg_score, IOU_score = seg_evaluate(seg_model, val_loader, device)
            seg_lr_policy.step()

            print('Segmentation_Dice_score : ', seg_score.item())
            print('Segmentation_IOU_score : ', IOU_score)

            if best_seg_score < seg_score:
                best_seg_score = seg_score
                best_IOU_score = IOU_score
                save_model(config.key + 'Seg_Best', seg_model, seg_optimizer, seg_lr_policy)

            torch.cuda.empty_cache()


    # cls_model = models.Vgg19(2)
    model_name = config.model_cls
    cls_model = EfficientNet.from_pretrained(model_name, num_classes=2)
    cls_optimizer = optim.Adam(cls_model.parameters(), lr=lr)
    cls_optimizer.zero_grad()

    if load_seg_model is None:
        load_model(config.key + 'Seg_Best', seg_model)
    else:
        load_model(load_seg_model, seg_model)

    seg_model.eval()
    seg_model.to(device=device)

    cls_lr_policy = torch.optim.lr_scheduler.ExponentialLR(cls_optimizer, 0.978)

    cls_model.train()
    cls_model.to(device=device)

    criterion_cls = nn.CrossEntropyLoss()
    criterion_cls.to(device=device)
    best_cls_score = 0
    best_correct = 0
    for epoch in range(num_epochs):
        dataloader_model = iter(train_loader)
        epoch_loss = 0
        with tqdm(total=len(train_loader)*batch_size, unit='img', ncols=120) as tbar:
            for step in range(len(train_loader)):
                batch = dataloader_model.next()
                Image = batch['img']
                Cate = batch['CATE']
                Image = Image.to(device=device)
                Cate = Cate.to(device=device)


                if config.GT is False:
                    with torch.no_grad():
                        pred_mask = seg_model(Image)

                    if isinstance(pred_mask, list):
                        pred_mask = pred_mask[0]
                else:
                    Mask = batch['mask']
                    pred_mask = Mask.type(torch.FloatTensor).to(device=device)
                cls_input = torch.cat([Image, Image, pred_mask], dim=1)

                cls_pred = cls_model(cls_input)
                cls_optimizer.zero_grad()
                loss = criterion_cls(cls_pred, Cate)
                loss.backward()
                cls_optimizer.step()

                epoch_loss += loss.item()
                tbar.set_postfix({'loss' : epoch_loss/(step+1)})
                tbar.update(batch_size)
                tbar.set_description(f'Cls_Epoch {epoch + 1}/{num_epochs}' "[Step %d/%d]" % (step + 1, len(train_loader)))

        cls_lr_policy.step()

        cls_score, correct = cls_evaluate(cls_model, seg_model, val_loader, device)
        torch.cuda.empty_cache()
        if best_cls_score < cls_score:
            best_cls_score = cls_score
            save_model(config.key+'cls_best', cls_model, cls_optimizer, cls_lr_policy)
        if best_correct < correct:
            best_correct = correct

        print('Classification_F1_score : ', cls_score, 'correct_num : ', correct)

    print('best_seg: (Dice)', best_seg_score, '(IOU)', best_IOU_score, 'best_cls:', best_cls_score, 'best_correct:', best_correct, '/364')





if __name__=='__main__':

    args = argparse.ArgumentParser()
    args.add_argument("--num_classes", type=int, default=2)
    args.add_argument("--lr", type=float, default=0.0005)
    args.add_argument("--num_epochs", type=int, default=100)
    args.add_argument("--model_seg", type=str, default="UNet")
    args.add_argument("--batch", type=int, default=4)
    args.add_argument("--key", type=str, default="testing")
    args.add_argument("--load_seg_model", type=str, default=None)
    args.add_argument("--model_cls", type=str, default="efficientnet-b0")
    args.add_argument("--GT", type=bool, default=False)

    config = args.parse_args()

    image_path = './TNSCUI2020_train/image'
    mask_path = './TNSCUI2020_train/mask'
    csv_file = './TNSCUI2020_train/train.csv'
    lr = config.lr
    num_epochs = config.num_epochs
    batch_size = config.batch
    device = 'cuda' if torch.cuda.is_available else 'cpu'
    predict_path = './vis/' + config.key
    load_seg_model = config.load_seg_model

    if not os.path.exists(predict_path):
        os.mkdir(predict_path)


    main()