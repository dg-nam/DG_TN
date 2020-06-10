from TN_Data import *
import models
import torch.optim as optim
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from evaluator import Evaluator

split_portion = 0.1
image_path = './TNSCUI2020_train/image'
mask_path = './TNSCUI2020_train/mask'
csv_file = './TNSCUI2020_train/train.csv'
lr = 0.01
num_epoch = 100
batch_size = 3
device = 'cuda'




def main():

    train_set = TN_Dataset(image_path, mask_path, csv_file, 'train')
    val_set = TN_Dataset(image_path, mask_path, csv_file, 'val')
    # dataset = TN_Dataset(image_path, mask_path, csv_file)
    # data_num = dataset.__len__()
    # n_val = int(data_num*split_portion)
    # n_train = data_num - n_val
    # train_set, val_set = data.random_split(dataset, [n_train, n_val])
    train_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True)
    val_loader = data.DataLoader(val_set, batch_size=1, shuffle=False, drop_last=True)

    model = models.UNet_TN_attention(n_ch=1, n_classes=1, n_cl_classes=1)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    optimizer.zero_grad()

    lr_policy = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.978)

    model.train()
    model.to(device=device)

    criterion_Seg = torch.nn.BCEWithLogitsLoss()
    criterion_Cls = torch.nn.CrossEntropyLoss()

    criterion_Seg.to(device=device)
    criterion_Cls.to(device=device)

    cudnn.benchmark = True

    torch.cuda.empty_cache()

    evaluator = Evaluator(val_loader, model, device=device)

    global_step = 0
    for epoch in range(num_epoch):
        dataloader_model = iter(train_loader)
        epoch_loss = 0
        epoch_seg_loss = 0
        epoch_cls_loss = 0
        with tqdm(total=len(train_loader)*batch_size, unit = 'img', ncols=120) as pbar:
            for step in range(len(train_loader)):
                batch = dataloader_model.next()
                Image = batch['img']
                Mask = batch['mask']
                CATE = batch['CATE']
                Image = Image.to(device=device)
                Mask = Mask.to(device=device)
                Mask = torch.unsqueeze(Mask, dim=1)
                CATE = CATE.to(device=device)

                optimizer.zero_grad()

                pred_seg, pred_cls = model(Image)
                Seg_loss = criterion_Seg(pred_seg, Mask)
                Cls_loss = criterion_Cls(pred_cls, CATE)
                loss = Seg_loss + Cls_loss
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                epoch_seg_loss += Seg_loss.item()
                epoch_cls_loss += Cls_loss.item()
                pbar.set_postfix({'loss': epoch_loss/(step+1), 'Seg_loss': epoch_seg_loss/(step+1), 'Cls_loss': epoch_cls_loss/(step+1)})
                pbar.update(batch_size)
                pbar.set_description(f'Epoch {epoch + 1}/{num_epoch}' "[Step %d/%d]" % (step + 1, len(train_loader)))
                global_step += 1
                if step == 10:
                    break

        lr_policy.step()

        Seg_score, IOU, Cls_score, F1 = evaluator.evaluate()

        print('Seg_score:', Seg_score.item(), 'Cls_score:', Cls_score.item())
        print('IOU:', IOU.item(), 'F1_Score:', F1)
        torch.cuda.empty_cache()

if __name__=='__main__':
    main()
