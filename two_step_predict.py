from TN_Data import *
import models
from tqdm import tqdm
import argparse
from efficientnet_pytorch import EfficientNet
import torch.backends.cudnn as cudnn
import cv2
import pandas as pd


def load_model(model_name, model, optimizer=None, scheduler=None):
    state = torch.load(os.path.join(model_name + '.pth'))
    model.load_state_dict(state['model'])
    if optimizer is not None:
        optimizer.load_state_dict(state['optimizer'])
    if scheduler is not None:
        scheduler.load_state_dict(state['scheduler'])
    print('model loaded')

def save_mask(path, ID, mask, size):
    save_path = os.path.join(path, ID[0])

    threshold = 0.5
    mask = torch.nn.Sigmoid()(mask)

    mask = (mask > threshold).float().detach().cpu().clone().numpy().squeeze(0).squeeze(0)

    mask = cv2.resize(mask, (size[1], size[0]), interpolation=cv2.INTER_NEAREST)

    cv2.imwrite(save_path, mask*255.0)

def main():
    test_set = TN_test_dataset(img_path)
    test_loader = data.DataLoader(test_set, batch_size=1, shuffle=False)

    segmentation_model = models.__dict__[seg_model](n_ch=1, n_classes=1)

    classification_model = EfficientNet.from_pretrained(cls_model, num_classes=2)

    load_model(seg_weight, segmentation_model)
    load_model(cls_weight, classification_model)

    segmentation_model.eval()
    classification_model.eval()

    segmentation_model.to(device=device)
    classification_model.to(device=device)

    cudnn.benchmark = True

    torch.cuda.empty_cache()

    test_data_loader = iter(test_loader)

    df = pd.DataFrame(columns=['ID', 'CATE'])

    with torch.no_grad():
        for idx in tqdm(range(len(test_loader)), desc=f'Predict', ncols=120):
            batch = test_data_loader.next()
            ID = batch['ID']
            Image = batch['img']
            img_size = batch['size']
            Image = Image.to(device=device)

            pred = segmentation_model(Image)
            if isinstance(pred, list):
                pred = pred[0]

            save_mask(key, ID, pred, img_size)

            input = torch.cat([Image, Image, pred], dim=1)

            cls_pred = classification_model(input)

            if cls_pred[0][0] < cls_pred[0][1]:
                pred_cate = 1
            else:
                pred_cate = 0

            df.loc[len(df)] = [str(ID[0]), pred_cate]


    df.to_csv(os.path.join(key, 'test.csv'), index=False)





if __name__=='__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--model_seg", type=str, default="UNet")
    args.add_argument("--model_cls", type=str, default="efficientnet-b0")
    args.add_argument("--seg_weight", type=str, default="seg")
    args.add_argument("--cls_weight", type=str, default="cls")
    args.add_argument("--key", type=str, default="test")

    config = args.parse_args()

    img_path = './TNSCUI2020_train/test'
    seg_model = config.model_seg
    cls_model = config.model_cls
    seg_weight = config.seg_weight
    cls_weight = config.cls_weight
    device = 'cuda' if torch.cuda.is_available else 'cpu'

    key = './' + config.key

    if not os.path.exists(key):
        os.mkdir(key)

    main()