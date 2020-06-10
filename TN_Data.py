import torch.utils.data as data
import os, cv2
import pandas as pd
import numpy as np
from glob import glob
import torch

class TN_Dataset(data.Dataset):
    def __init__(self, img_path, mask_path, csv_path, mode='train'):
        self.img_path = img_path
        self.mask_path = mask_path
        self.csv_path = csv_path
        self.mode = mode
        self.ID_list, self.CATE_list = self.get_images_name(self.csv_path)
        if mode == 'train':
            self.start_num = 0
        elif mode == 'val':
            self.start_num = 3280

        print("Found %d images" % len(self.ID_list), self.mode)

    def __len__(self): return len(self.ID_list)

    def __getitem__(self, index):
        id = self.ID_list[index+self.start_num]
        cate = self.CATE_list[index+self.start_num]
        if cate == 1:
            CATE = np.array(1)
        else:
            CATE = np.array(0)

        CATE = torch.from_numpy(CATE).type(torch.LongTensor)

        image = cv2.imread(os.path.join(self.img_path, id), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(os.path.join(self.mask_path, id), cv2.IMREAD_GRAYSCALE)
        image_size = image.shape[:2]
        image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
        image =image/255
        mask = mask/255
        image = torch.from_numpy(image).type(torch.FloatTensor)
        mask = torch.from_numpy(mask).type(torch.FloatTensor)
        image = torch.unsqueeze(image, axis=0)
        # mask = torch.unsqueeze(mask, axix=0)
        output_dict = dict(img=image, mask=mask, ID=id, CATE=CATE, size=image_size)

        return output_dict


    def get_images_name(self, csv_path):
        df = pd.read_csv(csv_path)
        if self.mode == 'train':
            ID_list = df['ID'][:3280]
            CATE_list = df['CATE'][:3280]
        elif self.mode == 'val':
            ID_list = df['ID'][3280:]
            CATE_list = df['CATE'][3280:]

        return ID_list, CATE_list