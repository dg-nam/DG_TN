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
        df = self.add_weight(self.csv_path, mode=False)
        self.ID_list, self.CATE_list = self.get_images_name(df)

        print("Found %d images" % len(self.ID_list), self.mode)

    def __len__(self): return len(self.ID_list)

    def __getitem__(self, index):
        id = self.ID_list[index]
        cate = self.CATE_list[index]
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
        output_dict = dict(img=image, mask=mask, ID=id, CATE=CATE, size=image_size)

        return output_dict

    def add_weight(self, csv_path, mode=False):
        df = pd.read_csv(csv_path)
        weights = []
        if mode is False:
            for i in range(len(df)):
                id = df['ID'][i]
                weight = 1
                weights.append(weight)
        weights_df = pd.DataFrame({'weight': weights})
        df = pd.concat([df, weights_df], axis=1)
        return df


    def get_images_name(self, df):
        ID_list = []
        CATE_list = []
        if self.mode == 'train':
            for i in range(3280):
                weight = df['weight'][i]
                for j in range(weight):
                    if j == 0:
                        ID_list.append(df['ID'][i])
                        CATE_list.append(df['CATE'][i])
                    else:
                        ID_list.append('aug_' + df['ID'][i])
                        CATE_list.append(df['CATE'][i])

        elif self.mode == 'val':
            for i in range(len(df) - 3280):
                ID_list.append(df['ID'][i + 3280])
                CATE_list.append(df['CATE'][i + 3280])

        return ID_list, CATE_list

class TN_test_dataset(data.Dataset):
    def __init__(self, img_path):
        self.img_path = img_path
        self.id_list = self.get_id_list()

    def __len__(self): return len(self.id_list)

    def __getitem__(self, index):
        file_name = self.id_list[index]
        file_path = os.path.join(self.img_path, file_name)

        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        image_size = image.shape[:2]
        image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
        image = image / 255
        image = torch.from_numpy(image).type(torch.FloatTensor)
        image = torch.unsqueeze(image, axis=0)

        output_dict = dict(img=image, ID=file_name, size=image_size)
        return output_dict

    def get_id_list(self):
        id_list = os.listdir(self.img_path)
        return id_list