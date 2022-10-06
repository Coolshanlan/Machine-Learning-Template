from email.mime import image
import torch.nn as nn
from  torch.utils.data import Dataset,DataLoader
import torch
import cv2

class GeneralImageDataset(Dataset):
    def __init__(self,df,transform=None):
        self.df =  df
        self.image_paths = df.image_path.values # to numpy array
        self.labels=df.label.values # to numpy array
        self.transform = transform

    def read_image(self,path):
        image=cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform :
            image = self.transform(image=image)['image']
        return image

    def __getitem__(self,idx):
        data = self.read_image(self.image_paths[idx])
        return data,self.labels[idx]

    def __len__(self):
        return len(self.image_paths)


class GeneralNumericalDataset(Dataset):
    def __init__(self,df,transform=None):
        self.df =  df
        self.features = df.features.values # to numpy array
        self.labels=df.label.values # to numpy array

    def __getitem__(self,idx):
        return self.features[idx],self.labels[idx]

    def __len__(self):
        return len(self.labels)
# transform = A.Compose([
#                          A.Resize(cfg.image_size[0],cfg.image_size[1],always_apply=True),
#                          A.ShiftScaleRotate(shift_limit=0.1,scale_limit=0.2,rotate_limit=0,p=0.5),
#                          A.RandomBrightnessContrast(brightness_limit=(-0.15,0.15), contrast_limit=(-0.15, 0.15), p=0.5),
#                          A.HorizontalFlip(p=0.5),
#                          A.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5]),#不可刪除
#                          ToTensorV2()#不可刪除
#                         ])

def get_dataloader(df,
                    transform=None,
                    batch_size=32,
                    shuffle=True,
                    num_workers=4,
                    drop_last=True,
                    pin_memory=True):
    dataset = GeneralImageDataset(df,transform)
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            num_workers=num_workers,
                            drop_last=drop_last,
                            pin_memory=pin_memory)
    return dataset,dataloader