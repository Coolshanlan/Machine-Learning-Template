import torch.nn as nn
from  torch.utils.data import Dataset,DataLoader
import torch
import cv2

class ImageDataset(Dataset):
    def __init__(self,image_paths,labels,transform=None):
        self.image_paths = image_paths
        self.labels= labels
        self.transform = transform

    def read_image(self,path):
        image=cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform :
            image = self.transform(image)
        return image

    def __getitem__(self,idx):
        data = self.read_image(self.image_paths[idx])
        return data,self.labels[idx]

    def __len__(self):
        return len(self.image_paths)

class SemanticImageDataset(Dataset):
    def __init__(self,image_paths,label_paths,transform=None):
        self.image_paths = image_paths
        self.label_paths= label_paths
        self.transform = transform

    def read_image(self,path):
        image=cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def __getitem__(self,idx):
        data = self.read_image(self.image_paths[idx])
        label = self.read_image(self.label_paths[idx])
        data,label = self.transform(data,label)
        return data,label

    def __len__(self):
        return len(self.image_paths)

class NormalDataset(Dataset):
    def __init__(self,data,labels):
        self.data = data
        self.labels=labels

    def __getitem__(self,idx):
        return self.data[idx],self.labels[idx]

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

# def get_dataloader(data,
#                    labels,
#                    transform=None,
#                    batch_size=32,
#                    shuffle=True,
#                    num_workers=4,
#                    drop_last=True,
#                    pin_memory=True):
#     def image_transform(image):
#         pass
#     dataset = ImageDataset(image_paths=data,
#                            labels=labels,
#                            transform=image_transform)
#     dataloader = DataLoader(dataset,
#                             batch_size=batch_size,
#                             shuffle=shuffle,
#                             num_workers=num_workers,
#                             drop_last=drop_last,
#                             pin_memory=pin_memory)
#     return dataset,dataloader