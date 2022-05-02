import torch
from torch.utils import data
from albumentations.pytorch import ToTensorV2
import albumentations as A
import numpy as np
import cv2
class Mydataset_no_read_class(data.Dataset):
    def __init__(self, img_paths, labels, transform):
        self.imgs = img_paths
        self.labels = labels
        self.transforms = transform

    def __getitem__(self, index):
        img = self.imgs[index]
        label = self.labels[index]
        labels = int(label)
        data = self.transforms(image=img)

        return data['image'], labels

    def __len__(self):
        return len(self.imgs)




class Mydataset(data.Dataset):
    def __init__(self, img_paths, labels, transform):
        self.imgs = img_paths
        self.labels = labels
        self.transforms = transform

    def __getitem__(self, index):
        img = self.imgs[index]
        label = self.labels[index]
        labels = int(label)
        data = self.transforms(image=img)

        return data['image'], labels

    def __len__(self):
        return len(self.imgs)

class Mydataset_cat(data.Dataset):
    def __init__(self, img_paths, mask_cat__,labels, transform):
        self.imgs = img_paths
        self.mask_cat = mask_cat__
        self.labels = labels
        self.transforms = transform

    def __getitem__(self, index):
        # print(index)
        img = (self.imgs[index]).copy()
        mask_cat_here = (self.mask_cat[index]).copy()
        label = self.labels[index]
        labels = int(label)
        data = self.transforms(image=img,mask=mask_cat_here)

        return data['image'],((data['mask'])/255), labels

    def __len__(self):
        return len(self.imgs)



class Mydataset2(data.Dataset):
    def __init__(self, img_paths, labels, transform):
        self.imgs = img_paths
        self.labels = labels
        self.transforms = transform

    def __getitem__(self, index):
        img = self.imgs[index]
        label = self.labels[index]
        # labels = int(label)
        if int(label) == 0:
            labels = int(label)
        else:
            labels = 1
        data = self.transforms(image=img)

        return data['image'], labels

    def __len__(self):
        return len(self.imgs)


class Mydataset_class_seg(data.Dataset):
    def __init__(self, img_paths, masks, labels,transform):
        self.imgs = img_paths
        self.masks = masks
        self.labels = labels
        self.transforms = transform

    def __getitem__(self, index):
        img0 = self.imgs[index]
        mask = self.masks[index]
        label = self.labels[index]
        labels = int(label)
        img = self.transforms(image=img0, mask=mask)

        return img['image'], (img['mask']/255).long(),labels#, label

    def __len__(self):
        return len(self.imgs)






def for_train_transform():
    # aug_size=int(size/10)
    train_transform = A.Compose([
        A.RandomRotate90(),
        A.Flip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0, scale_limit=(-0.2,0.1), rotate_limit=40, p=0.5),
        # A.RandomBrightnessContrast(
        #     brightness_limit=0.5,
        #     contrast_limit=0.1,
        #     p=0.5
        # ),
        # A.HueSaturationValue(hue_shift_limit=20,sat_shift_limit=100, val_shift_limit=80),
        # A.OneOf([
        #     A.CoarseDropout(max_holes=100,max_height=aug_size,max_width=aug_size,fill_value=[239, 234, 238]),
        #     A.GaussNoise()
        # ]),
        A.GaussNoise(),
        # A.OneOf([
        #     # A.ElasticTransform(),
        #     # A.GridDistortion(),
        #     # A.OpticalDistortion(distort_limit=0.5,shift_limit=0)
        # ]),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std= [0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0
        ),
        ToTensorV2()], p=1.)
    return train_transform


test_transform = A.Compose([
    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std= [0.229, 0.224, 0.225],
        max_pixel_value=255.0,
        p=1.0
    ),
    ToTensorV2()], p=1.)

def for_train_transform_05(size):
    aug_size=int(size/10)
    train_transform = A.Compose([
        A.RandomRotate90(),
        A.Flip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0, scale_limit=(-0.2,0.1), rotate_limit=40, p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=0.5,
            contrast_limit=0.1,
            p=0.5
        ),
        A.HueSaturationValue(hue_shift_limit=20,sat_shift_limit=100, val_shift_limit=80),
        A.OneOf([
            A.CoarseDropout(max_holes=100,max_height=aug_size,max_width=aug_size,fill_value=[239, 234, 238]),
            A.GaussNoise()
        ]),
        A.OneOf([
            A.ElasticTransform(),
            A.GridDistortion(),
            A.OpticalDistortion(distort_limit=0.5,shift_limit=0)
        ]),
        A.Normalize(
            mean=[0.5, 0.5, 0.5],
            std= [0.5, 0.5, 0.5],
            max_pixel_value=255.0,
            p=1.0
        ),
        ToTensorV2()], p=1.)
    return train_transform

test_transform_05 = A.Compose([
    A.Normalize(
        mean=[0.5, 0.5, 0.5],
        std= [0.5, 0.5, 0.5],
        max_pixel_value=255.0,
        p=1.0
    ),
    ToTensorV2()], p=1.)

import os
def get_image_paths(dataroot):

    paths = []
    if dataroot is not None:
        paths_img = os.listdir(dataroot)
        for _ in sorted(paths_img):
            path = os.path.join(dataroot, _)
            paths.append(path)
    return paths
class Mydataset_for_pre(data.Dataset):
    def __init__(self, img_paths,  resize,transform = test_transform):
        self.imgs = get_image_paths(img_paths)
        self.transforms = transform
        self.resize = resize
    def __getitem__(self, index):
        img_path = self.imgs[index]

        img = cv2.resize(cv2.imread(img_path), (self.resize,self.resize))[:,:,::-1]
        img = self.transforms(image=img)

        return img['image'],#, label

    def __len__(self):
        return len(self.imgs)