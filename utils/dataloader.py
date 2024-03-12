import warnings
from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image
import random
import torch
import torch.nn.functional as F
import glob

def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image

class ListDataset(Dataset):
    def __init__(self, list_path, trainf='train',img_size=416, multiscale=True, transform=None):
        #with open(list_path, "r") as file:
        #    self.img_files = file.readlines()
        self.img_files = sorted(glob.glob("%s/*.*" % list_path))
        self.trainf = trainf
        self.img_size = img_size
        self.max_objects = 100
        self.multiscale = multiscale
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0
        self.transform = transform

    def __getitem__(self, index):

        # ---------
        #  Image
        # ---------
        img_path = self.img_files[index % len(self.img_files)].rstrip()
            #img_path = img_path.replace('\n','')
        img = np.array(Image.open(img_path).convert('RGB'), dtype=np.uint8)
        msk = np.array(Image.open(img_path.replace(self.trainf,"trainmask")).convert('L'), dtype=np.uint8)

        #try:
        #    img_path = self.img_files[index % len(self.img_files)].rstrip()
            #img_path = img_path.replace('\n','')
        #    img = np.array(Image.open(img_path).convert('RGB'), dtype=np.uint8)
        #    msk = np.array(Image.open(img_path.replace("images","seglabs")).convert('L'), dtype=np.uint8)

        #except Exception:
        #    print(f"Could not read image '{img_path}'.")
        #    return
        # -----------
        #  Transform
        # -----------
        if self.transform:
            img = self.transform(img)
            msk = self.transform(msk)
            # try:
            #     img = self.transform(img)
            #     msk = self.transform(msk)
            # except Exception:
            #     print("Could not apply transform.")
            #     return
        #print('bb_targets==', img.shape, msk.shape)
        return img_path, img, msk

    def collate_fn(self, batch):
        self.batch_count += 1

        # Drop invalid images
        batch = [data for data in batch if data is not None]

        paths, imgs, msks = list(zip(*batch))

        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(
                range(self.min_size, self.max_size + 1, 32))

        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        msks = torch.stack([resize(msk, self.img_size) for msk in msks])

        return paths, imgs, msks

    def __len__(self):
        return len(self.img_files)

class ImageFolder(Dataset):
    def __init__(self, folder_path, img_size=416, transform=None):
        self.files = sorted(glob.glob("%s/*.*" % folder_path))
        self.transform = transform
        self.img_size = img_size

    def __getitem__(self, index):

        img_path = self.files[index % len(self.files)]
        img = np.array(
            Image.open(img_path).convert('RGB'),
            dtype=np.uint8)

        # Apply transforms
        if self.transform:
            img = self.transform(img)

        return img_path, img
    
    def collate_fn(self, batch):

        # Drop invalid images
        batch = [data for data in batch if data is not None]

        paths, imgs = list(zip(*batch))

        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        return paths, imgs

    def __len__(self):
        return len(self.files)
