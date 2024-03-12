import torch
#from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
import numpy as np
#import nibabel as nib
import os
import glob
import cv2
def LoaderNames(names_path):
    f = open(names_path, 'r')
    names = f.read().splitlines()
    f.close()
    return names


def default_loader(path, model_type):
    #image = cv2.imread(path,0)/255.
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)/255.
    #gabor = cv2.imdecode(np.fromfile(path.replace(str(model_type),'gabor_train'), dtype=np.uint8), -1)
    #gabor_w = gabor/(np.max(gabor)+1)
    label = cv2.imdecode(np.fromfile(path.replace(str(model_type),'trainmask'), dtype=np.uint8), -1)/255.
    return image,label


class MyDataset(Dataset):  
    def __init__(self, model_type, data_filename,sub_name='', transform=None, loader=default_loader): 
        super(MyDataset, self).__init__() 
        imgs = glob.glob(os.path.join(data_filename,model_type+'/'+sub_name+'*'))
        self.imgs = imgs
        self.transform = transform
        self.loader = loader
        self.data_filename = data_filename
        self.typestr = model_type
    def __getitem__(self, index):  
        img_str = self.imgs[index] 
        img,label = self.loader(img_str,self.typestr)
        
        img = torch.FloatTensor(img).unsqueeze(0)
        #gb = torch.FloatTensor(gb).unsqueeze(0)
        label = torch.FloatTensor(label).unsqueeze(0)

        #if self.transform is not None:
        #    img = self.transform(img)  # 数据标签转换为Tensor
        #    label = self.transform(label)  # 数据标签转换为Tensor
        #    gb = self.transform(gb)
        return img, label 

    def __len__(self): 
        return len(self.imgs)

