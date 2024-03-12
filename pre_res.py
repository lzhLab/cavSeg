import os
import datetime
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch import optim, nn
from dataset_gabor import MyDataset
import glob
from FPN_raw import FPN
from torchvision import transforms
from einops import rearrange
import torch.nn.functional as F
import numpy as np
import nibabel as nib
from PIL import Image
import cv2

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(41)


def load_checkpoint_model(model, ckpt_best, device):
    state_dict = torch.load(ckpt_best, map_location=device)
    #model.load_state_dict(state_dict['state_dict'])
    model.load_state_dict(state_dict)
    return model

def default_loader2(path):
    train_3D = nib.load(path)
    label = train_3D.dataobj[:, :, 0]/255.
    #data = train_3D.dataobj[m:64 + m, n:64 + n, 1:8]/255.
    #data = train_3D.dataobj[m:64 + m, n:64 + n, 3:6]/255.
    data = train_3D.dataobj[:,:, 3:6]/255.
    return data,label


def default_loader1(path):
    image = cv2.imread(path)/255.
    return image

def default_loader(path):
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)/255.
    gabor = cv2.imdecode(np.fromfile(path.replace('val','gabor_train'), dtype=np.uint8), -1)
    gabor_w = gabor/(np.max(gabor)+1)
    label = cv2.imdecode(np.fromfile(path.replace('val','trainmask'), dtype=np.uint8), -1)/255.
    return image,gabor_w,label

def val_epoch(model, pth):
    model.eval()
    names = pth[pth.rindex('/')+1:]
    x2,gb,lb = default_loader(pth)
    x2 = torch.Tensor(x2).float()
    x2 = rearrange(x2,'h w -> 1 1 h w')
    
    outputs = model(x2.to(device))
    
    out = rearrange(outputs, '1 1 h w -> h w', h=512,w=512)
    out = torch.sigmoid(out)
    pred = out.cuda().data.cpu().numpy()
    #pred = pred.transpose()
    pred = pred > 0.5
    pred = pred * 255
    Image.fromarray(pred.astype('uint8')).convert('L').save('pred_fpn/'+names)


def train():
    model = FPN([3,4,23,3],1,back_bone="resnet101")
    
    pth = 'ckpt/20220523090912_setr/ours_dataset_cur.pth'
    model = load_checkpoint_model(model, pth, device)
    
    model = model.to(device)
    model = nn.DataParallel(model)
    
    imgpth = glob.glob('../dataset/gz_2d_vessel_final/val/1.2*')
    for item in imgpth:
        val_epoch(model, item)
    
    
if __name__ == '__main__':
    
    train()
