import os
import datetime
import torch
from torch.utils.data import DataLoader
from utils.dataloader import ListDataset
from utils.ttansf import DEFAULT_TRANSFORMS
from tqdm import tqdm
from torch import optim, nn
from dataset_gabor import MyDataset
from FPN_raw import FPN
from torchvision import transforms
from einops import rearrange
import torch.nn.functional as F
from metrics import hausdorff95,Assd,asd,hausdorff,voe,rvd,dice,msd,recall
from models.models_muls import VesselSegNet

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(41)


def mkdirs(path):
    if not os.path.exists(path): os.makedirs(path)


def load_checkpoint_model(model, ckpt_best, device):
    state_dict = torch.load(ckpt_best, map_location=device)
    #model.load_state_dict(state_dict['state_dict'])
    model.load_state_dict(state_dict)
    return model


@torch.no_grad()
def test_epoch(model, dl):
    model.eval()
    v_Assd, v_asd, v_hausdorff, v_voe, v_rvd, v_dice, v_msd, ii = 0, 0, 0,0, 0, 0, 0, 0
    for pt, x2, mask in dl:
        stage_list, output =  model(x2.to(device))
        l = len(stage_list)
        outputs = stage_list[l-1]

        mask = mask.to(device)    
        outputs = outputs.detach().cpu().numpy()
        mask = mask.detach().cpu().numpy()
        outputs = (outputs>0).astype(int)
        mask = (mask>0).astype(int)

        v_dice += dice(outputs, mask)
        v_voe += voe(outputs, mask)
        v_rvd += rvd(outputs, mask)
        v_msd += msd(outputs, mask)
        v_hausdorff += hausdorff95(outputs, mask)
        v_asd += asd(outputs, mask)
        v_Assd += Assd(outputs, mask)

        ii += 1
    return v_dice / ii, v_voe / ii, v_rvd / ii, v_msd / ii, v_hausdorff / ii, v_asd / ii, v_Assd / ii

def validation_data_loader(img_path, batch_size, img_size, n_cpu):

    dataset = ListDataset(
        img_path,
        trainf='val',
        img_size=img_size,
        multiscale=False,
        transform=DEFAULT_TRANSFORMS)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn)
    return dataloader

def test():
    model = VesselSegNet()
    #pth = 'ckpt/20240102222322m10_setr/ours_dataset_best.pth'
    pth = 'ckpt/20240105211512m10_setr/ours_dataset_best.pth'
    
    model = load_checkpoint_model(model, pth, device)
    model = model.to(device)
    model = nn.DataParallel(model)
    
    #root_dir = '../dataset/3Dircadb_2d_dataset'
    #root_dir = '../dataset/MSD_2d_dataset'
    #root_dir = '../dataset/gz_2d_vessel_final'
    
    #root_dir = './dataset/3Dircadb_2d_256'
    root_dir = './dataset/MSD_2d_256'
    train_image_root = 'train'
    val_image_root = 'val'

    #root_dir = 'LVSD300'
    #train_image_root = 'train'
    #val_image_root = 'vali'

    #test_dataset = MyDataset(model_type=val_image_root, data_filename=root_dir,sub_name='')
    #test_dl = DataLoader(dataset=test_dataset, batch_size=8, num_workers=8, shuffle=False)
    test_dl = validation_data_loader(
        root_dir+'/'+val_image_root,
        8,
        256,
        8
    )

    fout_log = open('ours_crossdata.txt', 'w')
    #print(len(test_dataset))
    v_dice, v_voe, v_rvd, v_msd, v_hausdorff, v_asd, v_Assd = test_epoch(model, test_dl)
    log = "v_dice:%0.5f\n v_voe:%0.5f\n v_rvd:%0.5f\n v_msd:%0.5f\n v_hausdorff:%0.5f\n v_asd:%0.5f\n v_Assd:%.5f" % (
            v_dice, v_voe, v_rvd, v_msd, v_hausdorff, v_asd, v_Assd)
    print(log)
    fout_log.write(log)
    #fout_log.flush()
    fout_log.close()

    
if __name__ == '__main__':
    
    test()
