import os
import datetime
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch import optim, nn
from utils.dataloader import ListDataset
from torchvision import transforms
from utils.ttansf import DEFAULT_TRANSFORMS
import torch.nn.functional as F
from utils.dc_utils import worker_seed_set
from models.models_multis import VesselSegNet
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(41)

def train_data_loader(img_path, batch_size, img_size, n_cpu, multiscale_training=False):

    dataset = ListDataset(
        img_path,
        trainf='train',
        img_size=img_size,
        multiscale=multiscale_training,
        transform=DEFAULT_TRANSFORMS)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
        worker_init_fn=worker_seed_set)
    return dataloader

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

def mkdirs(path):
    if not os.path.exists(path): os.makedirs(path)


def dice_metric(output, target):
    output = output > 0
    dice = ((output * target).sum() * 2+0.1) / (output.sum() + target.sum() + 0.1)
    return dice

def load_checkpoint_model(model, ckpt_best, device):
    state_dict = torch.load(ckpt_best, map_location=device)
    #model.load_state_dict(state_dict['state_dict'])
    model.load_state_dict(state_dict)
    return model

def voe_metric(output, target):
    output = output > 0
    voe = ((output.sum() + target.sum()-(target*output).sum().float()*2)+0.1) / (output.sum() + target.sum()-(target*output).sum().float() + 0.1)
    return voe.item()

def rvd_metric(output, target):
    output = output > 0
    rvd = ((output.sum() / (target.sum() + 0.1) - 1) * 100)
    return rvd.item()

def acc_m(output,target):
    output = (output>0).float()
    target, output = target.view(-1), output.view(-1)
    acc = (target==output).sum().float() / target.shape[0]
    return acc

def sen_m(output,target):
    output = (output>0).float()
    target, output = target.view(-1), output.view(-1)
    p = (target*output).sum().float()
    sen = (p+0.1) / (output.sum()+0.1)
    return sen

def spe_m(output,target):
    output = (output>0).float()
    target, output = target.view(-1), output.view(-1)
    tn = target.shape[0] - (target.sum() + output.sum() - (target*output).sum().float())
    spe = (tn+0.1) / (target.shape[0] - output.sum()+0.1)
    return spe

class BinaryDiceLoss(nn.Module):
    def __init__(self):
        super(BinaryDiceLoss, self).__init__()

    def forward(self, input, targets):
        N = targets.size()[0]
        smooth = 1

        input_flat = input.view(N, -1)
        targets_flat = targets.view(N, -1)
        intersection = input_flat * targets_flat
        N_dice_eff = (2 * intersection.sum(1) + smooth) / (input_flat.sum(1) + targets_flat.sum(1) + smooth)

        loss = 1 - N_dice_eff.sum() / N
        return loss

def get_gaussian_kernels(kernel_size = 3, sigma = 2):
    if sigma == 0:
        print('Error!The value of sigma cannot be zero.')
        return
    else:
        sigma_3 = 3 * sigma
        X = np.linspace(-sigma_3, sigma_3, kernel_size)
        Y = np.linspace(-sigma_3, sigma_3, kernel_size)
        x, y = np.meshgrid(X, Y)
        gauss_1 = 1 / (2 * np.pi * sigma ** 2) * np.exp(- (x ** 2 + y ** 2) / (2 * sigma ** 2))
        Z = gauss_1.sum()  
        gauss_2 = (1 / Z) * gauss_1
    return gauss_2

# print(get_gaussian_kernels(7,2))

class StageLoss(nn.Module):
    def __init__(self, device=None, is_grad=False):
        super(StageLoss, self).__init__()
        self.device = device
        self.is_grad = is_grad
        self.stage_loss = nn.BCEWithLogitsLoss()
        self.stage_loss2 = nn.MSELoss()
        self.pl = nn.MaxPool2d(2)

    def build_target(self, targets, stage_num):
        karnel_size = 5 - 2*stage_num
        _, c, _, _ = targets.shape
        karnel = get_gaussian_kernels(karnel_size, 2)

        kernel = torch.FloatTensor(karnel).unsqueeze(0).unsqueeze(0)
        kernel = np.repeat(kernel, c, axis=0)
        weight = nn.Parameter(data=kernel, requires_grad=self.is_grad).to(self.device)
        x = F.conv2d(targets, weight, padding=karnel_size//2)
        return x

    def forward(self, output, targets=None):
        #targets = self.pl(targets)
        stloss = torch.zeros(1, device=self.device)
        for item in range(len(output)):
            new_tgs = self.build_target(targets, item).to(self.device)
            stloss += self.stage_loss(torch.sigmoid(output[item]), new_tgs)
        return stloss

def structure_loss2(pred, mask):
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    return (wbce + wiou).mean()

def structure_loss(pred, mask):
    weit = 1 + 3*torch.abs(F.avg_pool2d(mask, kernel_size=7, stride=1, padding=3) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    return (wbce + wiou).mean()

def train_epoch(epoch, model, dl, optimizer, criterion, criterion2):
    model.train()
    bar = tqdm(dl)
    bar.set_description_str("%02d" % epoch)
    loss_v, dice_v, loss_pre, loss_sur, ii = 0, 0, 0, 0, 0
    #for x2, mask in bar:
    for pt, x2, mask in bar:

        stage_list, output =  model(x2.to(device))
        mask = mask.to(device)


        l = len(stage_list)
        loss = torch.zeros(1, device=device)
        for i in range(l):
            #loss += structure_loss(stage_list[i], mask).to(device)
            loss +=criterion2(stage_list[i], mask).to(device)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        dice = dice_metric(stage_list[l-1], mask)
        dice_v += dice
        loss_v += loss.item()
        loss_pre += loss3.item()
        loss_sur += loss2.item()
        ii += 1
        bar.set_postfix(loss=loss.item(), dice=dice.item())
    return loss_v / ii, dice_v / ii, loss_pre / ii, loss_sur / ii


@torch.no_grad()
def val_epoch(model, dl, criterion):
    model.eval()
    loss_v, dice_v, voe_v, rvd_v,acc_v, sen_v, spe_v, ii = 0, 0, 0,0, 0, 0, 0, 0
    for pt, x2, mask in dl:
        stage_list, output =  model(x2.to(device))
        mask = mask.to(device)
        l = len(stage_list)
        loss_v += structure_loss(stage_list[l-1], mask).item()
        dice_v += dice_metric(stage_list[l-1], mask)
        voe_v += voe_metric(stage_list[l-1], mask)
        rvd_v += rvd_metric(stage_list[l-1], mask)
        acc_v += acc_m(stage_list[l-1], mask)
        sen_v += sen_m(stage_list[l-1], mask)
        spe_v += spe_m(stage_list[l-1], mask)
        ii += 1
    return loss_v / ii, dice_v / ii, voe_v / ii, rvd_v / ii, acc_v / ii, sen_v / ii, spe_v / ii


def train(opt):

    model = VesselSegNetNet()
    #if opt.w:
    #    model.load_state_dict(torch.load(opt.w))

    model = model.to(device)
    model = nn.DataParallel(model)

    #train_root_dir = './dataset/3Dircadb_2d_256/train'
    #valid_root_dir = './dataset/3Dircadb_2d_256/val'
    train_root_dir = './dataset/MSD_2d_256/train'
    valid_root_dir = './dataset/MSD_2d_256/val'

    train_dl = train_data_loader(
        train_root_dir,
        opt.batch_size,
        opt.input_size,
        opt.n_cpu,
        opt.multiscale_training)

    val_dl = validation_data_loader(
        valid_root_dir,
        opt.batch_size,
        opt.input_size,
        opt.n_cpu
    )

    optimizer = optim.Adam(params=model.parameters(), lr=1e-3, weight_decay=0.001)
    criterion = StageLoss(device=device)#nn.BCEWithLogitsLoss()
    criterion2 = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([2.0], device=device))#nn.MSELoss()

    # 一些文件日志信息
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, min_lr=1e-6,patience=10)
    best_dice_epoch, best_dice, b_voe, b_rvd, train_loss, train_dice, b_acc, b_sen, b_spe,pre_loss, sur_loss =  0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0
    save_dir = os.path.join(opt.ckpt, datetime.datetime.now().strftime('%Y%m%d%H%M%S')) + "m10_" + opt.name
    mkdirs(save_dir)

    w_dice_best = os.path.join(save_dir, 'ours_dataset_best.pth')
    

    fout_log = open(os.path.join(save_dir, 'ous_log.txt'), 'w')
    print(len(train_dl), len(val_dl), save_dir)
    for epoch in range(opt.max_epoch):
        if not opt.eval:
            train_loss, train_dice, pre_loss,sur_loss = train_epoch(epoch, model, train_dl, optimizer, criterion, criterion2)
        val_loss, val_dice, voe_v, rvd_v, acc_v, sen_v, spe_v = val_epoch(model, val_dl, criterion)
        if best_dice < val_dice:
            best_dice, best_dice_epoch, b_voe, b_rvd,b_acc, b_sen, b_spe = val_dice, epoch, voe_v, rvd_v, acc_v, sen_v, spe_v
            torch.save(model.module.state_dict() if hasattr(model, 'module') else model.state_dict(), w_dice_best)
        
        lr = optimizer.param_groups[0]['lr']
        log = "%02d train_loss:%0.3e, train_dice:%0.5f,pre_loss:%0.3e,sur_loss:%0.3e, val_loss:%0.3e, val_dice:%0.5f, lr:%.3e\n best_dice:%.5f, voe:%.5f, rvd:%.5f, acc:%.5f, sen:%.5f, spe:%.5f(%02d)\n" % (
            epoch, train_loss, train_dice, pre_loss, sur_loss, val_loss, val_dice, lr, best_dice, b_voe, b_rvd, b_acc, b_sen, b_spe, best_dice_epoch)
        print(log)
        fout_log.write(log)
        fout_log.flush()
        scheduler.step(val_loss)
        #cur = cur + 1
    fout_log.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='setr', help='study name')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('--input_size', type=int, default=256, help='input size')
    parser.add_argument("--multiscale_training", action="store_true", help="Allow multi-scale training")
    parser.add_argument('--n_cpu', type=int, default=4, help='cpu works')
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--ckpt', type=str, default='ckpt', help='the dir path to save model weight')
    parser.add_argument('--w', default='ckpt/20221116161214_setr/ours_dataset_best.pth', type=str, help='the path of model wight to test or reload')
    parser.add_argument('--suf', type=str, choices=['.dcm', '.JL', '.png'], help='suffix', default='.png')
    parser.add_argument('--eval', action="store_true", help='eval only need weight')
    parser.add_argument('--test_root', type=str, help='root_dir')

    opt = parser.parse_args()
    opt.w = False
    train(opt)
