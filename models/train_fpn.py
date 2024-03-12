import os
import datetime
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch import optim, nn
from dataset_sv import MyDataset
from models_ca import VesselNet
from FPN_raw import FPN
from torchvision import transforms
from einops import rearrange
import torch.nn.functional as F


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(41)


def mkdirs(path):
    if not os.path.exists(path): os.makedirs(path)


def dice_metric(output, target):
    output = (output > 0).float()
    dice = ((output * target).sum() * 2+0.1) / (output.sum() + target.sum() + 0.1)
    return dice

def load_checkpoint_model(model, ckpt_best, device):
    state_dict = torch.load(ckpt_best, map_location=device)
    #model.load_state_dict(state_dict['state_dict'])
    model.load_state_dict(state_dict)
    return model

def voe_metric(output, target):
    output = (output > 0).float()
    voe = ((output.sum() + target.sum()-(target*output).sum()*2)+0.1) / (output.sum() + target.sum()-(target*output).sum() + 0.1)
    return voe.item()

def rvd_metric(output, target):
    output = (output > 0).float()
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
    p = (target*output).sum()
    sen = (p+0.1) / (output.sum()+0.1)
    return sen

def spe_m(output,target):
    output = (output>0).float()
    target, output = target.view(-1), output.view(-1)
    tn = target.shape[0] - (target.sum() + output.sum() - (target*output).sum())
    spe = (tn+0.1) / (target.shape[0] - output.sum()+0.1)
    return spe


def active_contour_loss(y_true, y_pred, weight=100):
    '''
    y_true, y_pred: tensor of shape (B, C, H, W), where y_true[:,:,region_in_contour] == 1, y_true[:,:,region_out_contour] == 0.
    weight: scalar, length term weight.
    '''
    #print(y_true.shape)
    #print(y_pred.shape)
    # length term
    y_pred = torch.sigmoid(y_pred)
    delta_r = y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :]  # horizontal gradient (B, C, H-1, W)
    delta_c = y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1]  # vertical gradient   (B, C, H,   W-1)

    delta_r = delta_r[:, :, 1:, :-2] ** 2  # (B, C, H-2, W-2)
    delta_c = delta_c[:, :, :-2, 1:] ** 2  # (B, C, H-2, W-2)

    delta_pred = torch.abs(delta_r + delta_c)
    # print(delta_pred)
    epsilon = 1e-8  # where is a parameter to avoid square root is zero in practice.
    lenth = torch.mean(torch.sqrt(delta_pred + epsilon))  # eq.(11) in the paper, mean is used instead of sum.
    # print(lenth)
    # region term
    c_in = torch.ones_like(y_pred)
    # print((y_true - c_in) ** 2)
    c_out = torch.zeros_like(y_pred)
    # print((y_true - c_out) ** 2)

    region_in = torch.mean(y_pred * (y_true - c_in) ** 2)  # equ.(12) in the paper, mean is used instead of sum.
    region_out = torch.mean((1 - y_pred) * (y_true - c_out) ** 2)
    region = region_in + region_out

    #region = 0.5*region_in + 0.5*region_out
    #print("y_true_max=",torch.max(y_true))
    #print("y_pred_max=",torch.max(y_pred))
    loss = weight * lenth + region
    #loss = 0.9 * lenth + 0.1 * region
    return loss

def cycle_loss(input, target):
    #criteria1 = nn.BCELoss()
    criteria2 = nn.MSELoss()
    
    input = torch.sigmoid(input)
    #loss1 = criteria1(input, target)

    #环形边界c
    c_m_1 = torch.zeros_like(target)
    c_m_2 = torch.zeros_like(target)
    m_1 = target[:, :, 1:, :] - target[:, :, :-1, :]
    m_2 = target[:, :, :, 1:] - target[:, :, :, :-1]
    c_m_1[:, :, 1:, :] = m_1
    c_m_2[:, :, :, 1:] = m_2
    c = torch.abs(c_m_1 + c_m_2)
    c[c > 0] = 1
    #print('c=',c.shape)
    #高斯核
    # kernel = gaussian_kernel_2d_opencv()
    kernel = torch.Tensor([[[[0.0625,0.1250,0.0625],
                           [0.1250,0.2500,0.1250],
                           [0.0625,0.1250,0.0625]]]]).to(device)
    
    channels = 1
    gc = F.conv2d(c, weight=kernel, padding=1,groups=channels)
    bais = (gc>0).float()*(1-torch.max(gc))
    gc = gc + bais

    #取出预测值的边界
    input = input.masked_select(gc > 0)
    gc = gc.masked_select(gc > 0)
    #print(input.shape)
    #print(gc.shape)
    loss2 = criteria2(input, gc)

    return loss2

def gabor_bce_with_logits(input, target, weight=None, epoch=1):
    max_val = (-input).clamp(min=0)
    loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()
    #print('==',loss)

    alph = epoch // 5
    bata = 0.5 - 0.1*alph
    if bata<0:
        bata=0.01

    weight = weight.clamp(min=bata, max=1.0)

    if weight is not None:
        loss = loss * weight
    #print('===', loss)
    return loss.mean()


def train_epoch(epoch, model, dl, optimizer, criterion, criterion2):
    model.train()
    bar = tqdm(dl)
    bar.set_description_str("%02d" % epoch)
    loss_v, dice_v, loss_pre, loss_sur, ii = 0, 0, 0, 0, 0
    for x2, mask in bar:
    #for x2, gb, mask in bar:
        #x2 = rearrange(x2.float(),'b h w c -> b c h w')
        outputs = model(x2.to(device))
        mask = mask.to(device)
        
        #loss = gabor_bce_with_logits(outputs, mask, gb, epoch)
        loss = criterion(outputs, mask)
        #print("loss_res="+str(loss1)+",loss_cont="+str(loss3)+",loss_sur="+str(loss2))
        
        #loss = 0.4*loss1+0.6*loss2
        #loss = 0.3*loss3+0.4*loss2+0.3*loss1
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        dice = dice_metric(outputs, mask)
        dice_v += dice
        loss_v += loss.item()
        loss_pre += 0#loss3.item()
        loss_sur += 0#loss2.item()
        ii += 1
        bar.set_postfix(loss=loss.item(), dice=dice.item())
    return loss_v / ii, dice_v / ii, loss_pre / ii, loss_sur / ii


@torch.no_grad()
def val_epoch(model, dl, criterion):
    model.eval()
    loss_v, dice_v, voe_v, rvd_v,acc_v, sen_v, spe_v, ii = 0, 0, 0,0, 0, 0, 0, 0
    for x2, mask in dl:
    #for x2, gb, mask in dl:
        #x2 = torch.unsqueeze(x2, 1).float()
        #x2 = rearrange(x2.float(),'b h w c -> b c h w')
        #x2 = rearrange(x2, 'b 1 (n h) (m w) c -> (b n m) c h w', c=3,n=8,m=8,h=64,w=64)
        #sur, outputs = model(x2.to(device))
        outputs = model(x2.to(device))
        mask = mask.to(device)
        #mask.unsqueeze_(1)
        #mask = rearrange(mask, 'b 1 (n h) (m w) -> (b n m) 1 h w', n=8,m=8,h=64,w=64)
        #outputs.squeeze_(1)
        loss_v += criterion(outputs, mask).item()
        dice_v += dice_metric(outputs, mask)
        voe_v += voe_metric(outputs, mask)
        rvd_v += rvd_metric(outputs, mask)
        acc_v += acc_m(outputs, mask)
        sen_v += sen_m(outputs, mask)
        spe_v += spe_m(outputs, mask)

        ii += 1
    return loss_v / ii, dice_v / ii, voe_v / ii, rvd_v / ii, acc_v / ii, sen_v / ii, spe_v / ii


def train(opt):
    #model = VesselNet()
    model = FPN([3,4,23,3], 1, back_bone="resnet101")
    #model = UNet(n_channel=1, n_class=1)
    #if opt.w:
    #    model.load_state_dict(torch.load(opt.w))
    #pth = 'ckpt/20220425115307_setr/msd_dice_best.pth'
    #model = load_checkpoint_model(model, pth, device)
    model = model.to(device)
    #model = nn.DataParallel(model)
    
    #root_dir = 'dataset/MSD_2d_dataset'
    root_dir = 'dataset/3Dircadb_2d_dataset'
    #root_dir = '../dataset/gz_2d_vessel_final'
    train_image_root = 'train'
    val_image_root = 'val'

    #root_dir = 'LVSD300'
    #train_image_root = 'train'
    #val_image_root = 'vali'

    train_dataset = MyDataset(model_type=train_image_root, data_filename=root_dir,sub_name='')
    train_dl = DataLoader(dataset=train_dataset, batch_size=opt.batch_size, num_workers=8, shuffle=True)
    val_dataset = MyDataset(model_type=val_image_root, data_filename=root_dir,sub_name='')
    val_dl = DataLoader(dataset=val_dataset, batch_size=opt.batch_size, num_workers=8, shuffle=False)

    optimizer = optim.Adam(params=model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    criterion2 = nn.MSELoss()
    # 一些文件日志信息
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, min_lr=1e-6,patience=10)
    best_dice_epoch, best_dice, b_voe, b_rvd, train_loss, train_dice, b_acc, b_sen, b_spe,pre_loss, sur_loss =  0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0
    save_dir = os.path.join(opt.ckpt, datetime.datetime.now().strftime('%Y%m%d%H%M%S')) + "_" + opt.name
    mkdirs(save_dir)
    w_dice_best = os.path.join(save_dir, 'dircad_dice_best.pth')
    #w_dice_best = os.path.join(save_dir, 'msd_dice_best.pth')
    #w_dice_best = os.path.join(save_dir, 'ours_dataset_cur.pth')
    #cur = 0
    #fout_log = open(os.path.join(save_dir, 'msd_log.txt'), 'w')
    #fout_log = open(os.path.join(save_dir, 'dircad_log.txt'), 'w')
    fout_log = open(os.path.join(save_dir, 'ours_log.txt'), 'w')
    print(len(train_dataset), len(val_dataset), save_dir)
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
    parser.add_argument('--input_size', type=int, default=512, help='input size')
    parser.add_argument('--max_epoch', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--ckpt', type=str, default='ckpt', help='the dir path to save model weight')
    parser.add_argument('--w', type=str, help='the path of model wight to test or reload')
    parser.add_argument('--suf', type=str, choices=['.dcm', '.JL', '.png'], help='suffix', default='.png')
    parser.add_argument('--eval', action="store_true", help='eval only need weight')
    parser.add_argument('--test_root', type=str, help='root_dir')

    opt = parser.parse_args()
    train(opt)
