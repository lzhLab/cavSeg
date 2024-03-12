import torch.nn as nn
import torch
import torch.nn.functional as F
from functools import partial
from backbone.resnet3 import ResNet50, ResNet101
import numpy as np

nonlinearity = partial(F.relu, inplace=True)
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, ratio=16, kernel_size=7):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(c1, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])

    def forward(self, x):
        out = self.channel_attention(x) * x
        # print('outchannels:{}'.format(out.shape))
        out = self.spatial_attention(out) * out
        return out

class DACblock(nn.Module):
    def __init__(self, channel):
        super(DACblock, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=3, padding=3)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=5, padding=5)
        self.conv1x1 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0)
        self.conv_out = nn.Conv2d(channel*5, channel, kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(channel)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.conv1x1(self.dilate2(x)))
        dilate3_out = nonlinearity(self.conv1x1(self.dilate2(self.dilate1(x))))
        dilate4_out = nonlinearity(self.conv1x1(self.dilate3(self.dilate2(self.dilate1(x)))))
        out = torch.cat([x, dilate1_out, dilate2_out, dilate3_out, dilate4_out], dim=1)

        #out = self.bn(self.conv_out(out))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out

        return out

class Mish(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x * (torch.tanh(torch.nn.functional.softplus(x)))
        return x

# Res_unit
class Res_unit(nn.Module):
    def __init__(self, out_channels, nblocks=3, shortcut=True):
        super(Res_unit, self).__init__()
        self.CBM1 = Conv_Bn_Activation(out_channels,out_channels,kernel_size=1,stride=1, activation='relu')
        self.CBM2 = Conv_Bn_Activation(out_channels,out_channels,kernel_size=3,stride=1, activation='relu')
        self.shortcut = shortcut
        self.module_list = nn.ModuleList()
        for i in range(nblocks):
            resblock_one = nn.ModuleList()
            resblock_one.append(self.CBM1)
            resblock_one.append(self.CBM2)
            self.module_list.append(resblock_one)

    def forward(self, x):
        for module in self.module_list:
            h = x
            for res in module:
                h = res(h)
            x = x + h if self.shortcut else h
        return x


class AttBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(AttBlock, self).__init__()
        self.conv1x1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.spatial_attention1 = SpatialAttention(5)
        self.conv3x3 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.spatial_attention2 = SpatialAttention(5)
        self.conv5x5 = nn.Conv2d(planes, planes, kernel_size=5, padding=2, stride=stride, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.spatial_attention3 = SpatialAttention(5)
        self.conv7x7 = nn.Conv2d(planes, planes, kernel_size=7, padding=3, stride=stride, bias=False)
        self.bn4 = nn.BatchNorm2d(planes)
        self.spatial_attention4 = SpatialAttention(5)
        self.conv = nn.Conv2d(planes, planes * self.expansion, padding=1, kernel_size=3, stride=stride, bias=False)
        self.bn = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        # print(x.shape)
        out1 = self.bn1(self.conv1x1(x))
        out1 = self.spatial_attention1(out1) * out1

        out2 = self.bn2(self.conv3x3(x))
        out2 = self.spatial_attention2(out2) * out2

        out3 = self.bn3(self.conv5x5(x))
        out3 = self.spatial_attention3(out3) * out3

        out4 = self.bn4(self.conv7x7(x))
        out4 = self.spatial_attention4(out4) * out4

        out = identity + out1 + out2 + out3 + out4
        # out = self.conv(out)
        out = self.relu(out)

        return out

class Upsample(nn.Module):
    def __init__(self):
        super(Upsample, self).__init__()

    def forward(self, x, target_size, inference=False):
        assert (x.data.dim() == 4)

        if inference:
            return x.view(x.size(0), x.size(1), x.size(2), 1, x.size(3), 1).\
                    expand(x.size(0), x.size(1), x.size(2), target_size[2] // x.size(2), x.size(3), target_size[3] // x.size(3)).\
                    contiguous().view(x.size(0), x.size(1), target_size[2], target_size[3])
        else:
            return F.interpolate(x, size=(target_size[2], target_size[3]), mode='nearest')

# CSP
class CSP(nn.Module):
    def __init__(self, in_channels, out_channels, nblocks):
        super(CSP, self).__init__()
        self.CBM1 = Conv_Bn_Activation(in_channels, out_channels, kernel_size=3, stride=1, activation='relu')
        self.CBM2 = Conv_Bn_Activation(in_channels, out_channels, kernel_size=3, stride=1, activation='relu')
        self.CBM3 = Conv_Bn_Activation(out_channels, out_channels, kernel_size=3, stride=1, activation='relu')
        self.Res_unit = Res_unit(out_channels, nblocks=nblocks)

    def forward(self, x):
        x1 = self.CBM1(x)
        x1 = self.Res_unit(x1)
        x1 = self.CBM3(x1)
        x2 = self.CBM2(x)
        x = torch.cat([x1,x2], dim=1)
        return x

class Conv_Bn_Activation(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, activation, bn=True, bias=False):
        super().__init__()
        pad = (kernel_size - 1) // 2

        self.conv = nn.ModuleList()
        if bias:
            self.conv.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad))
        else:
            self.conv.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad, bias=False))
        if bn:
            self.conv.append(nn.BatchNorm2d(out_channels))
        if activation == "mish":
            self.conv.append(Mish())
        elif activation == "relu":
            self.conv.append(nn.ReLU(inplace=True))
        elif activation == "leaky":
            self.conv.append(nn.LeakyReLU(0.1, inplace=True))
        elif activation == "linear":
            pass
        else:
            print("activate error !!! ")

    def forward(self, x):
        for l in self.conv:
            x = l(x)
        return x

def build_backbone(back_bone, pretrained=False):

    if back_bone == "resnet50":
        return ResNet50(pretrained=pretrained)
    if back_bone == "resnet101":
        return ResNet101(pretrained=pretrained)

class for_denoise(nn.Module):
    def __init__(self, in_channels=128):
        super().__init__()

        self.conv_1 = Conv_Bn_Activation(in_channels, 2*in_channels, 3, 1, 'relu')
        self.conv_2 = Conv_Bn_Activation(2*in_channels, 4*in_channels, 3, 1, 'relu')
        self.conv_3 = Conv_Bn_Activation(4*in_channels, 2*in_channels, 3, 1, 'relu')
        self.conv_4 = Conv_Bn_Activation(2*in_channels, in_channels, 3, 1, 'relu')
    def forward(self, x):
        x = self.conv_1(x)
        x1 = F.interpolate(x, scale_factor=0.5, mode='bilinear')
        x1 = self.conv_2(x1)
        x1 = F.interpolate(x1, scale_factor=2, mode='bilinear')
        x1 = self.conv_3(x1)
        x = x + x1
        x = self.conv_4(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear')
        return x

class vessel_single_stage(nn.Module):
    def __init__(self, in_channel=128, out_num=2, nblocks=2):
        super(vessel_single_stage, self).__init__()
        self.conv_s = Conv_Bn_Activation(in_channel, in_channel, 3, 1, 'relu')
        #self.conv_down = Conv_Bn_Activation(in_channel, out_num*in_channel, 3, 2, 'relu')
        # self.attblock = AttBlock(in_channel, in_channel)

        self.attblock = CSP(in_channel, in_channel, nblocks)
        self.conv_m = Conv_Bn_Activation(2*in_channel, in_channel, 3, 1, 'relu')
        self.dblock = DACblock(in_channel)
        self.att = CBAM(2*in_channel)

        self.dn = for_denoise(in_channel)
        self.conv_e = Conv_Bn_Activation(in_channel, in_channel, 3, 1, 'relu')

        self.conv_out = nn.Conv2d(in_channel, 1, kernel_size=3, stride=1, padding=1)
    def forward(self, x, att):
        x0 = self.conv_s(x)
        if att is not None:
            x = x0 * torch.sigmoid(F.interpolate(att, scale_factor=0.5, mode='bilinear'))
        x = self.dblock(x)
        x = self.attblock(x)
        x = self.att(x)
        x = self.conv_m(x)
        x2 = F.interpolate(x0, scale_factor=0.5, mode='bilinear')#self.conv_down(x0)
        x2 = self.dn(x2)
        x = x + x2
        x = self.conv_e(x)
        out = self.conv_out(x)
        out = F.interpolate(out, scale_factor=2, mode='bilinear')
        
        if att is not None:
            out = out + att
        return x+x0, out


class VesselSegNet(nn.Module):
    def __init__(self, stage_num=3, num_classes=1, stage_in_channel=128,
                 pretrained=False, **kwargs):
        super(VesselSegNet, self).__init__()
        self.num_classes = num_classes
        self.stage_num = stage_num
        self.stage_in_channel = stage_in_channel
        self.conv1 = Conv_Bn_Activation(3, 64, 3, 1, 'relu')
        self.conv2 = Conv_Bn_Activation(64, 128, 5, 2, 'relu')

        self.conv3 = Conv_Bn_Activation(128, 64, 3, 1, 'relu')
        self.out = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)

        self.stages_modules = list()
        for i in range(self.stage_num):
            self.stages_modules.append(
                vessel_single_stage(
                    self.stage_in_channel
                )
            )
            setattr(self, 'stage%d' % i, self.stages_modules[i])

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)  # b, 128, 128, 128
        
        outputs = list()
        for i in range(self.stage_num):
            if i<1:
                x, res = eval('self.stage' + str(i))(x, None)
            else:
                x, res = eval('self.stage' + str(i))(x, outputs[i-1])
            outputs.append(res)
            
        x = F.interpolate(x, scale_factor=2, mode='bilinear')
        x = self.conv3(x)
        out = self.out(x)
        return outputs, out

