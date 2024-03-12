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
        #self.at_c = ChannelAttention(channel*5)
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
        #out = self.at_c(out)*out
        #out = self.bn(self.conv_out(out))
        out = self.conv_out(out)
        #out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out

        return out

class Mish(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x * (torch.tanh(torch.nn.functional.softplus(x)))
        return x

class ResBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, padding=1, kernel_size=3, stride=stride, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        # print(x.shape)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

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

class torch_filter(nn.Module):
    def __init__(self,
                 filter1_weight=np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]) / 9,  #均值滤波
                 filter2_weight=np.array([[0, -2, 0], [-2, 9, -2], [0, -2, 0]]),  #高通滤波
                 is_grad=False):
        super(torch_filter, self).__init__()
        assert type(filter1_weight) == np.ndarray
        k=filter1_weight.shape[0]
        filters1=torch.tensor(filter1_weight).unsqueeze(dim=0).unsqueeze(dim=0)
        filters2 = torch.tensor(filter2_weight).unsqueeze(dim=0).unsqueeze(dim=0)

        filters1 = filters1.expand(64, -1, -1, -1)
        filters2 = filters2.expand(64, -1, -1, -1)
        # filters = torch.cat([filter, filter, filter], dim=0)

        self.conv1 = nn.Conv2d(64, 64, kernel_size=k, groups=64, bias=False, padding=int((k-1)/2))
        self.conv1.weight.data.copy_(filters1)
        self.conv1.requires_grad_(is_grad)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=k, groups=64, bias=False, padding=int((k-1)/2))
        self.conv2.weight.data.copy_(filters2)

        self.conv2.requires_grad_(is_grad)
    def forward(self,x):
        output = self.conv1(x)
        # output = torch.clip(output, 0, 1)
        output = torch.sigmoid(output)
        output = self.conv2(output)
        output = torch.sigmoid(output)
        # output = torch.clip(output, 0, 1)
        return output

class VesselSegNet(nn.Module):
    def __init__(self, bk=ResBlock,num_classes=1, back_bone='resnet50',
                 pretrained=False):
        super(VesselSegNet, self).__init__()
        self.num_classes = num_classes

        self.conv1 = Conv_Bn_Activation(3, 64, 3, 1, 'relu')
        self.conv2 = Conv_Bn_Activation(64, 128, 7, 2, 'relu')
        self.conv3 = Conv_Bn_Activation(128, 64, 3, 1, 'relu')
        self.conv4 = Conv_Bn_Activation(64, 64, 3, 1, 'relu')

        self.attblock1 = AttBlock(64,64)
        self.dblock1 = DACblock(64)
        self.at_c1 = ChannelAttention(64)

        self.conv5 = Conv_Bn_Activation(64, 64, 3, 1, 'relu')
        self.conv6 = Conv_Bn_Activation(64, 64, 3, 1, 'relu')
        self.conv7 = Conv_Bn_Activation(64, 64, 3, 1, 'relu')

        self.attblock2 = AttBlock(64, 64)
        self.dblock2 = DACblock(64)
        self.at_c2 = ChannelAttention(64)

        self.attblock3 = AttBlock(64, 64)
        self.dblock3 = DACblock(64)
        self.at_c3 = ChannelAttention(64)
        
        bb = []
        for i in range(4):
            bb.append(bk(64,64))
        self.conv_ff = nn.Sequential(*bb)

        # self.at1 = CBAM(64)

        self.out1 = nn.Conv2d(64, self.num_classes, kernel_size=3, stride=1, padding=1)
        self.out2 = nn.Conv2d(64, self.num_classes, kernel_size=3, stride=1, padding=1)
        self.out3 = nn.Conv2d(64, self.num_classes, kernel_size=3, stride=1, padding=1)

        self.back_bone1 = build_backbone(back_bone, pretrained)

    def _upsample(self, x, h, w):
        return F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)  # b, 128, 128, 128
        x = F.interpolate(x, scale_factor=2, mode='bilinear')
        x = self.conv3(x)  # b, 64, 256, 256
        x = self.conv_ff(x)
        # x = self.conv4(x)  # b, 64, 512, 512

        px0 = self.back_bone1(x)

        x = x * px0
        # x = F.interpolate(x, scale_factor=2, mode='bilinear')
        x = self.conv5(x)
        x = self.attblock1(x)
        x = self.dblock1(x)
        x = self.at_c1(x) * x

        out1_f = self.out1(x)
        out1_f = out1_f + px0

        x = x * out1_f
        x = self.conv6(x)

        x = self.attblock2(x)
        x = self.dblock2(x)
        x = self.at_c2(x) * x
        out2_f = self.out2(x)

        out2_f = out2_f + out1_f
        x = x * out2_f
        x = self.conv7(x)
        x = self.attblock3(x)
        x = self.dblock3(x)
        x = self.at_c3(x) * x
        out3_f = self.out3(x)
        out3_f = out3_f + out2_f

        return out1_f, out2_f, out3_f
