import torch.nn as nn
import torch
import torch.nn.functional as F
from functools import partial

nonlinearity = partial(F.relu, inplace=True)


class RestNetBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(RestNetBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        output = self.conv1(x)
        output = F.relu(self.bn1(output))
        output = self.conv2(output)
        output = self.bn2(output)
        return F.relu(x + output)

class RestNetDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(RestNetDownBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride[0], padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride[1], padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.extra = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride[0], padding=0),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        extra_x = self.extra(x)
        output = self.conv1(x)
        out = F.relu(self.bn1(output))

        out = self.conv2(out)
        out = self.bn2(out)
        return F.relu(extra_x + out)

class RestPoolBlock(nn.Module):
    def __init__(self, in_channels, out_channels, pool_size):
        super(RestPoolBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.max_pool = nn.MaxPool2d(kernel_size=pool_size, stride=1, padding=pool_size//2)
        #self.max_pool = nn.AvgPool2d(kernel_size=pool_size, stride=1, padding=pool_size//2)
    def forward(self, x):

        x = self.conv1(x)
        x = self.max_pool(x)
        out = F.relu(self.bn1(x))

        return out

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

        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out

        return out

class Mish(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x * (torch.tanh(torch.nn.functional.softplus(x)))
        return x


class AttBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(AttBlock, self).__init__()
        self.conv1x1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.spatial_attention1 = SpatialAttention(5)

        self.conv3x3 = RestPoolBlock(in_planes, planes, 3)
        self.bn2 = nn.BatchNorm2d(planes)

        self.spatial_attention2 = SpatialAttention(5)
        self.conv5x5 = RestPoolBlock(in_planes, planes, 5)

        self.bn3 = nn.BatchNorm2d(planes)
        self.spatial_attention3 = SpatialAttention(5)
        self.conv7x7 = RestPoolBlock(in_planes, planes, 7)

        self.bn4 = nn.BatchNorm2d(planes)
        self.spatial_attention4 = SpatialAttention(5)

    def forward(self, x):
        identity = x
        # print(x.shape)
        # out1 = self.bn1(self.conv1x1(x))
        out1 = self.conv1x1(x)
        out1 = self.spatial_attention1(out1) * out1

        # out2 = self.bn2(self.conv3x3(x))
        out2 = self.conv3x3(x)
        out2 = self.spatial_attention2(out2) * out2

        # out3 = self.bn3(self.conv5x5(x))
        out3 = self.conv5x5(x)
        out3 = self.spatial_attention3(out3) * out3

        # out4 = self.bn4(self.conv7x7(x))
        out4 = self.conv7x7(x)
        out4 = self.spatial_attention4(out4) * out4

        out = identity + out1 + out2 + out3 + out4

        return out


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

class for_denoise(nn.Module):
    def __init__(self, in_channels=256):
        super().__init__()
        self.conv_1 = Conv_Bn_Activation(in_channels, 2*in_channels, 3, 1, 'relu')
        self.pool = nn.AvgPool2d(2)
        self.conv_2 = Conv_Bn_Activation(2*in_channels, 2*in_channels, 3, 1, 'relu')
        self.conv_3 = Conv_Bn_Activation(2*in_channels, 2*in_channels, 3, 1, 'relu')
        self.conv_4 = Conv_Bn_Activation(2 * in_channels, in_channels, 3, 1, 'relu')

    def forward(self, x):
        x1 = self.pool(x)
        x1 = self.conv_1(x1)
        x2 = self.pool(x1)
        x2 = self.conv_2(x2)
        x2 = F.interpolate(x2, scale_factor=2, mode='bilinear')
        x2 = self.conv_3(x2)
        x = x1+x2
        x = self.conv_4(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear')
        return x

class vessel_single_stage(nn.Module):
    def __init__(self, in_channel=128, out_num=2, nblocks=2):
        super(vessel_single_stage, self).__init__()
        self.conv_s = Conv_Bn_Activation(in_channel, in_channel, 3, 1, 'relu')
        self.conv_up = Conv_Bn_Activation(in_channel, in_channel//2, 3, 1, 'relu')
        self.dblock = DACblock(in_channel//2)
        self.conv_up2 = Conv_Bn_Activation(in_channel//2, in_channel, 5, 2, 'relu')


        self.attblock = AttBlock(in_channel, in_channel)
        # self.attblock = CSP(in_channel, in_channel, nblocks)
        self.conv_s = Conv_Bn_Activation(in_channel, in_channel, 3, 1, 'relu')
        self.ca = ChannelAttention(in_channel)

        self.dn = for_denoise(in_channel)

        self.conv_e = Conv_Bn_Activation(3*in_channel, in_channel, 3, 1, 'relu')

        self.conv_out = nn.Conv2d(in_channel, 1, kernel_size=3, stride=1, padding=1)
    def forward(self, x, pro):
        x0 = self.conv_s(x)
        x00 = x0
        if pro is not None:
            x0 = x0 * torch.sigmoid(pro)

        x1 = F.interpolate(x0, scale_factor=2, mode='bilinear')
        x1 = self.conv_up(x1)
        x1 = self.dblock(x1)
        x1 = self.conv_up2(x1)

        x2 = self.attblock(x0)
        x2 = self.conv_s(x2)
        x2 = self.ca(x2)*x2

        x3 = self.dn(x0)

        x = torch.cat([x1, x2, x3],1)
        x = self.conv_e(x)
        out = self.conv_out(x)

        return x+x00, out


class VesselSegNet(nn.Module):
    def __init__(self, stage_num=3, num_classes=1, stage_in_channel=128,
                 pretrained=False, **kwargs):
        super(VesselSegNet, self).__init__()
        self.num_classes = num_classes
        self.stage_num = stage_num
        self.stage_in_channel = stage_in_channel
        self.conv1 = Conv_Bn_Activation(3, 64, 7, 2, 'relu')
        self.layer1 = nn.Sequential(RestNetBasicBlock(64, 64, 1),
                                    RestNetBasicBlock(64, 64, 1))

        self.layer2 = nn.Sequential(RestNetDownBlock(64, 128, [2, 1]),
                                    RestNetBasicBlock(128, 128, 1))

        #self.conv2 = Conv_Bn_Activation(64, 128, 5, 2, 'relu')

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

    def _upsample(self, x, h, w):
        return F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)

    def forward(self, x):
        _,_,h,w = x.size()
        x = self.conv1(x)
        # x = self.conv2(x)  # b, 128, 128, 128
        x = self.layer1(x)
        x = self.layer2(x)

        outputs = list()
        for i in range(self.stage_num):
            if i<1:
                x, res = eval('self.stage' + str(i))(x, None)
            else:
                x, res = eval('self.stage' + str(i))(x, outputs[i-1])
            outputs.append(res)

        outputs = [self._upsample(out ,h, w) for out in outputs]
        x = F.interpolate(x, scale_factor=4, mode='bilinear')
        x = self.conv3(x)
        fout = self.out(x)
        return outputs, fout
