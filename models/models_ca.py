import torch.nn as nn
import torch
import torch.nn.functional as F

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

class Mish(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x * (torch.tanh(torch.nn.functional.softplus(x)))
        return x

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

class VesselSegNet(nn.Module):
    def __init__(self, num_classes=1):
        super(VesselSegNet, self).__init__()
        self.num_classes = num_classes
        self.maxpool = nn.MaxPool2d(2)
        self.upsample1 = nn.Upsample(1024)
        self.upsample2 = nn.Upsample(512)
        self.upsample3 = nn.Upsample(256)
        self.upsample4 = nn.Upsample(128)
        self.sa = SpatialAttention()

        self.conv1 = Conv_Bn_Activation(1, 64, 3, 1, 'leaky')
        self.conv2 = Conv_Bn_Activation(64, 8, 1, 1, 'leaky')

        self.conv3 = Conv_Bn_Activation(64, 128, 3, 1, 'leaky')

        self.conv4_2 = Conv_Bn_Activation(8, 32, 3, 2, 'leaky')

        self.conv5 = Conv_Bn_Activation(128, 256, 3, 1, 'leaky')


        self.conv6 = Conv_Bn_Activation(256, 128, 1, 1, 'leaky')

        self.smth1 = Conv_Bn_Activation(32+128, 256, 3, 1, 'leaky')

        self.conv7 = Conv_Bn_Activation(128, 256, 3, 1, 'leaky')
        self.conv8 = Conv_Bn_Activation(256, 512, 3, 1, 'leaky')
        self.conv9 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
        self.conv8_2 = Conv_Bn_Activation(256, 256, 3, 2, 'leaky')

        self.smth2 = Conv_Bn_Activation(256 + 256, 256, 3, 1, 'leaky')

        self.conv10 = Conv_Bn_Activation(256, 512, 3, 1, 'leaky')
        self.conv11 = Conv_Bn_Activation(512, 1024, 3, 1, 'leaky')
        self.conv12 = Conv_Bn_Activation(1024, 512, 1, 1, 'leaky')

        self.conv11_2 = Conv_Bn_Activation(256, 256, 3, 2, 'leaky')

        self.smth3 = Conv_Bn_Activation(256 + 512, 256, 3, 1, 'leaky')
        self.conv13 = Conv_Bn_Activation(256, 64, 1, 1, 'leaky')
        self.lastconv = nn.Conv2d(64, self.num_classes, kernel_size=1, stride=1, padding=0)

    def _upsample(self, x, h, w):
        return F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)

    def forward(self, x):
        _, _, h, w = x.size()
        x = self.conv1(x)

        up1 = self.conv2(x)
        up1 = self.upsample1(up1)
        #up1 = self.conv2(up1)
        # print(up1.shape)
        dn1 = self.maxpool(x)
        dn1 = self.conv3(dn1)
        # print(dn1.shape)
        att1 = self.sa(up1)*up1
        att1 = self.conv4_2(att1) # 还原


        f1 = self.conv5(dn1)
        f1 = self.conv6(f1)
        f1 = self.upsample2(f1)  # 还原

        out1 = torch.cat((att1, f1), dim=1)
        out1 = self.smth1(out1)

        dn2 = self.maxpool(dn1)
        dn2 = self.conv7(dn2)

        att2 = self.sa(out1) * out1
        att2 = self.conv8_2(att2)

        f2 = self.conv8(dn2)
        f2 = self.conv9(f2)
        f2 = self.upsample3(f2)

        out2 = torch.cat((att2, f2), dim=1)
        out2 = self.smth2(out2)

        dn3 = self.maxpool(dn2)
        dn3 = self.conv10(dn3)

        att3 = self.sa(out2) * out2
        att3 = self.conv11_2(att3)

        f3 = self.conv11(dn3)
        f3 = self.conv12(f3)
        f3 = self.upsample4(f3)

        out3 = torch.cat((att3, f3), dim=1)
        out3 = self.smth3(out3)

        out = self.conv13(out1+self._upsample(out2, h, w)+self._upsample(out3, h, w))
        return self.lastconv(out)

