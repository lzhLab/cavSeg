import math
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable
import torch.nn.functional as F
import torch

model_urls = {
    'resnet50':'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth'
    }

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
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



class ResNet(nn.Module):

    def __init__(self, block, layers, pretrained=False):
        super(ResNet, self).__init__()
        self.inplanes = 128

        # Modules
        self.conv1 = nn.Conv2d(64, 128, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        # self.layer1 = self._make_layer(block, 64, layers[0])
        # self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        # self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.layer1 = self._make_layer(block, 128, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 128, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 128, layers[3], stride=2)
        self._init_weights()
        self.out = nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1, bias=False)

        if pretrained:
            self._load_pretrained_model()


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride), 
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def forward(self, x):
       x = self.conv1(x)
       x = self.bn1(x)
       x = self.relu(x)
       low_level_feat1 = F.interpolate(x, scale_factor=2, mode='bilinear')
       x = self.maxpool(x)
       x = self.layer1(x)
       low_level_feat2 = F.interpolate(x, scale_factor=4, mode='bilinear')
       x = self.layer2(x)
       low_level_feat3 = F.interpolate(x, scale_factor=8, mode='bilinear')
       x = self.layer3(x)
       low_level_feat4 = F.interpolate(x, scale_factor=16, mode='bilinear')

       # print(low_level_feat1.shape, low_level_feat2.shape, low_level_feat3.shape, low_level_feat4.shape)
       # x = self.layer4(x)
       # low_level_feat5 = x
       # return [low_level_feat1, low_level_feat2, low_level_feat3, low_level_feat4, low_level_feat5]
       x = low_level_feat1 + low_level_feat2 + low_level_feat3 + low_level_feat4
       x = self.out(x)
       # x = F.interpolate(x, scale_factor=16, mode='bilinear')
       x = torch.sigmoid(x)
       return x

    def _init_weights(self):
       for m in self.modules():
           if isinstance(m, nn.Conv2d):
               n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
               m.weight.data.normal_(0, math.sqrt(2. / n))
           elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def _load_pretrained_model(self):
        pretrained_dict = model_zoo.load_url(model_urls['resnet101'])
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrained_dict.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)


def ResNet50(pretrained=True):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 5, 3])
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model

def ResNet101(pretrained=True):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], pretrained=pretrained)
    return model


if __name__ == "__main__":
    import torch
    model = ResNet50(pretrained=False)
    print(model)
    # model.cuda()
    input = torch.rand(1, 64, 256, 256)
    # input = Variable(input.cuda())
    input = Variable(input)
    low_level_features = model(input)

    print(len(low_level_features))
    print(low_level_features[0].shape)
    print(low_level_features[1].shape)
"""
torch.Size([1, 64, 128, 128])
torch.Size([1, 256, 128, 128])
torch.Size([1, 512, 64, 64])
torch.Size([1, 1024, 32, 32])
torch.Size([1, 2048, 16, 16])

torch.Size([1, 64, 256, 256])
torch.Size([1, 256, 256, 256])
torch.Size([1, 512, 128, 128])
torch.Size([1, 1024, 64, 64])
torch.Size([1, 2048, 32, 32])
"""
