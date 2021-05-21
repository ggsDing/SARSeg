# An implementation of the model in "A new fully convolutional neural network for semantic segmentation of polarimetric sar imagery in complex land cover ecosystem”

import torch.nn as nn
from torch.nn import functional as F
from torchvision import models
import torch

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride)

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class ResBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes):
        super(ResBlock, self).__init__()
        self.conv0 = nn.Sequential( conv1x1(inplanes, planes), nn.BatchNorm2d(planes) )
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv3x3(planes, planes)
        self.bn3 = nn.BatchNorm2d(planes)        
    
    def forward(self, x):
        identity = self.conv0(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out

class BasicConv2d(nn.Module):
    def __init__(
        self, in_channels, out_channels, **kwargs) -> None:
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

class Inception(nn.Module):
    def __init__(self, in_channels: int, in_planes=64):
        super(Inception, self).__init__()
        self.branch1x1 = BasicConv2d(in_channels, in_planes, kernel_size=1)

        self.branch5x5_1 = BasicConv2d(in_channels, in_planes*2, kernel_size=1)
        self.branch5x5_2 = BasicConv2d(in_planes*2, in_planes*2, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = BasicConv2d(in_channels, in_planes, kernel_size=1)
        self.branch3x3dbl_2 = BasicConv2d(in_planes, in_planes, kernel_size=3, padding=1)
        

    def _forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)

        outputs = [branch1x1, branch5x5, branch3x3dbl]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)

class Inc_FCN(nn.Module):
    def __init__(self, in_channels=3, num_classes=7, pretrained=True):
        super(Inc_FCN, self).__init__()
        
        self.conv1 = nn.Sequential(conv3x3(in_channels, 64), nn.BatchNorm2d(64), nn.ReLU())
        self.res1 = ResBlock(64, 64)
        self.conv2 = nn.Sequential(conv3x3(64, 128), nn.BatchNorm2d(128), nn.ReLU())
        self.conv3 = nn.Sequential(conv3x3(128, 128), nn.BatchNorm2d(128), nn.ReLU())
        self.res2 = ResBlock(128, 128)
        self.conv4 = nn.Sequential(conv3x3(128, 256), nn.BatchNorm2d(256), nn.ReLU())
        self.inception1 = Inception(256, 64)
        self.inception2 = Inception(256, 128)
                
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.deconv1 = nn.Sequential(nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=2, stride=2), nn.BatchNorm2d(512), nn.ReLU())
        self.deconv2 = nn.Sequential(nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=2, stride=2), nn.BatchNorm2d(128), nn.ReLU())
        self.conv5 = nn.Sequential(conv3x3(512, 256), nn.BatchNorm2d(256), nn.ReLU())
        self.conv6 = nn.Sequential(conv3x3(256+128, 128), nn.BatchNorm2d(128), nn.ReLU())
        self.conv7 = conv1x1(128+64, num_classes)
        
        self.skip1 = ResBlock(128, 128)
        self.skip2 = ResBlock(64, 64)

    def forward(self, x):
        x_size = x.size()
        
        x = self.conv1(x)
        x2 = self.res1(x)
        x = self.conv2(x2)
        x = self.maxpool(x)
        
        x = self.conv3(x)
        x1 = self.res2(x)
        x = self.conv4(x1)
        x = self.maxpool(x)
        
        x = self.inception1(x)
        x = self.inception2(x)
        
        skip1 = self.skip1(x1)
        skip2 = self.skip2(x2)
        
        x = self.deconv1(x)
        x = self.conv5(x)
        x = torch.cat([x, skip1], 1)
        x = self.conv6(x)
        x = self.deconv2(x)
        x = torch.cat([x, skip2], 1)
        out = self.conv7(x)
        
        return out