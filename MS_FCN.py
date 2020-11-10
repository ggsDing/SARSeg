import torch.nn as nn
from torch.nn import functional as F
from torchvision import models
import torch

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride)

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class MS_FCN(nn.Module):
    def __init__(self, in_channels=3, num_classes=7, pretrained=True):
        super(MS_FCN, self).__init__()
        resnet = models.resnet50(pretrained)
        newconv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if in_channels>3: newconv1.weight.data[:, 3:in_channels, :, :].copy_(resnet.conv1.weight.data[:, 0:in_channels-3, :, :])
          
        self.layer0 = nn.Sequential(newconv1, resnet.bn1, resnet.relu)
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.S1 = nn.Conv2d(256, num_classes, kernel_size=1)
        self.S2 = nn.Conv2d(512, num_classes, kernel_size=1)
        self.S3 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.S4 = nn.Conv2d(2048, num_classes, kernel_size=1)

    def forward(self, x):
        x_size = x.size()
        
        x = self.layer0(x)
        x = self.maxpool(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        
        out1 = self.S1(x1)
        out2 = self.S2(x2)
        out3 = self.S3(x3)
        out4 = self.S4(x4)
        
        out1 = F.upsample(out1, x_size[2:], mode='bilinear')
        out2 = F.upsample(out2, x_size[2:], mode='bilinear')
        out3 = F.upsample(out3, x_size[2:], mode='bilinear')
        out4 = F.upsample(out4, x_size[2:], mode='bilinear')
        
        out = out1+out2+out3+out4
        
        return out