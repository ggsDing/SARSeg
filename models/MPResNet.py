import torch
import numpy as np
import torch.nn as nn
from torchvision import models
from torch.nn import functional as F

def conv1x1(in_planes, out_planes, bias=False):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=bias)

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class FCN_res34(nn.Module):
    def __init__(self, in_channels=3, num_classes=7, pretrained=False):
        super(FCN_res34, self).__init__()
        resnet = models.resnet34(pretrained)
        newconv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        newconv1.weight.data[:, 0:3, :, :].copy_(resnet.conv1.weight.data[:, 0:3, :, :])
        if in_channels>3: newconv1.weight.data[:, 3:in_channels, :, :].copy_(resnet.conv1.weight.data[:, 0:in_channels-3, :, :])
          
        self.layer0 = nn.Sequential(newconv1, resnet.bn1, resnet.relu)
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        for n, m in self.layer3.named_modules():
            if 'conv1' in n or 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv1' in n or 'downsample.0' in n:
                m.stride = (1, 1)

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock,self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu = nn.ReLU()

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu(x)
        return x

class MPResNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=7):
        super(MPResNet, self).__init__()
        self.FCN = FCN_res34(in_channels, num_classes, pretrained=True)
        
        self.res1 = models.resnet34(pretrained=True).layer3
        self.res2 = models.resnet34(pretrained=True).layer4
        self.res3 = models.resnet34(pretrained=True).layer4        
        for n, m in self.res3.named_modules():
            if 'conv1' in n or 'downsample.0' in n: m.stride = (1, 1)
            
        self.dec4 = DecoderBlock(512, 512)
        self.dec3 = DecoderBlock(512, 512)
        
        self.classifier = nn.Sequential( conv1x1(512, 128), nn.BatchNorm2d(128), nn.ReLU(), conv1x1(128, num_classes, bias=True) )
        self.classifier_aux = nn.Sequential( conv1x1(512, 128), nn.BatchNorm2d(128), nn.ReLU(), conv1x1(128, num_classes, bias=True) )
        
    def forward(self, x):
        x_size = x.size()
        
        x = self.FCN.layer0(x)  #size:1/2
        x = self.FCN.maxpool(x) #size:1/4
        x = self.FCN.layer1(x)  #size:1/4
        e2 = self.FCN.layer2(x) #size:1/8
        e3 = self.res1(e2)      #size:1/16
        e4 = self.res2(e3)      #size:1/32
        
        e3 = self.res3(e3)
        e2 = self.FCN.layer3(e2)
        e2 = self.FCN.layer4(e2)
        aux = self.classifier_aux(e2)
        
        d3 = self.dec4(e4) + e3
        d2 = self.dec3(d3) + e2
        
        out = self.classifier(d2)
        
        # recommended weights for loss function: 0.3*aux_loss + 0.7*main_loss
        return F.upsample(aux, x_size[2:], mode='bilinear'), F.upsample(out, x_size[2:], mode='bilinear')
