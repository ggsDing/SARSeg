import torch
import numpy as np
import torch.nn as nn
from utils.misc import initialize_weights
from torchvision import models
from torch.nn import functional as F
from models.FCN_8s import FCN_res34 as FCN

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class ResBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.res = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes))
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        residual = self.res(residual)

        out += residual
        out = self.relu(out)

        return out

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

def copy_weight(model, model_copy):
    mp = list(model.parameters())
    mcp = list(model_copy.parameters())
    n = len(mp)
    for i in range(0, n):
        mp[i].data[:] = mcp[i].data[:]

class MSResNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=7):
        super(MSResNet, self).__init__()
        self.FCN = FCN(in_channels, num_classes, pretrained=False)
        #self.SA = Spatial_AttentionV2(128, reduction=8, pool_window=10, add_input=False)
        
        self.res1 = models.resnet34(pretrained=False).layer3
        self.res2 = models.resnet34(pretrained=False).layer4
        self.res3 = models.resnet34(pretrained=False).layer4        
        for n, m in self.res3.named_modules():
            if 'conv1' in n or 'downsample.0' in n: m.stride = (1, 1)
            
        self.dec4 = DecoderBlock(512, 512)
        self.dec3 = DecoderBlock(512, 512)
        
        self.classifier = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, num_classes, kernel_size=1) )
            
        initialize_weights(self.classifier, self.dec4, self.dec3)
        
    def forward(self, x):
        x_size = x.size()
        
        x = self.FCN.layer0(x)  #size:1/2
        x = self.FCN.maxpool(x)    #size:1/4
        x = self.FCN.layer1(x)  #size:1/4
        e2 = self.FCN.layer2(x) #size:1/8
        e3 = self.res1(e2)      #size:1/16
        e4 = self.res2(e3)      #size:1/32
        
        e3 = self.res3(e3)
        e2 = self.FCN.layer3(e2)
        e2 = self.FCN.layer4(e2)
        
        d3 = self.dec4(e4) + e3
        d2 = self.dec3(d3) + e2
        
        out = self.classifier(d2)
        
        return F.upsample(out, x_size[2:], mode='bilinear')