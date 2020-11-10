import torch
import torch.nn as nn
import torchvision

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

class _EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=True):
        super(_EncoderBlock, self).__init__()
        self.downsample = downsample
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        layers = [
            conv3x3(in_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            conv3x3(out_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        if self.downsample:
            x = self.maxpool(x)
        x = self.encode(x)
        return x


class _DecoderBlock(nn.Module):
    def __init__(self, in_channels_high, in_channels_low, out_channels):
        super(_DecoderBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels_high, in_channels_high, kernel_size=2, stride=2)
        in_channels = in_channels_high + in_channels_low
        self.decode = nn.Sequential(
            conv3x3(in_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            conv3x3(out_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, low_feat):
        x = self.up(x)
        x = torch.cat((x, low_feat), dim=1)
        x = self.decode(x)        
        return x


class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1):
        super(UNet, self).__init__()
        
        self.Enc0 = _EncoderBlock(in_channels, 64, downsample=False)
        self.Enc1 = _EncoderBlock(64,128)
        self.Enc2 = _EncoderBlock(128,256)
        self.bridge = _EncoderBlock(256, 512)  
        
        #self.Enc3 = _EncoderBlock(256, 512)
        #self.bridge = _EncoderBlock(512, 1024)        
        #self.Dec3 = _DecoderBlock(1024, 512, 512)
        self.Dec2 = _DecoderBlock(512, 256, 256)
        self.Dec1 = _DecoderBlock(256, 128, 128)
        self.Dec0 = _DecoderBlock(128, 64, 64)
        
        self.classifier = nn.Conv2d(64, num_classes, kernel_size=1)       

    def forward(self, x):
        x_size = x.size()
                
        enc0 = self.Enc0(x)
        enc1 = self.Enc1(enc0)
        enc2 = self.Enc2(enc1)
        enc3 = self.bridge(enc2)
        #enc3 = self.Enc3(enc2)
        #enc4 = self.bridge(enc3)
        #dec3 = self.Dec3(enc4, enc3)
        dec2 = self.Dec2(enc3, enc2)
        dec1 = self.Dec1(dec2, enc1)
        dec0 = self.Dec0(dec1, enc0)
        
        out = self.classifier(dec0) 
        
        return out
