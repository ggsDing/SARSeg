import torch.nn as nn
from torch.nn import functional as F
from torchvision import models
import torch
#from models.FCN_8s import FCN_res50 as FCN
from models.FCN_8s import FCN_res34 as FCN

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class ASPPModule(nn.Module):
    """
    Reference:
        Chen, Liang-Chieh, et al. *"Rethinking Atrous Convolution for Semantic Image Segmentation."*
    """

    def __init__(self, features, inner_features=256, out_features=512, dilations=(6, 12, 18)):
        super(ASPPModule, self).__init__()

        self.conv1 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                   nn.Conv2d(features, inner_features, kernel_size=1, padding=0, dilation=1,
                                             bias=False),
                                   nn.BatchNorm2d(inner_features, momentum=0.95),
                                   nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(features, inner_features, kernel_size=1, padding=0, dilation=1, bias=False),
            nn.BatchNorm2d(inner_features, momentum=0.95),
            nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(
            nn.Conv2d(features, inner_features, kernel_size=3, padding=dilations[0], dilation=dilations[0], bias=False),
            nn.BatchNorm2d(inner_features, momentum=0.95),
            nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(features, inner_features, kernel_size=3, padding=dilations[1], dilation=dilations[1], bias=False),
            nn.BatchNorm2d(inner_features, momentum=0.95),
            nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(
            nn.Conv2d(features, inner_features, kernel_size=3, padding=dilations[2], dilation=dilations[2], bias=False),
            nn.BatchNorm2d(inner_features, momentum=0.95),
            nn.ReLU(inplace=True))

        self.bottleneck = nn.Sequential(
            nn.Conv2d(inner_features * 5, out_features, kernel_size=1, padding=0, dilation=1, bias=False),
            nn.BatchNorm2d(out_features, momentum=0.95),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        _, _, h, w = x.size()

        feat1 = F.upsample(self.conv1(x), size=(h, w), mode='bilinear', align_corners=True)
        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        feat5 = self.conv5(x)
        out = torch.cat((feat1, feat2, feat3, feat4, feat5), 1)

        bottle = self.bottleneck(out)
        return bottle


class Deeplab(nn.Module):
    def __init__(self, in_channels=3, num_classes=7, pretrained=True):
        super(Deeplab, self).__init__()
        self.FCN = FCN(in_channels, num_classes)        
        #self.head = ASPPModule(2048)
        self.head = ASPPModule(512, inner_features=64, out_features=128)       
        self.fuse = nn.Sequential(nn.Conv2d(64+128, 64, kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d(64), nn.ReLU())                                
                                    
        self.classifier = nn.Conv2d(64, num_classes, kernel_size=1)
        
    def forward(self, x):
        x_size = x.size()
        
        x = self.FCN.layer0(x)
        x = self.FCN.maxpool(x)
        x0 = self.FCN.layer1(x)
        x = self.FCN.layer2(x0)
        x = self.FCN.layer3(x)
        x = self.FCN.layer4(x)
        x1 = self.head(x)
        
        x1 = F.upsample(x1, x0.size()[2:], mode='bilinear')        
        fuse = self.fuse( torch.cat([x0, x1], 1) )
        
        fuse = self.classifier(fuse)
        
        return F.upsample(fuse, x_size[2:], mode='bilinear')
