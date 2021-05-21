# An implementation of the model in  "Hrsar-net: A deep neural network for urban scene segmentation from highresolution sar data"
from __future__ import division
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.utils.data

class lunetN(nn.Module): 
    def __init__(self, numFeatures, num_classes):
        super().__init__()
        self.convM1 = nn.Conv2d(numFeatures, 16, kernel_size=5, padding=2)
        self.bnM1 = nn.BatchNorm2d(16)
        self.actM1 = nn.LeakyReLU(inplace=True)
        self.conv0a = nn.Conv2d(16, 16, kernel_size=3, padding=1*1)
        self.conv0b = nn.Conv2d(16, 16, kernel_size=3, padding=1*2, dilation=2)
        self.conv0c = nn.Conv2d(16, 16, kernel_size=3, padding=1*4, dilation=4)
        self.bn0 = nn.BatchNorm2d(3*16)
        self.act0 = nn.LeakyReLU(inplace=True)
        
        self.conv1 = nn.Conv2d(3*16, 16, kernel_size=1, padding=0, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.act1 = nn.LeakyReLU(inplace=True)
        self.conv2a = nn.Conv2d(16, 16, kernel_size=3, padding=1*1)
        self.conv2b = nn.Conv2d(16, 16, kernel_size=3, padding=1*2, dilation=2)
        self.conv2c = nn.Conv2d(16, 16, kernel_size=3, padding=1*4, dilation=4)
        self.bn2 = nn.BatchNorm2d(3*16)
        self.act2 = nn.LeakyReLU(inplace=True)
        
        self.conv3 = nn.Conv2d(3*16, 16, kernel_size=1, stride=2)
        self.bn3 = nn.BatchNorm2d(16)
        self.act3 = nn.LeakyReLU(inplace=True)
        self.conv4a = nn.Conv2d(16, 16, kernel_size=3, padding=1*1)
        self.conv4b = nn.Conv2d(16, 16, kernel_size=3, padding=1*2, dilation=2)
        self.conv4c = nn.Conv2d(16, 16, kernel_size=3, padding=1*4, dilation=4)
        self.bn4 = nn.BatchNorm2d(3*16)
        self.act4 = nn.LeakyReLU(inplace=True)
        
        self.conv5 = nn.Conv2d(3*16, 16, kernel_size=1, stride=2)
        self.bn5 = nn.BatchNorm2d(16)
        self.act5 = nn.LeakyReLU(inplace=True)
        self.conv6a = nn.Conv2d(16, 16, kernel_size=3, padding=1*1, groups=16)
        self.conv6b = nn.Conv2d(16, 16, kernel_size=3, padding=1*2, groups=16, dilation=2)
        self.conv6c = nn.Conv2d(16, 16, kernel_size=3, padding=1*4, groups=16, dilation=4)
        self.bn6 = nn.BatchNorm2d(3*16)
        self.act6 = nn.LeakyReLU(inplace=True)
        
        self.conv7 = nn.ConvTranspose2d(3*16, 16, kernel_size=4, stride=2, output_padding=0, padding=1) 
        self.bn7 = nn.BatchNorm2d(16)
        self.act7 = nn.LeakyReLU(inplace=True)
        self.conv8 = nn.ConvTranspose2d(16+3*16, 16, kernel_size=4, stride=2, output_padding=0, padding=1) 
        self.bn8 = nn.BatchNorm2d(16)
        self.act8 = nn.LeakyReLU(inplace=True)
        self.conv9 = nn.ConvTranspose2d(16+3*16, num_classes, kernel_size=4, stride=2, output_padding=0, padding=1)
        
    def forward(self, x):
        x = self.actM1(self.bnM1(self.convM1(x)))
        
        x = self.act0(self.bn0(torch.cat((self.conv0a(x),self.conv0b(x),self.conv0c(x)), dim=-3)))
        
        x = self.act1(self.bn1(self.conv1(x)))
        
        x = self.act2(self.bn2(torch.cat((self.conv2a(x),self.conv2b(x),self.conv2c(x)), dim=-3)))
        x1 = x
        x = self.act3(self.bn3(self.conv3(x)))
        
        x = self.act4(self.bn4(torch.cat((self.conv4a(x),self.conv4b(x),self.conv4c(x)), dim=-3)))
        x2 = x
        x = self.act5(self.bn5(self.conv5(x)))
        
        x = self.act6(self.bn6(torch.cat((self.conv6a(x),self.conv6b(x),self.conv6c(x)), dim=-3)))
        x = torch.cat((self.act7(self.bn7(self.conv7(x))), x2), dim=-3)
        x = torch.cat((self.act8(self.bn8(self.conv8(x))), x1), dim=-3)
        x = self.conv9(x)
        return x
		