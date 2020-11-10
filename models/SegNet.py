#https://github.com/yassouali/pytorch_segmentation/blob/master/models/segnet.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from itertools import chain
from math import ceil
from utils import initialize_weights

class SegNet_vgg(nn.Module):
    def __init__(self, in_channels=3, num_classes=1, pretrained=False):
        super(SegNet_vgg, self).__init__()
        vgg_bn = models.vgg16_bn(pretrained = pretrained)
        encoder = list(vgg_bn.features.children())

        # Adjust the input size
        if in_channels > 3:
            newconv = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)
            newconv.weight.data[:, 0:3, :, :].copy_(encoder[0].weight.data[:, :, :, :])
            newconv.weight.data[:, 3:in_channels, :, :].copy_(encoder[0].weight.data[:, 0:in_channels-3, :, :])
            encoder[0] = newconv

        # Encoder, VGG without any maxpooling
        self.stage1_encoder = nn.Sequential(*encoder[:6]) #num:7
        self.stage2_encoder = nn.Sequential(*encoder[7:13]) #num:7
        self.stage3_encoder = nn.Sequential(*encoder[14:23]) #num:10
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        # Decoder, same as the encoder but reversed, maxpool will not be used
        decoder = encoder
        decoder = [i for i in list(reversed(decoder)) if not isinstance(i, nn.MaxPool2d)]
        # Replace the last conv layer
        decoder[-1] = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        # When reversing, we also reversed conv->batchN->relu, correct it
        decoder = [item for i in range(0, len(decoder), 3) for item in decoder[i:i+3][::-1]]
        # Replace some conv layers & batchN after them
        for i, module in enumerate(decoder):
            if isinstance(module, nn.Conv2d):
                if module.in_channels != module.out_channels:
                    decoder[i+1] = nn.BatchNorm2d(module.in_channels)
                    decoder[i] = nn.Conv2d(module.out_channels, module.in_channels, kernel_size=3, stride=1, padding=1)

        self.stage1_decoder = nn.Sequential(*decoder[18:27]) #num:10
        self.stage2_decoder = nn.Sequential(*decoder[27:33]) #num:10
        self.stage3_decoder = nn.Sequential(*decoder[33:], #num:7
                nn.Conv2d(64, num_classes, kernel_size=3, stride=1, padding=1) )
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)

        initialize_weights(self.stage1_decoder, self.stage2_decoder, self.stage3_decoder)

    def forward(self, x):
        # Encoder
        x = self.stage1_encoder(x)
        x1_size = x.size()
        x, indices1 = self.pool(x) #scale: 1/2

        x = self.stage2_encoder(x)
        x2_size = x.size()
        x, indices2 = self.pool(x) #scale: 1/4

        x = self.stage3_encoder(x)
        x3_size = x.size()
        x, indices3 = self.pool(x) #scale: 1/8

        # Decoder
        x = self.unpool(x, indices=indices3, output_size=x3_size)
        x = self.stage1_decoder(x)

        x = self.unpool(x, indices=indices2, output_size=x2_size)
        x = self.stage2_decoder(x)

        x = self.unpool(x, indices=indices1, output_size=x1_size)
        x = self.stage3_decoder(x)

        return x

class Segnet(nn.Module):
    def __init__(self, input_nc, output_nc):
        super(Segnet, self).__init__()

        # Encoder
        self.conv11 = nn.Conv2d(input_nc, 64, kernel_size=3, padding=1)   ##[4,256,256]-->[64,256,256]
        self.bn11 = nn.BatchNorm2d(64)
        self.conv12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn12 = nn.BatchNorm2d(64)

        self.conv21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn21 = nn.BatchNorm2d(128)
        self.conv22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn22 = nn.BatchNorm2d(128)

        self.conv31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn31 = nn.BatchNorm2d(256)
        self.conv32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn32 = nn.BatchNorm2d(256)
        self.conv33 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn33 = nn.BatchNorm2d(256)

        self.conv41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn41 = nn.BatchNorm2d(512)
        self.conv42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn42 = nn.BatchNorm2d(512)
        self.conv43 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn43 = nn.BatchNorm2d(512)

        self.conv51 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn51 = nn.BatchNorm2d(512)
        self.conv52 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn52 = nn.BatchNorm2d(512)
        self.conv53 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn53 = nn.BatchNorm2d(512)

        # Decoder
        self.conv53d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn53d = nn.BatchNorm2d(512)
        self.conv52d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn52d = nn.BatchNorm2d(512)
        self.conv51d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn51d = nn.BatchNorm2d(512)

        self.conv43d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn43d = nn.BatchNorm2d(512)
        self.conv42d = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn42d = nn.BatchNorm2d(512)
        self.conv41d = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.bn41d = nn.BatchNorm2d(256)

        self.conv33d = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn33d = nn.BatchNorm2d(256)
        self.conv32d = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn32d = nn.BatchNorm2d(256)
        self.conv31d = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn31d = nn.BatchNorm2d(128)

        self.conv22d = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn22d = nn.BatchNorm2d(128)
        self.conv21d = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn21d = nn.BatchNorm2d(64)

        self.conv12d = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn12d = nn.BatchNorm2d(64)
        self.conv11d = nn.Conv2d(64, output_nc, kernel_size=3, padding=1)

    def forward(self, x):
        # Stage 1
        x11 = F.relu(self.bn11(self.conv11(x)), inplace=True)
        x12 = F.relu(self.bn12(self.conv12(x11)), inplace=True)
        x1p, id1 = F.max_pool2d(x12, kernel_size=2, stride=2, return_indices=True)

        # Stage 2
        x21 = F.relu(self.bn21(self.conv21(x1p)), inplace=True)
        x22 = F.relu(self.bn22(self.conv22(x21)), inplace=True)
        x2p, id2 = F.max_pool2d(x22, kernel_size=2, stride=2, return_indices=True)

        # Stage 3
        x31 = F.relu(self.bn31(self.conv31(x2p)), inplace=True)
        x32 = F.relu(self.bn32(self.conv32(x31)), inplace=True)
        x33 = F.relu(self.bn33(self.conv33(x32)), inplace=True)
        x3p, id3 = F.max_pool2d(x33, kernel_size=2, stride=2, return_indices=True)

        # Stage 4
        x41 = F.relu(self.bn41(self.conv41(x3p)), inplace=True)
        x42 = F.relu(self.bn42(self.conv42(x41)), inplace=True)
        x43 = F.relu(self.bn43(self.conv43(x42)), inplace=True)
        x4p, id4 = F.max_pool2d(x43, kernel_size=2, stride=2, return_indices=True)

        # Stage 5
        x51 = F.relu(self.bn51(self.conv51(x4p)), inplace=True)
        x52 = F.relu(self.bn52(self.conv52(x51)), inplace=True)
        x53 = F.relu(self.bn53(self.conv53(x52)), inplace=True)
        x5p, id5 = F.max_pool2d(x53, kernel_size=2, stride=2, return_indices=True)

        # Stage 5d
        x5d = F.max_unpool2d(x5p, id5, kernel_size=2, stride=2)
        x53d = F.relu(self.bn53d(self.conv53d(x5d)), inplace=True)
        x52d = F.relu(self.bn52d(self.conv52d(x53d)), inplace=True)
        x51d = F.relu(self.bn51d(self.conv51d(x52d)), inplace=True)

        # Stage 4d
        x4d = F.max_unpool2d(x51d, id4, kernel_size=2, stride=2)
        x43d = F.relu(self.bn43d(self.conv43d(x4d)), inplace=True)
        x42d = F.relu(self.bn42d(self.conv42d(x43d)), inplace=True)
        x41d = F.relu(self.bn41d(self.conv41d(x42d)), inplace=True)

        # Stage 3d
        x3d = F.max_unpool2d(x41d, id3, kernel_size=2, stride=2)
        x33d = F.relu(self.bn33d(self.conv33d(x3d)), inplace=True)
        x32d = F.relu(self.bn32d(self.conv32d(x33d)), inplace=True)
        x31d = F.relu(self.bn31d(self.conv31d(x32d)), inplace=True)

        # Stage 2d
        x2d = F.max_unpool2d(x31d, id2, kernel_size=2, stride=2)
        x22d = F.relu(self.bn22d(self.conv22d(x2d)), inplace=True)
        x21d = F.relu(self.bn21d(self.conv21d(x22d)), inplace=True)

        # Stage 1d
        x1d = F.max_unpool2d(x21d, id1, kernel_size=2, stride=2)
        x12d = F.relu(self.bn12d(self.conv12d(x1d)), inplace=True)    ##需要激活函数吗？
        x11d = self.conv11d(x12d)                             ##是不是少了bn层？
        # output = t.sigmoid(x11d)      ##sigmoid和softmax和全连接的区别？  #本来这应该是是一个像素分类层？

        return x11d

