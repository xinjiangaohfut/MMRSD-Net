#encoding:utf-8
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets 
from torchvision import transforms
from torchvision.utils import save_image
from torch.autograd import Variable
import numpy as np
import pdb
from PIL import Image
import os
import scipy.io as sio
import torch.utils.data as data
import math


def denorm(x):
    relu=nn.ReLU()
    mea=torch.mean(x)
    out=relu(x-mea)
    out=torch.sqrt(out)
    return out

# def denorm(x):
#     out = (x + 1) / 2
#     return out.clamp(0, 1)



def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

# def get_residue(tensor):
#     max_channel = torch.max(tensor, dim=1, keepdim=True)
#     min_channel = torch.min(tensor, dim=1, keepdim=True)
#     res_channel = max_channel[0] - min_channel[0]
#     return res_channel

def conv(c_in, c_out, k_size, stride=2, pad=1, bn=True):
    """Custom convolutional layer for simplicity."""
    layers = []
    layers.append(nn.Conv2d(c_in, c_out, k_size, stride, pad))
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    return nn.Sequential(*layers)

# 

# 

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class RemUNet(nn.Module):
    def __init__(self, n_channels, n_classes,bilinear=True):
        super(RemUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.ini = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, input, res,x11, x22, x33, x44, x55, x66,x77, x88, x99):
    # def forward(self, input):
        batch_size, row, col = input.size(0), input.size(2), input.size(3)
        xgray = input-res.expand(res.shape[0],3,res.shape[2],res.shape[3])
        x=input
        x1 = self.ini(x)-x11
        x2 = self.down1(x1)-x22
        x3 = self.down2(x2)-x33
        x4 = self.down3(x3)-x44
        x5 = self.down4(x4)-x55
        x6 = self.up1(x5, x4)-x66
        x7 = self.up2(x6, x3)-x77
        x8 = self.up3(x7, x2)-x88
        x9 = self.up4(x8, x1)-x99
        x10= self.outc(x9)
        # return x10
        return x10,xgray,x1,x2,x3,x4,x5,x6,x7,x8,x9

# class MakUNet(nn.Module):
#     def __init__(self, n_channels, n_classes,bilinear=True):
#         super(MakUNet, self).__init__()
#         self.n_channels = n_channels
#         self.n_classes = n_classes
#         self.bilinear = bilinear

#         self.ini = DoubleConv(n_channels, 64)
#         self.down1 = Down(64, 128)
#         self.down2 = Down(128, 256)
#         self.down3 = Down(256, 512)
#         self.down4 = Down(512, 512)
#         self.up1 = Up(1024, 256, bilinear)
#         self.up2 = Up(512, 128, bilinear)
#         self.up3 = Up(256, 64, bilinear)
#         self.up4 = Up(128, 64, bilinear)
#         self.outc = OutConv(64, n_classes)

#     def forward(self, input, res):
#         batch_size, row, col = input.size(0), input.size(2), input.size(3)
#         x = input+res.expand(res.shape[0],3,res.shape[2],res.shape[3])
#         x1 = self.ini(x)
#         # x11=self.ini(x)
#         x2 = self.down1(x1)
#         # x12=self.down1(x1)
#         x3 = self.down2(x2)
#         # x13=self.down2(x2)
#         x4 = self.down3(x3)
#         # x14=self.down3(x3)
#         x5 = self.down4(x4)
#         # x15=self.down4(x4)
#         x6 = self.up1(x5, x4)
#         x7 = self.up2(x6, x3)
#         x8 = self.up3(x7, x2)
#         x9 = self.up4(x8, x1)
#         x10= self.outc(x9)
#         # return x10
#         return x10,x1,x2,x3,x4,x5,x6,x7,x8,x9


class HfUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(HfUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, input):
        x1 = self.inc(input)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.up1(x5, x4)
        x7 = self.up2(x6, x3)
        x8 = self.up3(x7, x2)
        x9 = self.up4(x8, x1)
        x = self.outc(x9)
        return x, x1, x2, x3, x4, x5, x6,x7, x8, x9

# class GrayUNet(nn.Module):
#     def __init__(self, n_channels, n_classes, bilinear=True):
#         super(GrayUNet, self).__init__()
#         self.n_channels = n_channels
#         self.n_classes = n_classes
#         self.bilinear = bilinear
#         self.inc = DoubleConv(n_channels, 64)
#         self.incc = DoubleConv(n_channels, 64)
#         self.down1 = Down(64, 128)
#         self.down2 = Down(128, 256)
#         self.down3 = Down(256, 512)
#         self.down4 = Down(512, 512)
#         self.up1 = Up(1024, 256, bilinear)
#         self.up2 = Up(512, 128, bilinear)
#         self.up3 = Up(256, 64, bilinear)
#         self.up4 = Up(128, 64, bilinear)
#         self.outc = OutConv(64, n_classes)

#     def forward(self, input,x11, x22, x33, x44, x55, x66,x77, x88, x99):
#         x1 = self.inc(input)-x11     
#         x2 = self.down1(x1)-x22    
#         x3 = self.down2(x2)-x33     
#         x4 = self.down3(x3)-x44    
#         x5 = self.down4(x4)-x55      
#         x6 = self.up1(x5, x4)-x66    
#         x7 = self.up2(x6, x3)-x77     
#         x8 = self.up3(x7, x2)-x88       
#         x9 = self.up4(x8, x1)-x99       
#         x = self.outc(x9)    
#         return x

# class ColorUNet(nn.Module):
#     def __init__(self, n_channels, n_classes, bilinear=True):
#         super(ColorUNet, self).__init__()
#         self.n_channels = n_channels
#         self.n_classes = n_classes
#         self.bilinear = bilinear
#         self.inc = DoubleConv(n_channels, 64)
#         self.incc = DoubleConv(n_channels, 64)
#         self.down1 = Down(64, 128)
#         self.down2 = Down(128, 256)
#         self.down3 = Down(256, 512)
#         self.down4 = Down(512, 512)
#         self.up1 = Up(1024, 256, bilinear)
#         self.up2 = Up(512, 128, bilinear)
#         self.up3 = Up(256, 64, bilinear)
#         self.up4 = Up(128, 64, bilinear)
#         self.outc = OutConv(64, n_classes)

#     def forward(self, input):
#         x1 = self.inc(input)     
#         x2 = self.down1(x1)    
#         x3 = self.down2(x2)     
#         x4 = self.down3(x3)     
#         x5 = self.down4(x4)       
#         x6 = self.up1(x5, x4)    
#         x7 = self.up2(x6, x3)     
#         x8 = self.up3(x7, x2)       
#         x9 = self.up4(x8, x1)     
#         x = self.outc(x9)    
#         return x

class D(nn.Module):
    def __init__(self):
        super(D,self).__init__()
        self.conv1=nn.Sequential(
            conv(3,64,4,bn=False),
            nn.ReLU(inplace=True))

        self.conv2=nn.Sequential(
            conv(64,64*2,4),
            nn.ReLU(inplace=True))

        self.conv3=nn.Sequential(
            conv(64*2,64*4,4),
            nn.ReLU(inplace=True))

        self.conv4=nn.Sequential(
            conv(64*4,64*8,4),
            nn.ReLU(inplace=True))

        self.fc=nn.Sequential(
            nn.Linear(6*6*8*64,1),
            nn.Sigmoid())
 

    def forward(self,x):
        out=self.conv1(x)

        out=self.conv2(out)

        out=self.conv3(out)

        out=self.conv4(out)

        out=out.view(out.size(0),-1)

        out=self.fc(out)

        return out


class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h =  (x.size()[2]-1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

