# from DataLoader import * 

from Normolization import *
import torch
import torch.nn as nn
import cv2
from torchsummary import summary
import math



class BasicBlock(nn.Module):
    expansion = 1  

    def __init__(self,inplanes,planes,stride=1,downsample = None):
        super(BasicBlock,self).__init__()
        self.conv1 = nn.Conv2d(inplanes,planes,kernel_size=3,
                            stride=stride,padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes,planes,kernel_size=3,
                    padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride 
    
    def forward(self,x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        
        out += residual
        out = self.relu(out)

        return out 


class ResNet(nn.Module):

    def __init__(self,block,layers,num_classes):

        self.inplanes = 64 
        super(ResNet,self).__init__()
        self.layer1 = self.__make_layer(block,64,layers[0])
        self.layer2 = self.__make_layer(block,128,layers[1],stride=2)
        self.layer3 = self.__make_layer(block,256,layers[2],stride=2)
        self.layer4 = self.__make_layer(block,512,layers[3],stride=2)
        self.avgpool = nn.AvgPool2d(2)
        self.fc = nn.Linear(512*block.expansion*2*2,num_classes)
        self.bnfc = nn.BatchNorm1d(num_classes)
        #init parameters in network 
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                n = m.kernel_size[0]*m.kernel_size[1]*m.out_channels
                m.weight.data.normal_(0,math.sqrt(2./n))
            elif isinstance(m,nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m,nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0),-1)
        out = self.fc(out)
        out = self.bnfc(out)

        return out 


    def __make_layer(self,block,planes,num_blocks,stride=1):
        downsample = None
        if stride !=1 or self.inplanes!=planes*block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes,planes*block.expansion,
                kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )
        layers = []
        layers.append(block(self.inplanes,planes,stride,downsample))
        self.inplanes = planes*block.expansion 
        for i in range(1,num_blocks):
            layers.append(block(self.inplanes,planes))
        
        return nn.Sequential(*layers)


def select_norm(norm, dim):
    '''
    select normolization method
    norm: the one in ['gln','cln','bn']
    '''
    if norm not in ['gln', 'cln', 'bn']:
        raise RuntimeError("only accept['gln','cln','bn']")
    if norm == 'gln':
        return GlobalLayerNorm(dim, elementwise_affine=True)
    elif norm == 'cln':
        return CumulativeLayerNorm(dim, trainable=True)
    elif norm == 'bn':
        return nn.BatchNorm1d(dim)


class Conv1D(nn.Conv1d):
    '''
       Applies a 1D convolution over an input signal composed of several input planes.
    '''

    def __init__(self, *args, **kwargs):
        super(Conv1D, self).__init__(*args, **kwargs)

    def forward(self, x, squeeze=False):
        # x: N x C x L
        if x.dim() not in [2, 3]:
            raise RuntimeError("{} accept 2/3D tensor as input".format(
                self.__name__))
        x = super().forward(x if x.dim() == 3 else torch.unsqueeze(x, 1))
        if squeeze:
            x = torch.squeeze(x)
        return x


class Conv1D_Block(nn.Module):
    '''
    sub-block with the exponential growth dilation factors 2**d
    '''

    def __init__(self, in_channels=256, out_channels=128, kernel_size=3,
                 dilation=1, norm='bn', causal=True):
        super(Conv1D_Block, self).__init__()
        # this conv1d determines the number of channels
        # self.linear = Conv1D(in_channels, out_channels,kernel_size=3,stride=1,padding=1)  # set kernel_size=1
        self.ReLu = nn.ReLU(True)
        self.norm = select_norm(norm, out_channels)
        # keep time length unchanged
        self.pad = (dilation*(kernel_size-1))//2 if not causal else (
            dilation * (kernel_size-1))

        self.DepthwiseConv = Conv1D(out_channels, out_channels,
                                    kernel_size, groups=out_channels, padding=self.pad,dilation=dilation)
        self.SeparableConv = Conv1D(out_channels, in_channels, 1)
        self.causal = causal

    def forward(self, x):
        # c = self.linear(x)
        c = self.ReLu(x)
        c = self.norm(c)
        c = self.DepthwiseConv(c)
        if self.causal:
            c = c[:, :, :-self.pad]
        c = self.SeparableConv(c)
        return x+c


class Conv1D_Block_in_Visual(nn.Module):
    '''
    sub-block in visual part
    '''
    def __init__(self,in_channels=256,out_channels=512,kernel_size=3):
        super(Conv1D_Block_in_Visual,self).__init__()
        self.linear = Conv1D(in_channels,out_channels,kernel_size=1)
        self.ReLu = nn.ReLU(True)
        self.DepthwiseConv = Conv1D(out_channels, out_channels,
                                    kernel_size, groups=out_channels)
        self.SeparableConv = Conv1D(out_channels, in_channels, 1)
    
    def forward(self,x):
        out = self.linear(x)
        out = self.ReLu(x)
