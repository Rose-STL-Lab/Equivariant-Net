"""
Non-equivariant ResNet and U-net
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


def conv(input_channels, output_channels, kernel_size, stride):
    return nn.Sequential(
        nn.Conv2d(input_channels, output_channels, kernel_size = kernel_size,
                  stride = stride, padding=(kernel_size - 1) // 2),
        nn.ReLU()
    )

def deconv(input_channels, output_channels):
    return nn.Sequential(
        nn.ConvTranspose2d(input_channels, output_channels, kernel_size = 4,
                           stride = 2, padding=1),
        nn.ReLU()
    )

class Unet(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size):
        super(Unet, self).__init__()
        self.input_channels = input_channels
        self.conv1 = conv(input_channels, 64, kernel_size=kernel_size, stride=2)
        self.conv2 = conv(64, 128, kernel_size=kernel_size, stride=2)
        self.conv2_1 = conv(128, 128, kernel_size=kernel_size, stride=1)
        self.conv3 = conv(128, 256, kernel_size=kernel_size, stride=2)
        self.conv3_1 = conv(256, 256, kernel_size=kernel_size, stride=1)
        self.conv4 = conv(256, 512, kernel_size=kernel_size, stride=2)
        self.conv4_1 = conv(512, 512, kernel_size=kernel_size, stride=1)

        self.deconv3 = deconv(512, 128)
        self.deconv2 = deconv(384, 64)
        self.deconv1 = deconv(192, 32)
        self.deconv0 = deconv(96, 16)
        
        self.output_layer = nn.Conv2d(16 + input_channels, output_channels, kernel_size=kernel_size,
                                      stride = 1, padding=(kernel_size - 1) // 2)

    def forward(self, x):
        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2_1(self.conv2(out_conv1))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))

        out_deconv3 = self.deconv3(out_conv4)
        concat3 = torch.cat((out_conv3, out_deconv3), 1)
        out_deconv2 = self.deconv2(concat3)
        concat2 = torch.cat((out_conv2, out_deconv2), 1)
        out_deconv1 = self.deconv1(concat2)
        concat1 = torch.cat((out_conv1, out_deconv1), 1)
        out_deconv0 = self.deconv0(concat1)
        concat0 = torch.cat((x, out_deconv0), 1)
        out = self.output_layer(concat0)
        return out

    
class Resblock(nn.Module):
    def __init__(self, input_channels, hidden_dim, kernel_size):
        super(Resblock, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim, kernel_size = kernel_size, padding = (kernel_size-1)//2),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU()
        ) 
        self.layer2 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size = kernel_size, padding = (kernel_size-1)//2),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU()
        ) 
        
        if input_channels != hidden_dim:
            self.upscale = nn.Sequential(
                nn.Conv2d(input_channels, hidden_dim, kernel_size = kernel_size, padding = (kernel_size-1)//2),
                nn.LeakyReLU()
                )        
        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        
        
    def forward(self, xx):
        out = self.layer1(xx)  
        if self.input_channels != self.hidden_dim:
            out = self.layer2(out) + self.upscale(xx)
        else:
            out = self.layer2(out) + xx
        return out
    

class ResNet(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size):
        super(ResNet, self).__init__()
        layers = [Resblock(input_channels, 64, kernel_size), Resblock(64, 64, kernel_size)]
        layers += [Resblock(64, 128, kernel_size), Resblock(128, 128, kernel_size)]
        layers += [Resblock(128, 256, kernel_size), Resblock(256, 256, kernel_size)]
        layers += [Resblock(256, 512, kernel_size), Resblock(512, 512, kernel_size)]
        layers += [nn.Conv2d(512, output_channels, kernel_size = kernel_size, padding = (kernel_size-1)//2)]
        self.model = nn.Sequential(*layers)
             
    def forward(self, xx):
        out = self.model(xx)
        return out