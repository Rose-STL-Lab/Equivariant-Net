"""
Uniform Motion Equivariant ResNet and U-net
"""

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
class um_conv2d(nn.Module):
    def __init__(self, 
                 input_channels, 
                 output_channels, 
                 kernel_size, 
                 um_dim = 2, # The number of channels of each frame
                 activation = True, # whether to use activation functions
                 stride = 1, 
                 deconv = False # Whether this is used as a deconvolutional layer
                ):
        """
        Uniform Motion Equivariant 2D Convolutional Layers
        """
        super(um_conv2d, self).__init__()
        self.activation = activation
        self.input_channels = input_channels
        self.kernel_size = kernel_size
        self.um_dim = um_dim 
        self.stride = stride
        
        # Applied to unfolded input tensors (stride == kernel_size) 
        self.conv2d = nn.Conv2d(input_channels, output_channels, kernel_size, stride = kernel_size, bias = True)
        self.pad_size = (kernel_size - 1)//2
        self.input_channels = self.input_channels
        self.batchnorm = nn.BatchNorm2d(output_channels)
        self.deconv = deconv
      
    
    def unfold(self, x):
        """
        Extracts sliding local blocks from a batched input tensor.
        """
        if not self.deconv:
            x = F.pad(x, ((self.pad_size, self.pad_size)*2), mode='replicate')
            
        # Unfold input tensor
        out = F.unfold(x, kernel_size = self.kernel_size)
        out = out.reshape(out.shape[0], self.input_channels, self.kernel_size, self.kernel_size, out.shape[-1])
        
        # Batch_size x (in_channels x kernel_size x kernel_size) x 64 x 64
        out = out.reshape(out.shape[0], self.input_channels, self.kernel_size, self.kernel_size, int(np.sqrt(out.shape[-1])), int(np.sqrt(out.shape[-1])))
        
        if self.stride > 1:
            return out[:,:,:,:,::self.stride,::self.stride]
        return out
    
    def subtract_mean(self, x):
        """
        Shifts the mean of input sliding local blocks to zero.
        """   
        # Calculates and Subtracts mean velocity 
        out = x.reshape(x.shape[0], self.input_channels//self.um_dim, self.um_dim, self.kernel_size, self.kernel_size, x.shape[-2], x.shape[-1])
        avgs = out.mean((1,3,4), keepdim=True)
        out = out - avgs
        
        # Reshape the input
        out = out.reshape(out.shape[0], self.input_channels, self.kernel_size, self.kernel_size, x.shape[-2], x.shape[-1]).transpose(2,4).transpose(-1,-2)
        out = out.reshape(out.shape[0], self.input_channels, x.shape[-2]*self.kernel_size, x.shape[-1], self.kernel_size)
        
        # Batch_size x in_channels x (64 x kernel_size) x (64 x kernel_size)
        out = out.reshape(out.shape[0], self.input_channels, x.shape[-2]*self.kernel_size, x.shape[-1]*self.kernel_size)
        return out, avgs.squeeze(3).squeeze(3)
    
    
    def add_mean(self, out, avgs):
        """
        Shifts the mean of the outputs back.
        """   
        out = out.reshape(out.shape[0], out.shape[1]//self.um_dim, self.um_dim, out.shape[-2], out.shape[-1])
        out = out + avgs
        out = out.reshape(out.shape[0], -1, out.shape[-2], out.shape[-1])
        return out
    
    def forward(self, x, shift_back = True):
        x = self.unfold(x)
        x, avgs = self.subtract_mean(x)
        out = self.conv2d(x)
        
        if self.activation:
            out = F.relu(out)
            
        if shift_back:
            out = self.add_mean(out, avgs)
        return out
    

class um_deconv2d(nn.Module):
    def __init__(self, input_channels, output_channels):
        """
        Uniform Motion Equivariant 2D Transposed Convolutional Layers
        """
        super(um_deconv2d, self).__init__()
        self.conv2d = um_conv2d(input_channels = input_channels, output_channels = output_channels, kernel_size = 4, um_dim = 2,
                             activation = True, stride = 1, deconv = True)
    
    def pad(self, x):
        # Add padding inside of the input tensor
        # To preserve uniform motion equivariance, we use the input tensor itself as padding instead of zero.
        # Because the padding zero would change the original mean of input sliding blocks.
        pad_x = torch.zeros(x.shape[0], x.shape[1], x.shape[2]*2, x.shape[3]*2)
        pad_x[:,:,::2,::2].copy_(x)
        pad_x[:,:,1::2,::2].copy_(x)
        pad_x[:,:,::2,1::2].copy_(x)
        pad_x[:,:,1::2,1::2].copy_(x)
        pad_x = F.pad(pad_x, (1,2,1,2), mode='replicate')
        return pad_x
    
    def forward(self, x):
        out = self.pad(x).to(device)
        return self.conv2d(out)


# Uniform Motion Equivariant U_net.
class Unet_UM(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size):
        super(Unet_UM, self).__init__()
        self.input_channels = input_channels
        self.conv1 = um_conv2d(input_channels, 64, kernel_size = kernel_size, stride=2)
        self.conv2 = um_conv2d(64, 128, kernel_size = kernel_size, stride=2)
        self.conv2_2 = um_conv2d(128, 128, kernel_size = kernel_size, stride = 1)
        self.conv3 = um_conv2d(128, 256, kernel_size = kernel_size, stride=2)
        self.conv3_1 = um_conv2d(256, 256, kernel_size = kernel_size, stride=1)
        self.conv4 = um_conv2d(256, 512, kernel_size = kernel_size, stride=2)
        self.conv4_1 = um_conv2d(512, 512, kernel_size = kernel_size, stride=1)

        self.deconv3 = um_deconv2d(512, 128)
        self.deconv2 = um_deconv2d(384, 64)
        self.deconv1 = um_deconv2d(192, 32)
        self.deconv0 = um_deconv2d(96, 16)
    
        self.output_layer = um_conv2d(16 + input_channels, output_channels, kernel_size=kernel_size, activation = False)

    def forward(self, x):
        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2_2(self.conv2(out_conv1))
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


# Uniform Motion Equivariant ResNet.
class um_resblock(nn.Module):
    def __init__(self, input_channels, hidden_dim, kernel_size):
        super(um_resblock, self).__init__()
        self.layer1 = um_conv2d(input_channels, hidden_dim, kernel_size)
        self.layer2 = um_conv2d(hidden_dim, hidden_dim, kernel_size)
        if input_channels != hidden_dim:
            self.upscale = um_conv2d(input_channels, hidden_dim, kernel_size)

        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        
    def forward(self, x):
        out = self.layer1(x)  
        if self.input_channels == self.hidden_dim:
            out = self.layer2(out, False) + x
        else:
            out = self.layer2(out, False) + self.upscale(x)
             
        return out

class ResNet_UM(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size):
        super(ResNet_UM, self).__init__()
        layers = [um_resblock(input_channels, 64, kernel_size), um_resblock(64, 64, kernel_size)]
        layers += [um_resblock(64, 128, kernel_size), um_resblock(128, 128, kernel_size)]
        layers += [um_resblock(128, 256, kernel_size), um_resblock(256, 256, kernel_size)]
        layers += [um_resblock(256, 512, kernel_size), um_resblock(512, 512, kernel_size)]
        layers += [um_conv2d(512, output_channels, kernel_size = kernel_size, activation = False)]
        self.model = nn.Sequential(*layers)
             
    def forward(self, x):
        out = self.model(x)
        return out