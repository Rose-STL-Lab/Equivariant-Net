"""
Scale Equivariant ResNet and U-net
"""

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class scale_conv2d(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size, 
                 l = 3, # Number of levels of input
                 sout = 5, # Number of scales we model in the convolution layer. 
                 activation = True, # If add the activation function at the end
                 stride = 1,
                 deconv = False):
        super(scale_conv2d, self).__init__()
        self.out_channels= out_channels
        self.in_channels = in_channels
        self.l = l 
        self.sout = sout 
        self.activation = activation
        self.kernel_size = kernel_size
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        weight_shape = (out_channels, l, 2, in_channels//2, kernel_size, kernel_size) # The shape of scale equivariant conv2d kernels
        self.stdv = math.sqrt(1. / (kernel_size * kernel_size * in_channels * l))
        self.weights = nn.Parameter(torch.Tensor(*weight_shape))
        self.reset_parameters()
        self.stride = stride
        self.deconv = deconv
    
        
    def reset_parameters(self):
        self.weights.data.uniform_(-self.stdv, self.stdv)
        if self.bias is not None:
            self.bias.data.fill_(0)
            
    def shrink_kernel(self, kernel, up_scale):
        """
        Shrink the kernel via boundary padding and grid sampling. 
        """
        up_scale = torch.tensor(up_scale).float()
        # boundary padding based on the scaling law
        pad_in = (torch.ceil(up_scale**2).int())*((kernel.shape[2]-1)//2)
        pad_h = (torch.ceil(up_scale).int())*((kernel.shape[3]-1)//2)
        pad_w = (torch.ceil(up_scale).int())*((kernel.shape[4]-1)//2)
        padded_kernel = F.pad(kernel, (pad_w, pad_w, pad_h, pad_h, pad_in, pad_in))
        delta = up_scale%1
        
        if delta == 0:
            shrink_factor = 1
        else:
            # shrink_factor for coordinates.
            shrink_factor = (((kernel.shape[4]-1))/(padded_kernel.shape[-1]-1)*(up_scale+1))
            
            # Adjustment to deal with weird filtering on the grid sample function.
            shrink_factor = 1.5*(shrink_factor-0.5)**3 + 0.57   

        grid = torch.meshgrid(torch.linspace(-1, 1, kernel.shape[2])*(shrink_factor**2),
                              torch.linspace(-1, 1, kernel.shape[3])*shrink_factor, 
                              torch.linspace(-1, 1, kernel.shape[4])*shrink_factor)

        grid = torch.cat([grid[2].unsqueeze(0).unsqueeze(-1), 
                          grid[1].unsqueeze(0).unsqueeze(-1), 
                          grid[0].unsqueeze(0).unsqueeze(-1)], dim = -1).repeat(kernel.shape[0],1,1,1,1)

        new_kernel = F.grid_sample(padded_kernel, grid.to(device))
        if kernel.shape[-1] - 2*up_scale > 0:
            new_kernel = new_kernel * (kernel.shape[-1]**2/((kernel.shape[-1] - 2*up_scale)**2 + 0.01))
        return new_kernel
    
    def dilate_kernel(self, kernel, dilation):
        """
        upscale the kernel via inside padding and grid sampling. 
        """
        if dilation == 0:
            return kernel 
        # inside padding based on the scaling law
        dilation = torch.tensor(dilation).float()
        delta = dilation%1

        d_in = torch.ceil(dilation**2).int()
        new_in = kernel.shape[2] + (kernel.shape[2]-1)*d_in

        d_h = torch.ceil(dilation).int()
        new_h = kernel.shape[3] + (kernel.shape[3]-1)*d_h

        d_w = torch.ceil(dilation).int()
        new_w = kernel.shape[4] + (kernel.shape[4]-1)*d_h

        new_kernel = torch.zeros(kernel.shape[0], kernel.shape[1], new_in, new_h, new_w)
        new_kernel[:,:,::(d_in+1),::(d_h+1), ::(d_w+1)] = kernel
        dilate_factor = 1
        
        new_kernel = F.pad(new_kernel, ((kernel.shape[4]-1)//2, (kernel.shape[4]-1)//2)*3)

        dilate_factor = (new_kernel.shape[-1] - 1 - (kernel.shape[4]-1)*(delta))/(new_kernel.shape[-1] - 1) 

        grid = torch.meshgrid(torch.linspace(-1, 1, new_in)*(dilate_factor**2), 
                              torch.linspace(-1, 1, new_h)*dilate_factor, 
                              torch.linspace(-1, 1, new_w)*dilate_factor)

        grid = torch.cat([grid[2].unsqueeze(0).unsqueeze(-1), 
                          grid[1].unsqueeze(0).unsqueeze(-1), 
                          grid[0].unsqueeze(0).unsqueeze(-1)], dim = -1).repeat(kernel.shape[0],1,1,1,1)

        new_kernel = F.grid_sample(new_kernel, grid)         
            
        return new_kernel[:,:,-kernel.shape[2]:]
    
    
    def forward(self, xx):
        out = []
        for s in range(self.sout):
            t = np.minimum(s + self.l, self.sout)
            inp = xx[:,s:t].reshape(xx.shape[0], -1, xx.shape[-2], xx.shape[-1])
            w = self.weights[:,:(t-s),:,:,:].reshape(self.out_channels, 2*(t-s), self.in_channels//2, self.kernel_size, self.kernel_size).to(device)
            
            if (s - self.sout//2) < 0:
                new_kernel = self.shrink_kernel(w, (self.sout//2 - s)/2).to(device)
            elif (s - self.sout//2) > 0:
                new_kernel = self.dilate_kernel(w, (s - self.sout//2)/2).to(device)
            else:
                new_kernel = w.to(device)
    
            new_kernel = new_kernel.reshape(self.out_channels, (t-s)*self.in_channels, new_kernel.shape[-2], new_kernel.shape[-1])
            
            
            if self.deconv:
                if (s - self.sout//2) > 0:          
                    conv = F.conv2d(F.pad(inp, (1,2,1,2)), new_kernel)
                else:
                    conv = F.conv2d(inp, new_kernel)
            else:
                conv = F.conv2d(inp, new_kernel, padding = ((new_kernel.shape[-2]-1)//2, (new_kernel.shape[-1]-1)//2), stride = self.stride)
            
            out.append(conv.unsqueeze(1))

        out = torch.cat(out, dim = 1) 
        if self.activation: 
            out = F.leaky_relu(out)
        
        return out 

class scale_deconv2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(scale_deconv2d, self).__init__()  
        self.conv2d = scale_conv2d(in_channels, out_channels, kernel_size = 4, deconv = True)
    
    def pad(self, xx):
        new_xx = torch.zeros(xx.shape[0], xx.shape[1], xx.shape[2], xx.shape[3]*2+3, xx.shape[4]*2+3)
        new_xx[:,:,:,:-3,:-3][:,:,:,::2,::2] = xx
        return new_xx
        
    def forward(self, xx):
        out = self.pad(xx).to(device)
        return self.conv2d(out)
    
class scale_resblock(nn.Module):
    def __init__(self, in_channels, hidden_dim, kernel_size, skip = True):
        super(scale_resblock, self).__init__()
        self.layer1 = scale_conv2d(in_channels = in_channels, out_channels = hidden_dim, kernel_size = kernel_size)
        self.layer2 = scale_conv2d(in_channels = hidden_dim, out_channels = hidden_dim, kernel_size = kernel_size) 
        self.input_channels = in_channels
        self.hidden_dim = hidden_dim
        if in_channels != hidden_dim:
            self.upscale = scale_conv2d(in_channels = in_channels, out_channels = hidden_dim, kernel_size = kernel_size, activation = False)
        
    def forward(self, xx):
        out = self.layer1(xx)
        if self.input_channels != self.hidden_dim:
            out = self.layer2(out) + self.upscale(xx)
        else:
            out = self.layer2(out) + xx 
        return out

class ResNet_Scale(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size):
        super(ResNet_Scale, self).__init__()
        self.input_layer = scale_conv2d(out_channels = 32, in_channels = input_channels, kernel_size = kernel_size)        
        layers = [self.input_layer]
        layers += [scale_resblock(32, 32, kernel_size, True), scale_resblock(32, 32, kernel_size, True)]
        layers += [scale_resblock(32, 64, kernel_size, False), scale_resblock(64, 64, kernel_size, True)]
        layers += [scale_resblock(64, 128, kernel_size, False), scale_resblock(128, 128, kernel_size, True)]
        layers += [scale_resblock(128, 128, kernel_size, True), scale_resblock(128, 128, kernel_size, True)]
        layers += [scale_conv2d(out_channels = output_channels, in_channels = 128, kernel_size = kernel_size, sout = 1, activation = False)]
        self.model = nn.Sequential(*layers)
        
    def forward(self, xx):
        out = self.model(xx)
        out = out.squeeze(1)
        return out
    
    
class Unet_Scale(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size):
        super(Unet_Scale, self).__init__()
        self.conv1 = scale_conv2d(input_channels, 32, kernel_size = kernel_size, stride=2)
        self.conv2 = scale_conv2d(32, 64, kernel_size = kernel_size, stride=2)
        self.conv2_2 = scale_conv2d(64, 64, kernel_size = kernel_size, stride = 1)
        self.conv3 = scale_conv2d(64, 128, kernel_size = kernel_size, stride=2)
        self.conv3_1 = scale_conv2d(128, 128, kernel_size = kernel_size, stride=1)
        self.conv4 = scale_conv2d(128, 256, kernel_size = kernel_size, stride=2)
        self.conv4_1 = scale_conv2d(256, 256, kernel_size = kernel_size, stride=1)

        self.deconv3 = scale_deconv2d(256, 64)
        self.deconv2 = scale_deconv2d(192, 32)
        self.deconv1 = scale_deconv2d(96, 16)
        self.deconv0 = scale_deconv2d(48, 8)
    
        self.output_layer = scale_conv2d(8 + input_channels, output_channels, kernel_size=kernel_size, activation = False, sout = 1)

    def forward(self, x):
        
        out_conv1 = self.conv1(x)
        

        out_conv2 = self.conv2_2(self.conv2(out_conv1))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_deconv3 = self.deconv3(out_conv4)

        concat3 = torch.cat((out_conv3, out_deconv3), 2)
        out_deconv2 = self.deconv2(concat3)

        concat2 = torch.cat((out_conv2, out_deconv2), 2)
        out_deconv1 = self.deconv1(concat2)

        concat1 = torch.cat((out_conv1, out_deconv1), 2)
        out_deconv0 = self.deconv0(concat1)
   
        concat0 = torch.cat((x.reshape([x.shape[0], x.shape[1], -1, x.shape[4], x.shape[5]]), out_deconv0), 2)
        out = self.output_layer(concat0)
        out = out.squeeze(1)
        return out
    
    
