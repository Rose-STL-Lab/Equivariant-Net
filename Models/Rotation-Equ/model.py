##########################################
### Rotational Equivariant Neural Nets ###
##########################################



################Rot_ResNet######################
import os
import torch
import dill
from e2cnn import gspaces
from e2cnn import nn
import numpy as np
from torch.utils import data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Rot_Resblock(torch.nn.Module):
    def __init__(self, input_channels, hidden_dim, kernel_size, N):
        super(Rot_Resblock, self).__init__()
        
        r2_act = gspaces.Rot2dOnR2(N = N)
        feat_type_in = nn.FieldType(r2_act, input_channels*[r2_act.regular_repr])
        feat_type_hid = nn.FieldType(r2_act, hidden_dim*[r2_act.regular_repr])
        
        self.layer1 = nn.SequentialModule(
            nn.R2Conv(feat_type_in, feat_type_hid, kernel_size = kernel_size, padding = (kernel_size - 1)//2),
            nn.InnerBatchNorm(feat_type_hid),
            nn.ReLU(feat_type_hid)
        ) 
        
        self.layer2 = nn.SequentialModule(
            nn.R2Conv(feat_type_hid, feat_type_hid, kernel_size = kernel_size, padding = (kernel_size - 1)//2),
            nn.InnerBatchNorm(feat_type_hid),
            
        )    
        self.relu = nn.ReLU(feat_type_hid)
        
        self.upscale = None
        if input_channels != hidden_dim:
            self.upscale = nn.SequentialModule(
                nn.R2Conv(feat_type_in, feat_type_hid, kernel_size = kernel_size, padding = (kernel_size - 1)//2),
                nn.InnerBatchNorm(feat_type_hid),
                nn.ReLU(feat_type_hid)
            )    
        
    def forward(self, xx):
        residual = xx
        out = self.layer1(xx)
        out = self.layer2(out)
        if self.upscale:
            out += self.upscale(residual)
        else:
            out += residual
        out = self.relu(out)
        
        return out

class ResNet(torch.nn.Module):
    def __init__(self, input_frames, kernel_size, N):
        super(ResNet, self).__init__()
        r2_act = gspaces.Rot2dOnR2(N = N)
        self.feat_type_in = nn.FieldType(r2_act, input_frames*[r2_act.trivial_repr])#irrep(1)
        self.feat_type_in_hid = nn.FieldType(r2_act, 16*[r2_act.regular_repr])
        self.feat_type_hid_out = nn.FieldType(r2_act, 256*[r2_act.regular_repr])
        self.feat_type_out = nn.FieldType(r2_act, [r2_act.trivial_repr])#irrep(1)
        
        self.input_layer = nn.SequentialModule(
            nn.R2Conv(self.feat_type_in, self.feat_type_in_hid, kernel_size = kernel_size, padding = (kernel_size - 1)//2),
            nn.InnerBatchNorm(self.feat_type_in_hid),
            nn.ReLU(self.feat_type_in_hid)
        )
        
        layers = [self.input_layer]
        layers += [Rot_Resblock(16, 32, kernel_size, N)]
        layers += [Rot_Resblock(32, 64, kernel_size, N)]
        layers += [Rot_Resblock(64, 128, kernel_size, N)]
        layers += [Rot_Resblock(128, 256, kernel_size, N)]
        layers += [nn.R2Conv(self.feat_type_hid_out, self.feat_type_out, kernel_size = kernel_size, padding = (kernel_size - 1)//2)]
        self.model = torch.nn.Sequential(*layers)
    
        
    def forward(self, xx):
        xx = nn.GeometricTensor(xx, self.feat_type_in)
        out = self.model(xx)
        return out.tensor
    

    
##################Unet######################

class conv2d(torch.nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride, N, activation = True, deconv = False, last_deconv = False):
        super(conv2d, self).__init__()       
        r2_act = gspaces.Rot2dOnR2(N = N)
        
        feat_type_in = nn.FieldType(r2_act, input_channels*[r2_act.regular_repr])
        feat_type_hid = nn.FieldType(r2_act, output_channels*[r2_act.regular_repr])
        if not deconv:
            if activation:
                self.layer = nn.SequentialModule(
                    nn.R2Conv(feat_type_in, feat_type_hid, kernel_size = kernel_size, stride = stride, padding = (kernel_size - 1)//2),
                    nn.InnerBatchNorm(feat_type_hid),
                    nn.ReLU(feat_type_hid)
                ) 
            else:
                self.layer = nn.R2Conv(feat_type_in, feat_type_hid, kernel_size = kernel_size, stride = stride,padding = (kernel_size - 1)//2)
        else:
            if last_deconv:
                feat_type_in = nn.FieldType(r2_act, input_channels*[r2_act.regular_repr])
                feat_type_hid = nn.FieldType(r2_act, output_channels*[r2_act.trivial_repr])#irrep(1)
                self.layer = nn.R2Conv(feat_type_in, feat_type_hid, kernel_size = kernel_size, stride = stride, padding = 0)
            else:
                self.layer = nn.SequentialModule(
                        nn.R2Conv(feat_type_in, feat_type_hid, kernel_size = kernel_size, stride = stride, padding = 0),
                        nn.InnerBatchNorm(feat_type_hid),
                        nn.ReLU(feat_type_hid)
                    ) 
    
    def forward(self, xx):
        return self.layer(xx)
    
class deconv2d(torch.nn.Module):
    def __init__(self, input_channels, output_channels, N, last_deconv = False):
        super(deconv2d, self).__init__()
        self.conv2d = conv2d(input_channels = input_channels, output_channels = output_channels, kernel_size = 4, 
                             activation = True, stride = 1, N = N, deconv = True, last_deconv = last_deconv)
        r2_act = gspaces.Rot2dOnR2(N = N)
        self.feat_type = nn.FieldType(r2_act, input_channels*[r2_act.regular_repr])
        
    def pad(self, xx):
        new_xx = torch.zeros(xx.shape[0], xx.shape[1], xx.shape[2]*2 + 3, xx.shape[3]*2 + 3)
        new_xx[:,:,:-3,:-3][:,:,1::2,1::2] = xx
        new_xx = nn.GeometricTensor(new_xx, self.feat_type)
        return new_xx
    
    def forward(self, xx):
        out = self.pad(xx).to(device)
        return self.conv2d(out)
    
class U_net(torch.nn.Module):
    def __init__(self, input_frames, kernel_size, N):
        super(U_net, self).__init__()
        r2_act = gspaces.Rot2dOnR2(N = N)
        self.feat_type_in = nn.FieldType(r2_act, input_frames*[r2_act.trivial_repr])#irrep(1)
        self.feat_type_in_hid = nn.FieldType(r2_act, 32*[r2_act.regular_repr])
        self.feat_type_hid_out = nn.FieldType(r2_act, (16 + input_frames)*[r2_act.trivial_repr])#irrep(1)
        self.feat_type_out = nn.FieldType(r2_act, [r2_act.trivial_repr])#irrep(1)
        
        self.conv1 = nn.SequentialModule(
            nn.R2Conv(self.feat_type_in, self.feat_type_in_hid, kernel_size = kernel_size, stride = 2, padding = (kernel_size - 1)//2),
            nn.InnerBatchNorm(self.feat_type_in_hid),
            nn.ReLU(self.feat_type_in_hid)
        )
        self.conv1_1 = conv2d(32, 32, kernel_size = kernel_size, stride = 1, N = N)
        self.conv2 = conv2d(32, 64, kernel_size = kernel_size, stride = 1, N = N)
        #self.conv2_1 = conv2d(64, 64, kernel_size = kernel_size, stride = 1, N = N)
        self.conv3 = conv2d(64, 128, kernel_size = kernel_size, stride = 2, N = N)
        #self.conv3_1 = conv2d(128, 128, kernel_size = kernel_size, stride = 1, N = N)
        self.conv4 = conv2d(128, 256, kernel_size = kernel_size, stride = 2, N = N)
        #self.conv4_1 = conv2d(256, 256, kernel_size = kernel_size, stride = 1, N = N)

        self.deconv3 = deconv2d(256, 64, N)
        self.deconv2 = deconv2d(192, 32, N)
        self.deconv1 = deconv2d(96, 16, N, last_deconv = True)
    
        self.output_layer = nn.R2Conv(self.feat_type_hid_out, self.feat_type_out, kernel_size = kernel_size, padding = (kernel_size - 1)//2)
      

    def forward(self, x):
        
        x = nn.GeometricTensor(x, self.feat_type_in)
        out_conv1 = self.conv1_1(self.conv1(x))
        out_conv2 = self.conv2(out_conv1)#)self.conv2_1(
        out_conv3 = self.conv3(out_conv2)#)self.conv3_1(
        out_conv4 = self.conv4(out_conv3)#)self.conv4_1(

        out_deconv3 = self.deconv3(out_conv4.tensor)
        concat3 = torch.cat((out_conv3.tensor, out_deconv3.tensor), 1)
        out_deconv2 = self.deconv2(concat3)
        concat2 = torch.cat((out_conv2.tensor, out_deconv2.tensor), 1)
        out_deconv1 = self.deconv1(concat2)
        
        concat0 = torch.cat((x.tensor, out_deconv1.tensor), 1)
        concat0 = nn.GeometricTensor(concat0, self.feat_type_hid_out)
        out = self.output_layer(concat0)
        return out.tensor
 