import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils import data
import itertools
import re
import random
import time
from torch.autograd import Variable
import math
from scipy.ndimage import gaussian_filter
from torch.autograd import Variable
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
class Dataset(data.Dataset):
    def __init__(self, indices, input_length, mid, output_length, direc, stack_x):
        self.input_length = input_length
        self.mid = mid
        self.output_length = output_length
        self.stack_x = stack_x
        self.direc = direc
        self.list_IDs = indices
        
    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        ID = self.list_IDs[index]
        y = torch.load(self.direc + str(ID) + ".pt")[self.mid:(self.mid+self.output_length)]
        if self.stack_x:
            x = torch.load(self.direc + str(ID) + ".pt")[(self.mid-self.input_length):self.mid].reshape(-1, y.shape[-2], y.shape[-1])
        else:
            x = torch.load(self.direc + str(ID) + ".pt")[(self.mid-self.input_length):self.mid]
        
        return x.float(), y.float()

def train_epoch(train_loader, model, optimizer, loss_function):
    train_mse = []
    for xx, yy in train_loader:
        xx = xx.to(device)
        yy = yy.to(device)
        loss = 0
        ims = []
        for y in yy.transpose(0,1):
            im = model(xx)
            xx = torch.cat([xx[:, 2:], im], 1)
            loss += loss_function(im, y)
        train_mse.append(loss.item()/yy.shape[1]) 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_mse = round(np.sqrt(np.mean(train_mse)),5)
    return train_mse



def eval_epoch(valid_loader, model, loss_function):
    valid_mse = []
    preds = []
    trues = []
    with torch.no_grad():
        for xx, yy in valid_loader:
            xx = xx.to(device)
            yy = yy.to(device)
            loss = 0
            ims = []
            for y in yy.transpose(0,1):
                im = model(xx)
                xx = torch.cat([xx[:, 2:], im], 1)
                loss += loss_function(im, y)
                ims.append(im.unsqueeze(1).cpu().data.numpy())
                
            ims = np.concatenate(ims, axis = 1)
            preds.append(ims)
            trues.append(yy.cpu().data.numpy())
            valid_mse.append(loss.item()/yy.shape[1])
        preds = np.concatenate(preds, axis = 0)  
        trues = np.concatenate(trues, axis = 0)  
        valid_mse = round(np.sqrt(np.mean(valid_mse)), 5)
    return valid_mse, preds, trues

def test_epoch(valid_loader, model, loss_function):
    valid_mse = []
    preds = []
    trues = []
    with torch.no_grad():
        loss_curve = []
        for xx, yy in valid_loader:
            xx = xx.to(device)
            yy = yy.to(device)
            loss = 0
            ims = []
            
            for y in yy.transpose(0,1):
                im = model(xx)
                xx = torch.cat([xx[:, 2:], im], 1)
                mse = loss_function(im, y)
                loss += mse
                loss_curve.append(mse.item())
                ims.append(im.unsqueeze(1).cpu().data.numpy())
           
            ims = np.concatenate(ims, axis = 1)
            preds.append(ims)
            trues.append(yy.cpu().data.numpy())
            valid_mse.append(loss.item()/yy.shape[1])
            
        loss_curve = np.array(loss_curve).reshape(-1,yy.shape[1])
        preds = np.concatenate(preds, axis = 0)  
        trues = np.concatenate(trues, axis = 0)  
        valid_mse = np.mean(valid_mse)
        loss_curve = np.sqrt(np.mean(loss_curve, axis = 0))
    return valid_mse, preds, trues, loss_curve



### Functions for Scale equivariant models ##########

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
class gaussain_blur(nn.Module):
    def __init__(self, size, sigma, dim, channels):
        super(gaussain_blur, self).__init__()
        self.kernel = self.gaussian_kernel(size, sigma, dim, channels).to(device)
        
    def gaussian_kernel(self, size, sigma, dim, channels):

        kernel_size = 2*size + 1
        kernel_size = [kernel_size] * dim
        kernel = 1
        meshgrids = torch.meshgrid([torch.arange(size, dtype=torch.float32) for size in kernel_size])

        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * torch.exp(-((mgrid - mean) / (2 * std)) ** 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(1, channels, 1, 1, 1)

        return kernel
    
    def forward(self, xx):
        xx = xx.reshape(xx.shape[0]*2, 1, xx.shape[2], xx.shape[3], xx.shape[4])
        xx = F.conv3d(xx, self.kernel, padding = (self.kernel.shape[-1]-1)//2)
        return xx.reshape(xx.shape[0]//2, 2, xx.shape[2], xx.shape[3], xx.shape[4])
    
    
def blur_input(xx): 
    out = []
    for s in np.linspace(-1, 1, 5):
        if s > 0:
            blur = gaussain_blur(size = np.ceil(s), sigma = [s**2, s, s], dim  = 3, channels = 1).to(device)
            out.append(blur(xx).unsqueeze(1)*(s+1))
        elif s<0:
            out.append(xx.unsqueeze(1)*(1/(np.abs(s)+1)))
        else:
            out.append(xx.unsqueeze(1))
    out = torch.cat(out, dim = 1)
    return out

class Dataset_scale(data.Dataset):
    def __init__(self, indices, input_length, mid, output_length, direc):
        self.input_length = input_length
        self.mid = mid
        self.output_length = output_length
        self.direc = direc
        self.list_IDs = indices
        
    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        ID = self.list_IDs[index]
        x = torch.load(self.direc + str(ID) + ".pt")[(self.mid-self.input_length):self.mid].transpose(0,1)
        y = torch.load(self.direc + str(ID) + ".pt")[self.mid:(self.mid+self.output_length)].transpose(0,1)
        return x.float(), y.float()

# Training functions for scale equivariant models.
def train_epoch_scale(train_loader, model, optimizer, loss_function):
    train_mse = []
    for xx, yy in train_loader:
        xx = xx.to(device)
        yy = yy.to(device)
        loss = 0
        ims = []
        for i in range(yy.shape[2]):
            blur_xx = blur_input(xx)
            im = model(blur_xx)
            # print(xx.shape, im.shape)
            xx = torch.cat([xx[:, :, 1:], im.unsqueeze(2)], 2)
            loss += loss_function(im, yy[:,:,i])         
        train_mse.append(loss.item()/yy.shape[2]) 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_mse = round(np.sqrt(np.mean(train_mse)),5)
    return train_mse



def eval_epoch_scale(valid_loader, model, loss_function):
    valid_mse = []
    preds = []
    trues = []
    with torch.no_grad():
        for xx, yy in valid_loader:
            xx = xx.to(device)
            yy = yy.to(device)
            loss = 0
            ims = []
            for i in range(yy.shape[2]):
                blur_xx = blur_input(xx)
                im = model(blur_xx)
                xx = torch.cat([xx[:, :, 1:], im.unsqueeze(2)], 2)
                loss += loss_function(im, yy[:,:,i])
                ims.append(im.unsqueeze(2).cpu().data.numpy())
                
            valid_mse.append(loss.item()/yy.shape[2])    
            ims = np.concatenate(ims, axis = 2)
            preds.append(ims)
            trues.append(yy.cpu().data.numpy())
        try:
            preds = np.concatenate(preds, axis = 0)  
            trues = np.concatenate(trues, axis = 0)  
        except:
            print("can't concatenate")
        valid_mse = round(np.sqrt(np.mean(valid_mse)), 5)
    return valid_mse, preds, trues

def test_epoch_scale(valid_loader, model, loss_function):
    valid_mse = []
    preds = []
    trues = []
    with torch.no_grad():
        loss_curve = []
        for xx, yy in valid_loader:
            xx = xx.to(device)
            yy = yy.to(device)
            loss = 0
            ims = []
            
            for i in range(yy.shape[2]):
                im = model(xx)
                xx = torch.cat([xx[:, :, 1:], im.unsqueeze(2)], 2)
                mse = loss_function(im, yy[:,:,i])
                loss += mse
                loss_curve.append(mse.item())
                ims.append(im.unsqueeze(2).cpu().data.numpy())
                
            ims = np.concatenate(ims, axis = 2)
            preds.append(ims)
            trues.append(yy.cpu().data.numpy())
            valid_mse.append(loss.item()/yy.shape[1])
        loss_curve = np.array(loss_curve).reshape(-1,yy.shape[1])
        valid_mse = round(np.mean(valid_mse), 5)
        loss_curve = np.sqrt(np.mean(loss_curve, axis = 0))
    return valid_mse, preds, trues, loss_curve

