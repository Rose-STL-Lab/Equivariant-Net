import torch
import torch.nn as nn
import numpy as np
from torch.utils import data
from torch.autograd import Variable
import kornia
import radialProfile
import os

def TKE(preds):
    mean_flow = np.expand_dims(np.mean(preds, axis = 0), axis = 0)
    tur_preds = np.mean((preds - mean_flow)**2, axis = 0)
    tke = (tur_preds[0] + tur_preds[1])/2
    return tke

def TKE_mean(preds):
    accum = []
    for i in range(len(preds)):
        mean_flow = np.expand_dims(np.mean(preds[i], axis = 0), axis = 0)
        tur_preds = np.mean((preds[i] - mean_flow)**2, axis = 0)
        tke = (tur_preds[0] + tur_preds[1])/2
        accum.append(tke)
    return np.mean(np.array(accum), axis = 0)

def tke2spectrum(tke):
    """Convert TKE field to spectrum"""
    sp = np.fft.fft2(tke)
    sp = np.fft.fftshift(sp)
    sp = np.real(sp*np.conjugate(sp))
    sp1D = radialProfile.azimuthalAverage(sp)
    return np.log10(sp1D)
 
def spectrum_band(tensor):
    spec = np.array([tke2spectrum(TKE(tensor[i])) for i in range(tensor.shape[0])])
    return np.mean(spec, axis = 0)



spec_rmse = []
rmse = []
for i in range(1, 6):
    direc = "/global/cscratch1/sd/rwang2/Ocean/Scale/ResNet_"
    file = torch.load(direc + str(i) +"_time.pt")
    preds = file["preds"]
    trues = file["trues"]
    spec_rmse.append(np.sqrt(np.mean((spectrum_band(preds) - spectrum_band(trues))**2)))
    rmse.append(file["rmse"])
print(np.mean(rmse), np.mean(spec_rmse))
