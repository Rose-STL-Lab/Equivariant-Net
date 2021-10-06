import os
import time
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
import torch.nn.functional as F
from models import ResNet, Unet, ResNet_UM, Unet_UM, ResNet_Mag, Unet_Mag, ResNet_Rot, Unet_Rot, ResNet_Scale, Unet_Scale
import matplotlib.pyplot as plt
from utils import train_epoch, eval_epoch, test_epoch, Dataset, get_lr, train_epoch_scale, eval_epoch_scale, test_epoch_scale, Dataset_scale
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


###### Hyperparameter ######
save_name = "ResNet_UM"
n_epochs = 1000
learning_rate = 0.001 # 0.0005 for mag_equ resnet; 0.0001 for scale_equ resnet
batch_size = 16
input_length = 24
train_output_length = 3 # 4 for all Unets
test_output_length = 10
lr_decay = 0.9
###########################

########## Data ###########
train_direc = ".../data_64/sample_"
valid_direc = ".../data_64/sample_"
test_direc = ".../data_64/sample_"
train_indices = list(range(0, 6000))
valid_indices = list(range(6000, 8000))
test_indices = list(range(8000, 10000))

train_set = Dataset(train_indices, input_length, 30, train_output_length, train_direc, True)
valid_set = Dataset(valid_indices, input_length, 30, train_output_length, valid_direc, True)
test_set = Dataset(test_indices, input_length, 30, test_output_length, test_direc, True)
# use Dataset_scale for scale equivariant models
# train_set = Dataset_scale(train_indices, input_length, 30, output_length, train_direc)
# valid_set = Dataset_scale(valid_indices, input_length, 30, output_length, train_direc)
# test_set = Dataset_scale(test_indices, input_length, 40, 10, test_direc)

train_loader = data.DataLoader(train_set, batch_size = batch_size, shuffle = True, num_workers = 8)
valid_loader = data.DataLoader(valid_set, batch_size = batch_size, shuffle = False, num_workers = 8)
test_loader = data.DataLoader(test_set, batch_size = batch_size, shuffle = False, num_workers = 8)
###########################

### Model ###
model = nn.DataParallel(ResNet_UM(input_channels = input_length*2, output_channels = 2, kernel_size = 3).to(device))
#model = nn.DataParallel(Unet_Rot(input_frames = input_length, output_frames = 1, kernel_size = 3, N = 8).to(device))
#model = nn.DataParallel(ResNet_Scale(input_channels = input_length*2, output_channels = 2, kernel_size = 3).to(device))

optimizer = torch.optim.Adam(model.parameters(), learning_rate,betas=(0.9, 0.999), weight_decay=4e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size= 1, gamma=lr_decay)
loss_fun = torch.nn.MSELoss()



min_mse = 100
train_mse = []
valid_mse = []
test_mse = []

for i in range(n_epochs):
    start = time.time()
    scheduler.step()

    model.train()
    # use train_epoch_scale/eval_epoch_scale for training scale equivariant models
    train_mse.append(train_epoch(train_loader, model, optimizer, loss_fun))
    model.eval()
    mse, _, _ = eval_epoch(valid_loader, model, loss_fun)
    valid_mse.append(mse)

    if valid_mse[-1] < min_mse:
        min_mse = valid_mse[-1] 
        best_model = model
        torch.save(best_model, save_name + ".pth")
    end = time.time()
    
    # Early Stopping but train at least for 50 epochs
    if (len(train_mse) > 50 and np.mean(valid_mse[-5:]) >= np.mean(valid_mse[-10:-5])):
            break
    print(i+1,train_mse[-1], valid_mse[-1], round((end-start)/60,5), format(get_lr(optimizer), "5.2e"))

test_mse, preds, trues, loss_curve = test_epoch(test_loader, best_model, loss_fun)
torch.save({"preds": preds,
            "trues": trues,
            "test_mse":test_mse,
            "loss_curve": loss_curve}, 
            name + ".pt")



