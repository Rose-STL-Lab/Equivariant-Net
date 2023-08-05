import os
import time
import argparse
import torch
import numpy as np
import torch.nn as nn
import random
import torch.optim as optim
from torch.utils import data
import torch.nn.functional as F
from evaluation import spectrum_band
from models import ResNet, Unet, ResNet_UM, Unet_UM, ResNet_Mag, Unet_Mag, ResNet_Rot, Unet_Rot, ResNet_Scale, Unet_Scale
import matplotlib.pyplot as plt
from utils import train_epoch, eval_epoch, test_epoch, Dataset, get_lr, train_epoch_scale, eval_epoch_scale, test_epoch_scale, Dataset_scale
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


###### Hyperparameter ######

parser = argparse.ArgumentParser(description='Deep Equivariant Dynamics Models')
parser.add_argument('--dataset', type=str, required=False, default="RBC", help='RBC or Ocean')
parser.add_argument('--kernel_size', type=int, required=False, default="3", help='convolution kernel size')
parser.add_argument('--symmetry', type=str, required=False, default="UM", help='None, UM, Rot, Mag, Scale')
parser.add_argument('--architecture', type=str, required=False, default="ResNet", help='ResNet or Unet')
parser.add_argument('--output_length', type=int, required=False, default="4", help='number of prediction losses used for backpropagation')
parser.add_argument('--input_length', type=int, required=False, default="24", help='input length')
parser.add_argument('--batch_size', type=int, required=False, default="16", help='batch size')
parser.add_argument('--num_epoch', type=int, required=False, default="100", help='maximum number of epochs')
parser.add_argument('--learning_rate', type=float, required=False, default="0.001", help='learning rate')
parser.add_argument('--decay_rate', type=float, required=False, default="0.95", help='learning decay rate')
parser.add_argument('--seed', type=int, required=False, default="0", help='random seed')
args = parser.parse_args()


random.seed(args.seed)  # python random generator
np.random.seed(args.seed)  # numpy random generator

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


symmetry = args.symmetry
model_name = args.architecture + "_" + args.symmetry
num_epoch = args.num_epoch
learning_rate = args.learning_rate # 0.0005 for mag_equ resnet; 0.0001 for scale_equ resnet
batch_size = args.batch_size
input_length = args.input_length
train_output_length = args.output_length # 4 for all Unets
test_output_length = 10
kernel_size = args.kernel_size
lr_decay = args.decay_rate
###########################

########## Data ###########
if args.dataset == "RBC":
    train_direc = "data_64/sample_"
    valid_direc = "data_64/sample_"
    train_indices = list(range(0, 6000))
    valid_indices = list(range(6000, 8000))

    # test on future time steps
    test_future_direc = "data_64/sample_"
    test_future_indices = list(range(8000, 10000)) 

    # test on data applied with symmetry transformations 
    test_domain_direc = "data_64/sample_" if args.symmetry == "None" else "data_" + symmetry.lower() + "/sample_" 
    print(test_domain_direc)
    test_domain_indices = list(range(8000, 10000)) 
    
elif args.dataset == "Ocean":
    train_direc = "ocean_train/sample_"
    valid_direc = "ocean_train/sample_"
    train_indices = list(range(0, 8000))
    valid_indices = list(range(8000, 10000))

    # test on future time steps
    test_future_direc = "ocean_train/sample_"
    test_future_indices = list(range(10000, 12000)) 

    # test on data from different domain
    test_domain_direc = "ocean_test/sample_"
    test_domain_indices = list(range(0, 2000)) 
    
else:
    print("Invalid dataset name entered!")

if symmetry != "Scale":
    train_set = Dataset(train_indices, input_length, 30, train_output_length, train_direc, True)
    valid_set = Dataset(valid_indices, input_length, 30, train_output_length, valid_direc, True)
    test_future_set = Dataset(test_future_indices, input_length, 30, test_output_length, test_future_direc, True)
    test_domain_set = Dataset(test_domain_indices, input_length, 30, test_output_length, test_domain_direc, True)
else:
    # use Dataset_scale for scale equivariant models
    train_set = Dataset_scale(train_indices, input_length, 30, train_output_length, train_direc)
    valid_set = Dataset_scale(valid_indices, input_length, 30, train_output_length, train_direc)
    test_future_set = Dataset_scale(test_future_indices, input_length, 30, test_output_length, test_future_direc)
    test_domain_set = Dataset_scale(test_domain_indices, input_length, 30, test_output_length, test_domain_direc)

train_loader = data.DataLoader(train_set, batch_size = batch_size, shuffle = True, num_workers = 8, pin_memory=True)
valid_loader = data.DataLoader(valid_set, batch_size = batch_size, shuffle = False, num_workers = 8, pin_memory=True)
test_future_loader = data.DataLoader(test_future_set, batch_size = batch_size, shuffle = False, num_workers = 8, pin_memory=True)
test_domain_loader = data.DataLoader(test_domain_set, batch_size = batch_size, shuffle = False, num_workers = 8, pin_memory=True)



save_name = args.dataset + "_model{}_bz{}_inp{}_pred{}_lr{}_decay{}_kernel{}_seed{}".format(model_name,
                                                                                            batch_size,
                                                                                            input_length,
                                                                                            train_output_length,
                                                                                            learning_rate,
                                                                                            lr_decay,
                                                                                            kernel_size, 
                                                                                            args.seed)
                                                                                     
print(save_name)
####### Select Model #######
if model_name == "ResNet_UM":
    model = nn.DataParallel(ResNet_UM(input_channels = input_length*2, output_channels = 2, kernel_size = kernel_size).to(device))
elif model_name == "Unet_UM":
    model = nn.DataParallel(Unet_UM(input_channels = input_length*2, output_channels = 2, kernel_size = kernel_size).to(device))
elif model_name == "ResNet_Rot":
    model = nn.DataParallel(ResNet_Rot(input_frames = input_length, output_frames = 1, kernel_size = kernel_size, N = 8).to(device))
elif model_name == "Unet_Rot":
    model = nn.DataParallel(Unet_Rot(input_frames = input_length, output_frames = 1, kernel_size = kernel_size, N = 8).to(device))
elif model_name == "ResNet_Mag":
    model = nn.DataParallel(ResNet_Mag(input_channels = input_length*2, output_channels = 2, kernel_size = kernel_size).to(device))
elif model_name == "Unet_Mag":  
    model = nn.DataParallel(Unet_Mag(input_channels = input_length*2, output_channels = 2, kernel_size = kernel_size).to(device))
elif model_name == "ResNet_Scale":
    model = nn.DataParallel(ResNet_Scale(input_channels = input_length*2, output_channels = 2, kernel_size = kernel_size).to(device))
elif model_name == "Unet_Scale":  
    model = nn.DataParallel(Unet_Scale(input_channels = input_length*2, output_channels = 2, kernel_size = kernel_size).to(device))
elif model_name == "ResNet_None":
    model = nn.DataParallel(ResNet(input_channels = input_length*2, output_channels = 2, kernel_size = kernel_size).to(device))
elif model_name == "Unet_None":
    model = nn.DataParallel(Unet(input_channels = input_length*2, output_channels = 2, kernel_size = kernel_size).to(device))
else:
    print("Invalid model name entered!")


optimizer = torch.optim.Adam(model.parameters(), learning_rate,betas=(0.9, 0.999), weight_decay=4e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=lr_decay)
loss_fun = torch.nn.MSELoss()

min_rmse = 1e6
train_rmse = []
valid_rmse = []
test_rmse = []

for i in range(num_epoch):
    start = time.time()
    
    if symmetry != "Scale":
        model.train()
        train_rmse.append(train_epoch(train_loader, model, optimizer, loss_fun))
        model.eval()
        rmse, _, _ = eval_epoch(valid_loader, model, loss_fun)
        valid_rmse.append(rmse)
    else:
        model.train()
        train_rmse.append(train_epoch_scale(train_loader, model, optimizer, loss_fun))
        model.eval()
        rmse, _, _ = eval_epoch_scale(valid_loader, model, loss_fun)
        valid_rmse.append(rmse)

    if valid_rmse[-1] < min_rmse:
        min_rmse = valid_rmse[-1] 
        best_model = model
    end = time.time()
    
    # Early Stopping but train at least for 50 epochs
    if (len(train_rmse) > 50 and np.mean(valid_rmse[-5:]) >= np.mean(valid_rmse[-10:-5])):
            break
    print("Epoch {} | T: {:0.2f} | Train RMSE: {:0.3f} | Valid RMSE: {:0.3f}".format(i + 1, (end-start) / 60, train_rmse[-1], valid_rmse[-1]))
    scheduler.step()
    
    
if symmetry != "Scale":
    test_future_rmse, test_future_preds, test_future_trues, test_future_loss_curve = test_epoch(test_future_loader, best_model, loss_fun)
    test_domain_rmse, test_domain_preds, test_domain_trues, test_domain_loss_curve = test_epoch(test_domain_loader, best_model, loss_fun)
else:
    test_future_rmse, test_future_preds, test_future_trues, test_future_loss_curve = test_epoch_scale(test_future_loader, best_model, loss_fun)
    test_domain_rmse, test_domain_preds, test_domain_trues, test_domain_loss_curve = test_epoch_scale(test_domain_loader, best_model, loss_fun)

# Compute Energy Spectrum Errors
test_future_ese = np.sqrt(np.mean((spectrum_band(test_future_preds) - spectrum_band(test_future_trues))**2))
test_domain_ese = np.sqrt(np.mean((spectrum_band(test_domain_preds) - spectrum_band(test_domain_trues))**2))
print("Model: {} | Symmetry: {} | Future RMSE: {:0.3f} | Future ESE: {:0.3f} | Domain RMSE: {:0.3f} | Domain ESE: {:0.3f} ".format(args.architecture, 
                                                                                                                                   args.symmetry, 
                                                                                                                                   test_future_rmse, 
                                                                                                                                   test_future_ese, 
                                                                                                                                   test_domain_rmse, 
                                                                                                                                   test_domain_ese))

torch.save({"test_future": [test_future_rmse, test_future_ese, test_future_preds[::10], test_future_trues[::10]],
            "test_domain": [test_domain_rmse, test_domain_ese, test_domain_preds[::10], test_domain_trues[::10]]}, 
            save_name + ".pt")



