import os
import time
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
import torch.nn.functional as F
from model_um import ResNet_UM, Unet_UM
from utils import train_epoch, eval_epoch, test_epoch, Dataset, get_lr
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

name = "resnet_um"
train_direc = ".../sample_"
test_direc = ".../sample_"


learning_rate = 0.001
min_mse = 1
output_length = 3
batch_size = 16
input_length = 25
train_indices = list(range(0, 7000))
valid_indices = list(range(6000, 7000))
test_indices = list(range(7000, 8000))

train_set = Dataset(train_indices, input_length, 40, output_length, train_direc, True)
valid_set = Dataset(valid_indices, input_length, 40, output_length, train_direc, True)
test_set = Dataset(test_indices, input_length, 40, 10, test_direc, True)

test_loader = data.DataLoader(test_set, batch_size = batch_size, shuffle = False, num_workers = 8)
train_loader = data.DataLoader(train_set, batch_size = batch_size, shuffle = True, num_workers = 8)
valid_loader = data.DataLoader(valid_set, batch_size = batch_size, shuffle = False, num_workers = 8)

print("Initializing...")
model = nn.DataParallel(ResNet(input_channels = input_length*2, output_channels = 2, kernel_size = 3).to(device))
print("Done")

optimizer = torch.optim.Adam(model.parameters(), learning_rate,betas=(0.9, 0.999), weight_decay=4e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size= 1, gamma=0.9)
loss_fun = torch.nn.MSELoss()


train_mse = []
valid_mse = []
test_mse = []

for i in range(100):
    start = time.time()
    scheduler.step()

    model.train()
    train_mse.append(train_epoch(train_loader, model, optimizer, loss_fun))
    model.eval()
    mse, _, _ = eval_epoch(valid_loader, model, loss_fun)
    valid_mse.append(mse)

    if valid_mse[-1] < min_mse:
        min_mse = valid_mse[-1] 
        best_model = model
        torch.save(best_model, name + ".pth")
    end = time.time()
    if (len(train_mse) > 50 and np.mean(valid_mse[-5:]) >= np.mean(valid_mse[-10:-5])):
            break
    print(i+1,train_mse[-1], valid_mse[-1], round((end-start)/60,5), format(get_lr(optimizer), "5.2e"), name)



best_model = torch.load(name + ".pth")
loss_fun = torch.nn.MSELoss()
test_mse, preds, trues, loss_curve = test_epoch(test_loader, best_model, loss_fun)

torch.save({"preds": preds,
            "trues": trues,
            "test_mse":test_mse,
            "loss_curve": loss_curve}, 
            name + ".pt")
