import os
import time
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
import torch.nn.functional as F
from model_vector import ResNet#, U_net
from train_vector import train_epoch, eval_epoch, test_epoch, Dataset, get_lr
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4,5,6,7"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

name = "ResNet_2"

train_direc = "/gpfs/wolf/gen138/proj-shared/deepcfd/data/Ocean_Data_DeepCFD/Data/train/sample_"
valid_direc = "/gpfs/wolf/gen138/proj-shared/deepcfd/data/Ocean_Data_DeepCFD/Data/valid/sample_"
test_direc_time = "/gpfs/wolf/gen138/proj-shared/deepcfd/data/Ocean_Data_DeepCFD/Data/test/sample_"
test_direc_domain = "/gpfs/wolf/gen138/proj-shared/deepcfd/data/Ocean_Data_DeepCFD/Data/test_domain/sample_"
train_indices = list(range(72)) 
valid_indices = list(range(16))
test_indices_time = list(range(16))
test_indices_domain = list(range(16))

test_direc = test_direc_time


batch_size    = 4
input_length  = 25
output_length = 3
learning_rate = 0.001


train_set = Dataset(train_indices, input_length, 40, output_length, train_direc)
valid_set = Dataset(valid_indices, input_length, 40, 6, train_direc)
train_loader = data.DataLoader(train_set, batch_size = batch_size, shuffle = True, num_workers = 8)
valid_loader = data.DataLoader(valid_set, batch_size = batch_size, shuffle = False, num_workers = 8)

print("Initializing...")
model = nn.DataParallel(ResNet(in_channels = input_length*2, out_channels = 2, kernel_size = 5).to(device))
print("Done")

optimizer = torch.optim.Adam(model.parameters(), learning_rate, betas=(0.9, 0.999), weight_decay=4e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size= 1, gamma=0.9)
loss_fun = torch.nn.MSELoss()


train_mse = []
valid_mse = []
test_mse = []

min_mse = 1000000

for i in range(1):
    print("A")
    start = time.time()
    scheduler.step()

    print("B")
    model.train()

    print("C")
    train_mse.append(train_epoch(train_loader, model, optimizer, loss_fun))

    print("D")
    model.eval()

    print("E")
    mse, _, _ = eval_epoch(valid_loader, model, loss_fun)

    print("F")
    valid_mse.append(mse)

    # if valid_mse[-1] < min_mse:
    #     min_mse = valid_mse[-1] 
    #     best_model = model
    #     torch.save(model, name + ".pth")
    # end = time.time()
    # if (len(train_mse) > 40 and np.mean(valid_mse[-5:]) >= np.mean(valid_mse[-10:-5])):
    #         break
    # print(i+1, train_mse[-1], valid_mse[-1], round((end-start)/60,5), format(get_lr(optimizer), "5.2e"), name)

# best_model = torch.load(name + ".pth")
# loss_fun = torch.nn.MSELoss()
# test_set = Dataset(test_indices_time, input_length, 40, 10, test_direc, True)
# test_loader = data.DataLoader(test_set, batch_size = batch_size, shuffle = False, num_workers = 8)
# rmse, preds, trues = eval_epoch(test_loader, best_model, loss_fun)
#
# torch.save({"preds": preds[::10],
#             "trues": trues[::10],
#             "rmse": rmse}, 
#             name +".pt")

