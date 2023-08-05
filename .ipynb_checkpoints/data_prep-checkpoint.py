import torch
import numpy as np
#from netCDF4 import Dataset
import os
import cv2
import random
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image
import torch.nn.functional as F
from netCDF4 import Dataset

################ Data Preprocessing ################
# read data
data = torch.load("rbc_data.pt")

# create folder for original data samples
orig_data_direc = "data_64/"
os.mkdir(orig_data_direc)

# standardization
std = torch.std(data)
avg = torch.mean(data)
data = (data - avg)/std
data = data[:,:,::4,::4]

# divide each rectangular snapshot into 7 subregions
# data_prep shape: num_subregions * time * channels * w * h
data_prep = torch.stack([data[:,:,:,k*64:(k+1)*64] for k in range(7)])

# use sliding windows to generate 10000 samples
# training 6000, validation 2000, test 2000
for j in range(0, 1500):
    for i in range(7):
        torch.save(data_prep[i, j : j + 50].double().float(), orig_data_direc + "sample_" + str(j*7+i) + ".pt")
        
        
        
################ Generate Transformed Test Sets ################
# Magnitude Transformation
mag_data_direc = "data_mag/"
os.mkdir(mag_data_direc)
for i in range(8000, 10000):
    # multiplied by random values sampled from U(0, 2);
    mag_transformed_img = torch.load(orig_data_direc + "sample_" + str(i) + ".pt") * torch.rand(1) * 2
    torch.save(mag_transformed_img, mag_data_direc + "sample_" + str(i) + ".pt")
    
# Uniform Motion Transformation
um_data_direc = "data_um/"
os.mkdir(um_data_direc)
for i in range(8000, 10000):
    # added random vectors drawn from U(−2, 2);
    um_transformed_img = torch.load(orig_data_direc + "sample_" + str(i) + ".pt") + (torch.rand(1, 2, 1, 1)*4-2)
    torch.save(um_transformed_img, um_data_direc + "sample_" + str(i) + ".pt")
    
# Rotation Transformation
def rotate_image(mat, angle):
    """
    Rotates an image (angle in degrees) and expands image to avoid cropping
    """

    height, width = mat.shape[:2] # image shape has 3 dimensions
    image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0,0]) 
    abs_sin = abs(rotation_mat[0,1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    return rotated_mat

def normalize(tensor):
    return (tensor - torch.min(tensor))/(torch.max(tensor) - torch.min(tensor))

def rotate(img, degree):
    #img shape 2*128*128
    #2*2 2*1*128*128 -> 2*1*128*128
    theta = torch.tensor(degree/180*np.pi)
    rot_m = torch.tensor([[torch.cos(theta), -torch.sin(theta)], [torch.sin(theta), torch.cos(theta)]])
    img = torch.einsum("ab, bcde -> acde",(rot_m, img.unsqueeze(1))).squeeze(1)
    
    mmin = torch.min(img)
    mmax = torch.max(img)
    img = normalize(img).data.numpy()
    x = TTen(TF.rotate(Image.fromarray(np.uint8(img[0]*255)),degree, expand = True))
    y = TTen(TF.rotate(Image.fromarray(np.uint8(img[1]*255)),degree, expand = True))
    rot_img = torch.cat([x, y], dim = 0)#)normalize(
    #print(np.max(img), np.min(img), torch.max(rot_img), torch.min(rot_img))
    rot_img[rot_img!=0] = rot_img[rot_img!=0]*(mmax - mmin) + mmin
    return rot_img

rot_data_direc = "data_rot/"
os.mkdir(rot_data_direc)
PIL = transforms.ToPILImage()
TTen = transforms.ToTensor()
for i in range(8000, 10000):
    degree = random.choice([90, 180, 270, 360])
    img = torch.load(orig_data_direc + "sample_" + str(i) + ".pt")
    rot_img = torch.cat([rotate(img[j], degree).unsqueeze(0) for j in range(img.shape[0])], dim = 0)
    torch.save(img,  rot_data_direc + "sample_" + str(i) + ".pt")
    
# Scale Transformation
scale_data_direc = "data_scale/"
os.mkdir(scale_data_direc)
for i in range(8000, 10000):
    img = torch.load(orig_data_direc + "sample_" + str(i) + ".pt")
    factor = (torch.rand(1)*9+1)/2
    scale_transformed_img = F.interpolate(img.transpose(0,1).unsqueeze(0), scale_factor = (factor**2, factor, factor), mode="trilinear", align_corners=None)[0,:,:100].transpose(0,1)/factor
    torch.save(scale_transformed_img, scale_data_direc + "sample_" + str(i) + ".pt")
    

    
############### Preprocess Ocean Data ##################
def load_nc(path):
    nc = Dataset(path)
    u0 = torch.from_numpy(np.array([nc["uo"][i].filled()[0] for i in range(len(nc["uo"]))])).float().unsqueeze(1)
    v0 = torch.from_numpy(np.array([nc["vo"][i].filled()[0] for i in range(len(nc["vo"]))])).float().unsqueeze(1)
    w = torch.cat([u0, v0], dim = 1)
    w[w<-1000] = 0
    w[w>10000] = 0
    return w


atlantic = load_nc("atlantic.nc")
indian = load_nc("indian.nc")
north_pacific = load_nc("north_pacific.nc")
south_pacific_test = load_nc("south_pacific_test.nc")


os.mkdir("ocean_train")
os.mkdir("ocean_test")

k = 0
for t in range(500):
    for i in range(3):
        for j in range(3):
            torch.save(atlantic[t:t+50,:,64*i:64*(i+1),64*j:64*(j+1)].double().float(), "ocean_train/sample_" + str(k) + ".pt")
            k += 1
            torch.save(indian[t:t+50,:,64*i:64*(i+1),64*j:64*(j+1)].double().float(), "ocean_train/sample_" + str(k) + ".pt")
            k += 1
            torch.save(north_pacific[t:t+50,:,64*i:64*(i+1),64*j:64*(j+1)].double().float().float(), "ocean_train/sample_" + str(k) + ".pt")
            k += 1
            
k = 0
for t in range(300):
    for i in range(3):
        for j in range(3):
            torch.save(south_pacific_test[t:t+50,:,64*i:64*(i+1),64*j:64*(j+1)].double().float(), "ocean_test/sample_" + str(k) + ".pt")
            k += 1