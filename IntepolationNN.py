import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torchvision import datasets
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

import os
import pandas as pd
import numpy as np
from torchvision.io import read_image
from torchvision.transforms import ToTensor

# class AnimationSet(Dataset):
#     def __init__(self, data_folder, img_folder, planar_location, threeD_location, 
#                  impact_frames, planar_size):
#         self.data_folder = data_folder
#         self.img_fold = img_folder
#         self.planar_location = planar_location
#         self.threeD_location = threeD_location
#         self.impact_frames = impact_frames
#         self.planar_size = planar_size
        
#     def __len__(self):
#         return len(self.planar_location)
        
#     def __getitem__(self, idx):
#         imgPath = os.path.join(self.data_folder, self.img_folder, self.
#                                )
#         image = read_image(imgPath)
#         label = self.planar_location.iloc[idx, 1]
#         if self.transform:
#             image = self.transform(image)
#         if self.target_transform:
#             label = self.target_transform(label)
#         return image, label

#Gets only the data_folder for the dataset, getItem does all the heavy lifting
class AnimationSet(Dataset):
    def __init__(self, data_folder, data_folder_list):
        self.data_folder = data_folder
        with open(data_folder_list, 'r') as f:
            self.animations = f.read().splitlines()
        
    def __len__(self):
        return len(self.animations)
        
    def __getitem__(self, idx):
        path = os.path.join(self.data_folder)
        animation_folder = self.animations[idx]
        #Access start frame, impact frame, and mass of ball
        start_frame = ToTensor(read_image(transforms.Resize(160, 90)(path+animation_folder+'\image_folder.csv')))
        impact_frame = ToTensor(read_image(transforms.Resize(160, 90)(path+animation_folder+'\impact_frames.csv')))
        mass = torch.full((160,90),path+animation_folder+'\animation_settings.csv')
        expected_frames = pd.read_csv(path+animation_folder+'\image_folder.csv')
        
        return impact_frame, start_frame, mass, expected_frames


#Defining sets
print(os.getcwd())
#C:\Users\Alex\Desktop\ML Proj
ls = os.getcwd()
#Trains on 64 animation folders at a time
batch_size = 64

#Iterate over every folder in the working directory ls
#Store all animation sets in animations array

animation = AnimationSet(
        data_folder = ls, #temporary, replace with actual working directory
        data_folder_list = ls+'\animation_folders.csv'#Contains the list of animation folders
    )
    

train_dataloader = DataLoader(animation, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(animation, batch_size=batch_size, shuffle=False)

device = "cuda" if torch.cuda.is_available() else "cpu"

#Note to self, us tanh function instead of ReLu to prevent explosion to inf

class CNNModelConv(nn.Module):
    def __init__(self):
        super(CNNModelConv, self).__init__()
        self.conv1 = self.sequential_set(3, 576) #576 = 30*18 kernel_size divided!
        
        
    def sequential_set(self, in_c, out_c):
        conv = nn.Sequential(
            nn.Conv3d(in_channels=in_c, out_channels=out_c, kernel_size= (5, 5, 3),padding=1, padding_mode='zeros'),
            nn.Tanh(),
            nn.MaxPool3d(3)
        )
        return conv  

    def forward(self, x):
        output = []
        for i in 150
            output = ?


        return output
    
def trainer():
    for idx, #
return stuff

def validation():
    for idx, #
return stuff


