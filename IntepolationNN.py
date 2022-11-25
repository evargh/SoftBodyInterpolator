import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import os
import pandas as pd # data processing with .csv
import numpy as np # linear algebra if necessary?
import gc
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
        #Access start frame, impact frame, and mass of ball, use .csv to access
        start_frame = ToTensor(read_image(transforms.Resize(160, 90)(path+animation_folder+'\image_folder.csv')))
        impact_frame = ToTensor(read_image(transforms.Resize(160, 90)(path+animation_folder+'\impact_frames.csv')))
        mass = torch.full((160,90),path+animation_folder+'\animation_settings.csv')
        expected_frames = pd.read_csv(path+animation_folder+'\image_folder.csv')
        #Note, turn expected_frames into a matrix.
        
        return impact_frame, start_frame, mass, expected_frames
    
class CNNModelConv(nn.Module):
    def __init__(self):
        super(CNNModelConv, self).__init__()
        self.conv1 = self.sequential_set(3, 576) #576 = 30*18 kernel_size divided!
        self.conv2 = self.sequential_set(576, 1152) 
        self.conv3 = self.sequential_set(1152, 2304)
        self.ups1 = self.upsample_set(2304, 1152)
        self.ups2= self.upsample_set(1152, 576)
        self.ups3 = self.upsample_set(576, 150)
        self.fc1 = nn.Linear(2304, 150)
        self.fc2 = nn.Linear(150, 1)
        
        
    def sequential_set(self, in_c, out_c):
        conv = nn.Sequential(
            nn.Conv3d(in_channels=in_c, out_channels=out_c, kernel_size= (5, 5, 3),padding=1, padding_mode='zeros'),
            nn.BatchNorm3d(out_c),
            nn.Tanh(),
            nn.MaxPool3d(kernel_size=3,padding=1, padding_mode='zeros')
        )
        return conv 
    
    def upsample_set(self, in_c, out_c):
        conv = nn.Sequential(
            nn.Conv3d(in_channels=in_c, out_channels=out_c, kernel_size= (5, 5, 3),padding=1, padding_mode='zeros'),
            nn.BatchNorm3d(out_c),
            nn.Tanh(),
            nn.MaxUnpool3d(kernel_size=3,padding=1, padding_mode='zeros')
        )
        return conv   

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.fc1(x) # Double linear regression
        x = self.fc2(x)
        x = self.ups1(x)
        x = self.ups2(x) 
        x = self.ups2(x)       
        
        return x


#Defining sets
print(os.getcwd())
#C:\Users\Alex\Desktop\ML Proj
ls = os.getcwd()+'\Animations'

#Trains on 64 animation folders at a time
batch_size = 64

#Iterate over every folder in the working directory ls
#Store all animation sets in animations array

animation = AnimationSet(
        data_folder = ls, #temporary, replace with actual working directory
        data_folder_list = ls+'\animation_list.csv'#Contains the list of animation folders
    )
    
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CNNModelConv().to(device)
optimizer = torch.optim.RMSprop(model.parameters, lr=1e-5)
criterion = nn.MSELoss()

train_dataloader = DataLoader(animation, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(animation, batch_size=batch_size, shuffle=False)
    
def trainer():
    #Default Trainer
    model.train()
    
    for epoch in range(500):
        running_loss = 0.0
        for i, data in enumerate(train_dataloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
    
            # zero the parameter gradients
            optimizer.zero_grad()
    
            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
            running_loss += loss.item()
            gc.collect() #Garbage Collection
    
        print('Training Loss: {}'.format(running_loss))
    print('Finished Training')

def validation():
    
    model.eval()
    with torch.no_grad():
        for epoch in range(500):
            running_loss = 0.0
            for i, data in enumerate(test_dataloader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
        
                # zero the parameter gradients
                optimizer.zero_grad()
        
                # forward + backward + optimize
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        
                running_loss += loss.item()
                gc.collect() #Garbage Collection
            print('Validation Loss: {}'.format(running_loss))
            
    print('Finished Validating')
    


