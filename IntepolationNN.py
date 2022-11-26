import os

import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import ToTensor
from torchvision.io import read_image

import pandas as pd # data processing with .csv
import numpy as np # linear algebra if necessary?
import gc

import cv2

T_LOWER = 50  # Lower Threshold
T_UPPER = 150  # Upper threshold

class AnimationSet(Dataset):
    # gathering the list of animations
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
        frame_csv_location = os.path.join(path, animation_folder, 'frames_list.csv')
        impact_csv_location = os.path.join(path, animation_folder, 'impact_frames.csv')
        settings_csv_location = os.path.join(path, animation_folder, 'animation_settings.csv')

        impact_file = pd.read_csv(impact_csv_location, header=None)
        frame_file = pd.read_csv(frame_csv_location, header=None)
        settings_file = pd.read_csv(settings_csv_location, header=None)

        mass_value = settings_file.iloc[0].values.tolist()[3]
        mass_layer = torch.full((1,540,960), mass_value)

        first_impact = 0
        image_transforms = transforms.ToTensor()
        
        first_impact = impact_file.iloc[0].values.tolist()[0]
        
        # reading and edge detecting start frame
        start_frame_img = cv2.imread(os.path.join(path, animation_folder, frame_file.iloc[0].values.tolist()[0]))
        start_frame_edges = cv2.Canny(start_frame_img, T_LOWER, T_UPPER)
        start_frame = image_transforms(start_frame_edges)

        # reading and edge detected end frame
        impact_frame_img = cv2.imread(os.path.join(path, animation_folder, frame_file.iloc[first_impact].values.tolist()[0]))
        impact_frame_edges = cv2.Canny(impact_frame_img, T_LOWER, T_UPPER)
        impact_frame = image_transforms(impact_frame_edges)

        image = torch.stack([start_frame, impact_frame, mass_layer], 0)

        # reading and edge detecting all frames in between
        expected_frames = [0] * (first_impact-2)
        for i in range(0,first_impact-2):
            middle_frame = cv2.imread(os.path.join(path, animation_folder, frame_file.iloc[i+1].values.tolist()[0]))
            middle_frame_edges = cv2.Canny(middle_frame, T_LOWER, T_UPPER)
            expected_frames[i] = image_transforms(middle_frame_edges)

        # this is all an example to show that, in the end, you can take the big tensor apart into a series of images
        # or heres hoping
        label = torch.stack(expected_frames)
        # labelnd = label.cpu().detach().numpy()
        # testim = np.reshape(labelnd[0, :, :], (540, 960, 1))*255
        # cv2.imwrite('test.png', testim)

        return image, label
    
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
            nn.MaxPool3d(kernel_size=3,padding=1)
        )
        return conv 
    
    def upsample_set(self, in_c, out_c):
        conv = nn.Sequential(
            nn.Conv3d(in_channels=in_c, out_channels=out_c, kernel_size= (5, 5, 3),padding=1, padding_mode='zeros'),
            nn.BatchNorm3d(out_c),
            nn.Tanh(),
            nn.MaxUnpool3d(kernel_size=3,padding=1)
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
ls = os.path.join(os.getcwd(),'animations')

#Trains on 64 animation folders at a time
batch_size = 64

#Iterate over every folder in the working directory ls
#Store all animation sets in animations array

animation = AnimationSet(
        data_folder = ls, #temporary, replace with actual working directory
        data_folder_list = os.path.join(ls,'animation_list.csv')#Contains the list of animation folders
    )

train_dataloader = DataLoader(animation, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(animation, batch_size=batch_size, shuffle=False)
    
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
    


