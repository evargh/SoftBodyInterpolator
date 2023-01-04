import os

import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms
from torchvision.io import read_image

import pandas as pd
import numpy as np
import gc

import cv2

T_LOWER = 50  # Lower Threshold
T_UPPER = 150  # Upper threshold

IM_HEIGHT = 270
IM_WIDTH = 480

TWO_SECOND_FRAME_COUNT = 24

class AnimationSet(Dataset):
    # gathering the list of animations
    def __init__(self, data_folder, data_folder_list):
        self.data_folder = data_folder
        with open(data_folder_list, 'r') as f:
            self.animations = f.read().splitlines()
        
    def __len__(self):
        return len(self.animations)
    

    # TODO: Log full first appearance of the ball by seeing when both "ends" are in camera view
    def __getitem__(self, idx):
        path = os.path.join(self.data_folder)
        animation_folder = self.animations[idx]
        #Access start frame, impact frame, and mass of ball, use .csv to access
        frame_csv_location = os.path.join(path, animation_folder, 'frames_list.csv')
        impact_csv_location = os.path.join(path, animation_folder, 'impact_frames.csv')
        settings_csv_location = os.path.join(path, animation_folder, 'animation_settings.csv')
        frames_location = os.path.join(path, animation_folder, 'frames')

        impact_file = pd.read_csv(impact_csv_location, header=None)
        frame_file = pd.read_csv(frame_csv_location, header=None)
        settings_file = pd.read_csv(settings_csv_location, header=None)

        mass_value = settings_file.iloc[0].values.tolist()[3]
        mass_layer = torch.full((1,IM_HEIGHT,IM_WIDTH), mass_value) #Resizing images for the sake of my RAM not dying input resize by 10

        first_impact = 0
        image_transforms = transforms.ToTensor()
        
        first_impact = impact_file.iloc[0].values.tolist()[0]
        
        # reading and edge detecting start frame
        start_frame_loc = os.path.join(path, frames_location, frame_file.iloc[0].values.tolist()[0])
        start_frame_img = cv2.imread(start_frame_loc)
        start_frame_edges = cv2.cvtColor(start_frame_img, cv2.COLOR_BGR2GRAY)
        start_frame_edges = cv2.resize(start_frame_edges, (IM_WIDTH, IM_HEIGHT))
        start_frame_edges = cv2.threshold(start_frame_edges, 50, 255, cv2.THRESH_BINARY)[1]
        #start_frame_edges = np.subtract(start_frame_edges, start_frame_edges.max())
        start_frame = image_transforms(start_frame_edges)

        # reading and edge detected end frame
        # need to find the position of the impact even with any deleted frames
        indexer_list = np.reshape(frame_file.iloc[:].values.tolist(), (len(frame_file.iloc[:].values.tolist())))
        filename = 'f{val}.png'.format(val=first_impact)
        realspot = indexer_list.tolist().index(filename)
        # get the index of the first impact and pass that to frame_file.iloc

        impact_frame_loc = os.path.join(path, frames_location, frame_file.iloc[realspot].values.tolist()[0])
        impact_frame_img = cv2.imread(impact_frame_loc)
        impact_frame_edges = cv2.cvtColor(impact_frame_img, cv2.COLOR_BGR2GRAY)
        impact_frame_edges = cv2.resize(impact_frame_edges, (IM_WIDTH, IM_HEIGHT))
        impact_frame_edges = cv2.threshold(impact_frame_edges, 50, 255, cv2.THRESH_BINARY)[1]
        #impact_frame_edges = np.subtract(impact_frame_edges, impact_frame_edges.max())*-1
        impact_frame = image_transforms(impact_frame_edges)

        image = torch.squeeze(torch.stack([start_frame, impact_frame, mass_layer], 1))

        intermediate_frame_count = first_impact - 2
        skip_value = intermediate_frame_count / TWO_SECOND_FRAME_COUNT

        firstpoint = int(frame_file.iloc[0].values.tolist()[0].split(".")[0][1:])

        # reading and edge detecting all frames in between
        # this tensor is also entirely booleans as a form of compression,
        # since edges are either true or false
        expected_frames = [0] * TWO_SECOND_FRAME_COUNT
        for i in range(0,TWO_SECOND_FRAME_COUNT):
            access_frame_idx = int(i*skip_value+.5+1 + firstpoint)
            filename = 'f{val}.png'.format(val=access_frame_idx)
            realspot = indexer_list.tolist().index(filename)

            middle_frame_loc = os.path.join(path, frames_location, frame_file.iloc[realspot].values.tolist()[0])
            middle_frame_img = cv2.imread(middle_frame_loc)
            middle_frame_edges = cv2.cvtColor(middle_frame_img, cv2.COLOR_BGR2GRAY)
            middle_frame_edges = cv2.resize(middle_frame_edges, (IM_WIDTH, IM_HEIGHT))
            middle_frame_edges = cv2.threshold(middle_frame_edges, 50, 255, cv2.THRESH_BINARY)[1]
            #print(type(middle_frame_edges))
            expected_frames[i] = image_transforms(middle_frame_edges)

        # this is all an example to show that, in the end, you can take the big tensor apart into a series of images
        # or heres hoping
        label = torch.squeeze(torch.stack(expected_frames, 1))
        labelnd = label.cpu().detach().numpy()
        testim = np.reshape(labelnd[0, :, :], (IM_HEIGHT, IM_WIDTH))
        cv2.imwrite('test.png', testim)
        
        #print(torch.Tensor.size(image), torch.Tensor.size(label))

        return image, label
    
class Encoder_Decoder(nn.Module):
    def __init__(self, batch, encoded_space_dim):
        super().__init__()

        ### Convolutional section
        self.encoder_cnn = nn.Sequential(
            nn.Conv3d(batch, 16, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv3d(16, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU(True),
            nn.Conv3d(32, 64, (1, 3, 3), stride=2, padding=0),
            nn.ReLU(True)
        )

        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)
        ### Linear section
        self.encoder_lin = nn.Sequential(
            nn.Linear(33*59, 128),
            nn.ReLU(True),
            nn.Linear(128, encoded_space_dim)
        )

        self.decoder_lin = nn.Sequential(
            nn.Linear(encoded_space_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 33*59),
            nn.ReLU(True)
        )

        self.unflatten = nn.Unflatten(dim=1,
        unflattened_size=(1, 33, 59))

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose3d(64, 32, (5, 3, 3),
            stride=2, output_padding=(1, 0, 1)),
            nn.BatchNorm2d(6),
            nn.ReLU(True),
            nn.ConvTranspose3d(32, 16, 3, stride=2,
            padding=1, output_padding=(0, 1, 0)),
            nn.BatchNorm2d(11),
            nn.ReLU(True),
            nn.ConvTranspose3d(16, batch, 3, stride=2,
            padding=0, output_padding=1)
        )

    def forward(self, x):
        acts = []

        x = self.encoder_cnn(x)
        acts.append(x)
        x = self.flatten(x)
        acts.append(x)
        x = self.encoder_lin(x)
        acts.append(x)

        x = x + acts[2]
        x = self.decoder_lin(x)
        x = x + acts[1]
        x = self.unflatten(x)
        x = x + acts[0]
        x = self.decoder_conv(x)
        return x

ls = os.path.join(os.getcwd(),'animations')

#Trains on 64 animation folders at a time
batch_size = 16

#Iterate over every folder in the working directory ls
#Store all animation sets in animations array

animation = AnimationSet(
        data_folder = ls, #temporary, replace with actual working directory
        data_folder_list = os.path.join(ls,'animation_list.csv')#Contains the list of animation folders
    )

m=int(len(animation)*.8)
train_set, test_set = torch.utils.data.random_split(animation, [m, len(animation)-m])

train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=True, drop_last = True)

encoder_decoder = Encoder_Decoder(batch=batch_size, encoded_space_dim=32)
params_to_optimize = [
    {'params': encoder_decoder.parameters()},
]

optim = torch.optim.Adam(params_to_optimize, lr=0.001, weight_decay=1e-05)

device = "cuda" if torch.cuda.is_available() else "cpu"
encoder_decoder.to(device)

def train_epoch(encoder, device, dataloader, loss_fn, optimizer):
    # Set train mode for both the encoder and the decoder
    encoder.train()
    train_loss = []

    for image_batch, label_batch in dataloader:
        image_batch = image_batch.to(device)
        encoded_decoded_data = encoder(image_batch)
        loss = loss_fn(encoded_decoded_data, label_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('\t partial train loss (single batch): %f' % (loss.data))
        train_loss.append(loss.detach().cpu().numpy())

    return np.mean(train_loss)

### Testing function
def test_epoch(encoder, device, dataloader, loss_fn):
    # Set evaluation mode for encoder and decoder
    encoder.eval()
    with torch.no_grad():
        conc_out = []
        conc_label = []
        for image_batch, label_batch in dataloader:
            image_batch = image_batch.to(device)
            encoded_decoded_data = encoder(image_batch)

            conc_out.append(encoded_decoded_data.cpu())
            conc_label.append(label_batch.cpu())

            # TODO: write each frame to a file
            outputnd = encoded_decoded_data.cpu().detach().numpy()
            labelnd = label_batch.cpu().detach().numpy()
            outputtest = (np.reshape(outputnd[0, 12, :, :], (IM_HEIGHT, IM_WIDTH))*255).astype(int)
            labeltest = (np.reshape(labelnd[0, 12, :, :], (IM_HEIGHT, IM_WIDTH))*255).astype(int)
            cv2.imwrite('output.png', outputtest)
            cv2.imwrite('label.png', labeltest)

        conc_out = torch.cat(conc_out)
        conc_label = torch.cat(conc_label) 

        val_loss = loss_fn(conc_out, conc_label)
    return val_loss.data
    
loss_fn = torch.nn.L1Loss()

num_epochs = 10
diz_loss = {'train_loss':[],'val_loss':[]}
for epoch in range(num_epochs):
    train_loss = train_epoch(encoder_decoder,device,train_dataloader,loss_fn,optim)
    val_loss = test_epoch(encoder_decoder,device,test_dataloader,loss_fn)
    print('\n EPOCH {}/{} \t train loss {} \t val loss {}'.format(epoch + 1, num_epochs,train_loss,val_loss))
    diz_loss['train_loss'].append(train_loss)
    diz_loss['val_loss'].append(val_loss)

    