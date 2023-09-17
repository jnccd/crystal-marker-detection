import os
import glob
import cv2
import numpy as np
import ast
import scipy
import tqdm
from datetime import datetime
from geomloss import SamplesLoss

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models

from utils import *

LR = 1e-3
EPOCHS = 5
BATCH_SIZE = 32
DEVICE = "cuda"
DIM_KEYPOINTS = 2
NUM_KEYPOINTS = 4
IMG_SIZE = 224
RESIZE_ON_LOAD = True

root_dir = Path(__file__).resolve().parent
dataset_dir = root_dir/'..'/'traindata-creator/dataset/pet-0-pet-test-red-rects'
dataset_train_dir = dataset_dir / 'train'
dataset_val_dir = dataset_dir / 'val'
output_folder = create_dir_if_not_exists(root_dir / 'output/mbn')

# --- Dataloader ----------------------------------------------------------------------------------------

class DataseriesLoader(Dataset):
    def __init__(self, dataseries_dir):
        # Read filenames
        self.image_filenames = get_files_from_folders_with_ending([dataseries_dir], '.png')
        self.label_filenames = get_files_from_folders_with_ending([dataseries_dir], '.txt')
        self.image_label_filenames = list(zip(self.image_filenames, self.label_filenames))

        print(f"Found {len(self.image_label_filenames)} images in {dataseries_dir}")

    def __len__(self):
        return len(self.image_label_filenames)
    
    def __getitem__(self, idx):
        # Read image
        image = cv2.imread(self.image_label_filenames[idx][0])
        if RESIZE_ON_LOAD:
            image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2,0,1)

        # Read label
        label = torch.from_numpy(
            np.array(ast.literal_eval(read_textfile(self.image_label_filenames[idx][1])))
        ).float()

        return image, label
    
# --- Model ----------------------------------------------------------------------------------------

class SimpleCNN(torch.nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        self.maxpool = torch.nn.MaxPool2d(2,2)
        self.dropout = torch.nn.Dropout2d(0.2)
        self.conv1 = torch.nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = torch.nn.Conv2d(64, 128, 3, padding=1)
        self.fc1 = torch.nn.Linear(128 * 16 * 16, 8)

    def forward(self, x):
        # Convolutional layers
        x = self.conv1(x)
        x = torch.nn.functional.relu(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        x = torch.nn.functional.relu(x)
        x = self.maxpool(x)

        x = self.conv3(x)
        x = torch.nn.functional.relu(x)
        x = self.maxpool(x)

        # Flatten
        x = x.view(-1, 128 * 16 * 16)

        # Fully connected
        x = self.fc1(x)
        
        # Reshape to (batch_size, 4, 2)
        x = x.view(-1, 4, 2)

        return x

vgg16 = models.vgg16(pretrained=True)
vgg16 = vgg16.features
class CustomVGG16Head(nn.Module):
    def __init__(self, num_keypoints):
        super(CustomVGG16Head, self).__init__()
        self.conv1 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(256 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_keypoints)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
custom_head = CustomVGG16Head(NUM_KEYPOINTS * DIM_KEYPOINTS)
custom_vgg16 = nn.Sequential(vgg16, custom_head)

# --- Loss ----------------------------------------------------------------------------------------

def loss_mse(label_pred, label_true):
    return torch.nn.functional.mse_loss(label_pred, label_true)

def loss_repulsion(label_pred, label_true):
    # Compute pairwise distances between all predicted corners
    return 1. / torch.min(torch.cdist(label_pred, label_true, p=2.0))

def hungarian_loss(pred, target):
    assert pred.shape[0] == target.shape[0], "Batch sizes do not match"

    # Initialize loss
    loss = 0.0

    # Loop over batch
    for i in range(pred.shape[0]):
        # Compute pairwise distances for current example
        pairwise_dist = torch.cdist(pred[i], target[i])

        # Convert to numpy for scipy compatibility
        pairwise_dist_np = pairwise_dist.detach().cpu().numpy()

        # Compute optimal assignment
        row_ind, col_ind = scipy.optimize.linear_sum_assignment(pairwise_dist_np)

        # Add to loss
        loss += pairwise_dist[row_ind, col_ind].sum()

    # Average loss over batch
    loss /= pred.shape[0]

    return loss

# --- Train ----------------------------------------------------------------------------------------

# define dataloaders
train_loader = DataLoader(
    DataseriesLoader(dataset_train_dir), 
    batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(
    DataseriesLoader(dataset_val_dir), 
    batch_size=BATCH_SIZE, shuffle=True)

# Set up training 
model = custom_vgg16.to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = SamplesLoss(loss="sinkhorn", p=2, blur=.05)
metrics = [loss_mse, loss_repulsion]
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

avg_vloss = 0.0
best_vloss = 999_999

# Train loop
for i_epoch in range(EPOCHS):
    # Train
    model.train(True)
    with tqdm.tqdm(total=len(train_loader)) as pbar:
        for image_batch, label_batch in train_loader:
            # Shift to GPU
            image_batch = image_batch.to(DEVICE)
            label_batch = label_batch.to(DEVICE)
            
            # Forward pass
            label_pred = model(image_batch)
            label_pred = label_pred[:label_batch.shape[0]]
            label_pred = label_pred.view(-1, NUM_KEYPOINTS, DIM_KEYPOINTS)
            # print('pred, batch shapes:',label_pred.shape, label_batch.shape)
            
            # Compute loss
            loss = loss_fn(label_pred, label_batch)
            if len(loss.shape) != 0:
                loss = torch.mean(loss)

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            pbar.set_description(f"Ep {i_epoch}, L {loss.item():.4f} 上VaL {avg_vloss}, " + 
                ', '.join([metric.__name__ + ' ' + str(torch.mean(metric(label_pred, label_batch)).item()) for metric in metrics]))
            pbar.update(1)
            
    # Validate
    model.eval()
    running_vloss = 0.0
    with torch.no_grad():
        for i, val_data in enumerate(val_loader):
            # Get Input
            val_inputs, val_labels = val_data
            val_inputs = val_inputs.to(DEVICE)
            val_labels = val_labels.to(DEVICE)
            # print(val_inputs.shape, val_inputs.dtype)
            
            # Get network out
            val_output = model(val_inputs)
            val_output = val_output.view(-1, NUM_KEYPOINTS, DIM_KEYPOINTS)
            val_labels = val_labels.view(-1, NUM_KEYPOINTS, DIM_KEYPOINTS)
            
            # Compute Loss
            # print('out')
            # print(val_output.shape, val_output.dtype)
            # print(val_output.shape, val_output.dtype)
            val_loss = loss_fn(val_output, val_labels)
            if len(val_loss.shape) != 0:
                val_loss = torch.mean(val_loss)
            
            running_vloss += val_loss
    avg_vloss = running_vloss / (i + 1)
    
    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
        best_vloss = avg_vloss
        model_path = output_folder / f'model_{timestamp}_{i_epoch}.pt'
        torch.save(model.state_dict(), model_path)

# Pass visualization image through network
# val_loader = DataseriesLoader(dataset_val_dir, batch_size=BATCH_SIZE)
# with torch.no_grad():
#     for i, vdata in enumerate(val_loader):
#         vinputs, vlabels = vdata
#         voutputs = model(vinputs)
#         label_pred = voutputs[0].cpu().numpy()
#     label_pred = model(image_vis[None, ...])
#     label_pred = label_pred[0].cpu().numpy()
#     image_vis_np = image_vis.detach().cpu().permute(1,2,0).numpy()

#     # Visualize
#     image_vis_np = (image_vis_np * 255).astype(np.uint8)

#     # Insert predicted corners
#     for i_corner in range(4):
#         marker_pos = label_pred[i_corner,:]
#         cv2.drawMarker(image_vis_np, tuple(marker_pos.astype(np.int32)), (0,0,255), markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2)

#     cv2.imshow("image", image_vis_np)
#     cv2.waitKey(1)

