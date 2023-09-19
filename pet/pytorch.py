import os
import glob
import cv2
from matplotlib import pyplot as plt
import numpy as np
import ast
import scipy
import tqdm
from datetime import datetime
from geomloss import SamplesLoss
import albumentations as A

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models

from utils import *

LR = 1e-3
EPOCHS = 10
BATCH_SIZE = 32
DEVICE = "cuda"
DIM_KEYPOINTS = 2
NUM_KEYPOINTS = 4
IMG_SIZE = 224
MODEL = 'vgg16'

root_dir = Path(__file__).resolve().parent
dataset_dir = root_dir/'..'/'traindata-creator/dataset/pet-0-man-pet'
dataset_train_dir = dataset_dir / 'train'
dataset_val_dir = dataset_dir / 'val'
output_folder = create_dir_if_not_exists(root_dir / 'output/pt')

# --- Dataloader ----------------------------------------------------------------------------------------

class DataseriesLoader(Dataset):
    def __init__(self, dataseries_dir, aug = False):
        # Read filenames
        self.image_filenames = get_files_from_folders_with_ending([dataseries_dir], '.png')
        self.label_filenames = get_files_from_folders_with_ending([dataseries_dir], '.txt')
        self.image_label_filenames = list(zip(self.image_filenames, self.label_filenames))
        
        # Define transform pipeline
        if aug:
            self.transform = A.Compose([
                A.Resize(IMG_SIZE, IMG_SIZE, always_apply=True),
                A.RandomRotate90(),
                A.Transpose(),
                A.SafeRotate(always_apply=True),
                A.ShiftScaleRotate(rotate_limit=0, shift_limit=0.15),
                A.HueSaturationValue(),
                A.ColorJitter(),
            ], keypoint_params=A.KeypointParams(format='xy'))
        else:
            self.transform = A.Compose([
                A.Resize(IMG_SIZE, IMG_SIZE, always_apply=True),
            ], keypoint_params=A.KeypointParams(format='xy'))

        print(f"Found {len(self.image_label_filenames)} images in {dataseries_dir}")

    def __len__(self):
        return len(self.image_label_filenames)
    
    def __getitem__(self, idx):
        # Read image and points
        image = cv2.imread(self.image_label_filenames[idx][0])
        points = ast.literal_eval(read_textfile(self.image_label_filenames[idx][1]))
        
        # Augment
        if self.transform != None:
            transformed = self.transform(image=image, keypoints=points)
            image = transformed['image']
            points = transformed['keypoints']
            #print(points)
        
        # Prepare for torch
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2,0,1)
        label = torch.from_numpy(
            np.array(points)
        ).float()

        return image, label
    
def interactive_validate_dataloader(loader: DataLoader):
    k = 0
    with torch.no_grad():
        for i, data in enumerate(loader):
            inputs, labels = data
            inputs = inputs.to('cpu')
            labels = labels.to('cpu')
            
            for (ii, input), (li, labels) in zip(enumerate(inputs), enumerate(labels)):
                
                image_np = input.detach().cpu().permute(1,2,0).numpy()
                image_np = np.ascontiguousarray((image_np * 255), dtype=np.uint8)
                
                print(image_np.shape, image_np.dtype)
                
                gt = labels.cpu().numpy()

                for ip in range(4):
                    gt_point = gt[ip,:]
                    cv2.drawMarker(image_np, tuple(gt_point.astype(np.int32)), (0,0,255-ip*30), markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2)

                cv2.imshow("image", image_np)
                k = cv2.waitKey(0)
                if k == ord('q'):
                    break
            if k == ord('q'):
                    break
    
# --- Model ----------------------------------------------------------------------------------------

def get_model():
    if MODEL == 'scnn':
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
        model = SimpleCNN()
    elif MODEL == 'vgg16':
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
        model = nn.Sequential(vgg16, custom_head)
        
    return model

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

# Test train data
interactive_validate_dataloader(
    DataLoader(
        DataseriesLoader(
            dataset_train_dir, 
            aug=True
        ), 
        batch_size=BATCH_SIZE, 
        shuffle=True
    )
)

# define dataloaders
train_loader = DataLoader(
    DataseriesLoader(
        dataset_train_dir, 
        aug=True
    ), 
    batch_size=BATCH_SIZE, 
    shuffle=True
)
val_loader = DataLoader(
    DataseriesLoader(dataset_val_dir), 
    batch_size=BATCH_SIZE
)

# Set up training 
model = get_model().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = SamplesLoss(loss="sinkhorn", p=2, blur=.05)
metrics = [loss_mse, loss_repulsion]
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
model_path = output_folder / f'best_model_{name_of_object(loss_fn)}.pt'

avg_val_loss = 0.0
best_vloss = 999_999
loss_history = []
val_loss_history = []

# Train loop
for i_epoch in range(EPOCHS):
    # Train
    model.train(True)
    batch_losses = []
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
            batch_losses.append(loss.to('cpu').detach().numpy())

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            pbar.set_description(f"Ep {i_epoch}, L {np.average(batch_losses):.4f} 上VaL {avg_val_loss}, " + 
                ', '.join([metric.__name__ + ' ' + str(torch.mean(metric(label_pred, label_batch)).item()) for metric in metrics]))
            pbar.update(1)
    avg_loss = np.average(batch_losses)
    loss_history.append(avg_loss)
    
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
    avg_val_loss = running_vloss / (i + 1)
    val_loss_history.append(avg_val_loss.to('cpu').detach().numpy())
    
    # Track best performance, and save the model's state
    if val_loss_history[-1] < best_vloss:
        best_vloss = val_loss_history[-1]
        if i_epoch > 5:
            torch.save(model.state_dict(), model_path)

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, EPOCHS), loss_history, label="train_loss")
plt.plot(np.arange(0, EPOCHS), val_loss_history, label="val_loss")
plt.title("Keypoint detection loss over epochs")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.savefig(output_folder / 'plot.png')

model = get_model().to(DEVICE)
model.load_state_dict(torch.load(model_path))

# Visualize val data out
val_loader = DataLoader(
    DataseriesLoader(dataset_val_dir), 
    batch_size=BATCH_SIZE
)
val_img_idx = 0
with torch.no_grad():
    for i, vdata in enumerate(val_loader):
        val_inputs, val_labels = vdata
        val_inputs = val_inputs.to(DEVICE)
        val_labels = val_labels.to(DEVICE)
        
        val_outputs = model(val_inputs)
        val_outputs = val_outputs.view(-1, NUM_KEYPOINTS, DIM_KEYPOINTS)
        # print(val_inputs.shape, val_outputs.shape, val_labels.shape)
       
        for (vii, val_input), (voi, val_outputs), (vli, val_labels) in zip(enumerate(val_inputs), enumerate(val_outputs), enumerate(val_labels)):
            
            val_image_np = val_input.detach().cpu().permute(1,2,0).numpy()
            val_image_np = np.ascontiguousarray((val_image_np * 255), dtype=np.uint8)
            
            print(val_image_np.shape, val_image_np.dtype)
            
            pred = val_outputs.cpu().numpy()
            gt = val_labels.cpu().numpy()

            # print('pred', pred)
            # print('gt', gt)
            for ip in range(4):
                pred_point = pred[ip,:]# * IMG_SIZE
                cv2.drawMarker(val_image_np, tuple(pred_point.astype(np.int32)), (255,0,0), markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2)
                
                gt_point = gt[ip,:]
                cv2.drawMarker(val_image_np, tuple(gt_point.astype(np.int32)), (0,0,255), markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2)

            cv2.imshow("image", val_image_np)
            k = cv2.waitKey(0)
            if k == ord('q'):
                    break

