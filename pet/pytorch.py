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
from torchsummary import summary

from utils import *

EPOCHS = 400
BATCH_SIZE = 32
DEVICE = "cuda"
DIM_KEYPOINTS = 2
NUM_KEYPOINTS = 4
IMG_SIZE = 224
MODEL = 'vgg16'

root_dir = Path(__file__).resolve().parent
dataset_dir = root_dir/'..'/'traindata-creator/dataset/pet-0-man-pet-v2'
dataset_train_dir = dataset_dir / 'train'
dataset_val_dir = dataset_dir / 'val'
output_folder = create_dir_if_not_exists(root_dir / 'output/pt-vgg16-2')
eval_folder = create_dir_if_not_exists(output_folder / 'eval')

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
                    # A.Transpose(),
                    A.ShiftScaleRotate(shift_limit=0.05, rotate_limit=180, border_mode=cv2.BORDER_CONSTANT, p=1),
                    A.Perspective(scale=(0, 0)),
                    # A.Affine(shear=(-20, 20))
                    A.ColorJitter(hue=0.8),
                ],
                keypoint_params=A.KeypointParams(
                    format='xy',
                    remove_invisible = False
                ),
            )
        else:
            self.transform = A.Compose([
                    A.Resize(IMG_SIZE, IMG_SIZE, always_apply=True),
                    #A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.01, p=1),
                ], keypoint_params=A.KeypointParams(
                    format='xy',
                    remove_invisible = False
                ),
            )

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
                
                #print(image_np.shape, image_np.dtype)
                
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
        vgg16 = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        vgg16 = vgg16.features
        class CustomVGG16Head(nn.Module):
            def __init__(self, num_keypoints):
                super(CustomVGG16Head, self).__init__()
                self.conv1 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
                self.conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
                self.conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
                self.conv4 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
                self.relu = nn.ReLU()
                self.fc1 = nn.Linear(32 * 7 * 7, num_keypoints)

            def forward(self, x):
                x = self.conv1(x)
                x = self.relu(x)
                x = self.conv2(x)
                x = self.relu(x)
                x = self.conv3(x)
                x = self.relu(x)
                x = self.conv4(x)
                x = self.relu(x)
                x = x.view(x.size(0), -1)  # Flatten
                x = self.fc1(x)
                return x
        custom_head = CustomVGG16Head(NUM_KEYPOINTS * DIM_KEYPOINTS)
        model = nn.Sequential(vgg16, custom_head)
    elif MODEL == 'mobilenet':
        # Define the custom head
        class CustomHead(nn.Module):
            def __init__(self, in_features, num_classes):
                super(CustomHead, self).__init__()
                self.fc = nn.Sequential(
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(in_features, 512),
                    nn.ReLU(),
                    #nn.Dropout(0.5),
                    nn.Linear(512, num_classes),
                )
                # self.fc = nn.Sequential(
                #     nn.Conv2d(1280, 320, kernel_size=1),
                #     nn.BatchNorm2d(320),
                #     nn.ReLU(),
                #     nn.Conv2d(320, 160, kernel_size=3, stride=2),
                #     nn.BatchNorm2d(160),
                #     nn.ReLU(),
                #     nn.Conv2d(160, 40, kernel_size=3, stride=2),
                #     nn.BatchNorm2d(40),
                #     nn.ReLU(),
                #     nn.Conv2d(40, num_classes, kernel_size=1),
                #     nn.BatchNorm2d(num_classes),
                #     nn.ReLU(),
                #     nn.Flatten(),
                # )
                # self.fc = nn.Sequential(
                #     nn.Conv2d(1280, 320, kernel_size=1, stride=2),
                #     nn.BatchNorm2d(320),
                #     nn.ReLU(),
                #     nn.AdaptiveAvgPool2d((1, 1)),
                #     nn.Flatten(),
                #     nn.Linear(320, 128),
                #     nn.ReLU(),
                #     #nn.Dropout(0.5),
                #     nn.Linear(128, num_classes),
                # )
                # self.fc = nn.Sequential(
                #     nn.AdaptiveAvgPool2d((1, 1)),
                #     nn.Flatten(),
                #     nn.Linear(in_features, 1024),
                #     nn.ReLU(),
                #     nn.Linear(1024, 256),
                #     nn.ReLU(),
                #     nn.Linear(256, num_classes),
                # )

            def forward(self, x):
                x = self.fc(x)
                return x

        # Load the MobileNetV2 base model
        base_model = models.mobilenet_v2(pretrained=True)
        # Freeze all layers in the base model
        # for param in base_model.parameters():
        #     param.requires_grad = False

        # Modify the final classification layer of the base model
        in_features = base_model.classifier[1].in_features
        custom_head = CustomHead(in_features, num_classes=NUM_KEYPOINTS * DIM_KEYPOINTS)

        # Create the full model by combining the base model and custom head
        model = nn.Sequential(base_model.features, custom_head)
        
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
summary(model, input_size=(3, IMG_SIZE, IMG_SIZE))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
#optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.95, weight_decay=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
loss_fn = SamplesLoss(loss="sinkhorn", p=2, blur=.05)
metrics = [loss_mse, loss_repulsion]
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
model_path = output_folder / f'best_{MODEL}.pt'

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
            scheduler.step()
            optimizer.step()
            
            pbar.set_description(f"Ep {i_epoch}, L {np.average(batch_losses):.4f} 上VaL {avg_val_loss}, LR {optimizer.param_groups[0]['lr']}, " + 
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
        torch.save(model.state_dict(), model_path)

if EPOCHS > 2:
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(1, EPOCHS), loss_history[1:], label="train_loss")
    plt.plot(np.arange(1, EPOCHS), val_loss_history[1:], label="val_loss")
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
            
            #print(val_image_np.shape, val_image_np.dtype)
            
            pred = val_outputs.cpu().numpy()
            gt = val_labels.cpu().numpy()

            # print('pred', pred)
            # print('gt', gt)
            for ip in range(4):
                pred_point = pred[ip,:]# * IMG_SIZE
                cv2.drawMarker(val_image_np, tuple(pred_point.astype(np.int32)), (255,0,0), markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2)
                
                gt_point = gt[ip,:]
                cv2.drawMarker(val_image_np, tuple(gt_point.astype(np.int32)), (0,0,255), markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2)

            cv2.imwrite(str(eval_folder / f'{i}_{vii}.png'), val_image_np)
            cv2.imshow("image", val_image_np)
            k = cv2.waitKey(0)
            if k == ord('q'):
                    break

