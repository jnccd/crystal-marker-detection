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
from torch.nn import ConvTranspose2d
from torch.nn import Conv2d
from torch.nn import MaxPool2d
from torch.nn import Module
from torch.nn import ModuleList
from torch.nn import ReLU
from torchvision.transforms import CenterCrop
from torch.nn import functional as F

from utils import *

EPOCHS = 100
BATCH_SIZE = 32
DEVICE = "cuda"
DIM_KEYPOINTS = 2
NUM_KEYPOINTS = 4
IMG_SIZE = 160
MODEL = 'unet'

root_dir = Path(__file__).resolve().parent
dataset_dir = root_dir/'..'/'traindata-creator/dataset/segpet-0-man-pet-v2'
dataset_train_dir = dataset_dir / 'train'
dataset_val_dir = dataset_dir / 'val'
output_folder = create_dir_if_not_exists(root_dir / 'output/pt-seg')
eval_folder = create_dir_if_not_exists(output_folder / 'eval')

# --- Dataloader ----------------------------------------------------------------------------------------

class DataseriesLoader(Dataset):
    def __init__(self, dataseries_dir, aug = False):
        # Read filenames
        self.image_filenames = get_files_from_folders_with_ending([dataseries_dir], '_in.png')
        self.label_filenames = get_files_from_folders_with_ending([dataseries_dir], '_seg.png')
        self.image_label_filenames = list(zip(self.image_filenames, self.label_filenames))
        
        # Define transform pipeline
        if aug:
            self.transform = A.Compose([
                    A.Resize(IMG_SIZE, IMG_SIZE, always_apply=True),
                    A.RandomRotate90(),
                    # A.Transpose(),
                    A.ShiftScaleRotate(shift_limit=0.05, rotate_limit=270, border_mode=cv2.BORDER_CONSTANT, p=1),
                    A.Perspective(scale=(0, 0)),
                    # A.Affine(shear=(-20, 20))
                    # A.HueSaturationValue(),
                    # A.ColorJitter(),
                ],
            )
        else:
            self.transform = A.Compose([
                    A.Resize(IMG_SIZE, IMG_SIZE, always_apply=True),
                    #A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.01, p=1),
                ],
            )

        print(f"Found {len(self.image_label_filenames)} images in {dataseries_dir}")

    def __len__(self):
        return len(self.image_label_filenames)
    
    def __getitem__(self, idx):
        # Read image and points
        image = cv2.imread(self.image_label_filenames[idx][0])
        mask = cv2.imread(self.image_label_filenames[idx][1], cv2.IMREAD_GRAYSCALE)
        
        # Augment
        if self.transform != None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
            #print(points)
        
        # Prepare for torch
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2,0,1)
        mask = mask.astype(np.float32) / 255.0
        mask = torch.from_numpy(mask)

        return image, mask
    
def interactive_validate_dataloader(loader: DataLoader):
    k = 0
    with torch.no_grad():
        for i, data in enumerate(loader):
            inputs, labels = data
            inputs = inputs.to('cpu')
            labels = labels.to('cpu')
            
            for (ii, input), (li, mask) in zip(enumerate(inputs), enumerate(labels)):
                
                image_np = input.detach().cpu().permute(1,2,0).numpy()
                image_np = np.ascontiguousarray((image_np * 255), dtype=np.uint8)
                
                #print(image_np.shape, image_np.dtype)
                
                mask_np = mask.detach().cpu().numpy()
                mask_np = np.ascontiguousarray((mask_np * 255), dtype=np.uint8)
                
                cv2.imshow("image", cv2.hconcat([image_np, cv2.cvtColor(mask_np, cv2.COLOR_GRAY2BGR)]))
                k = cv2.waitKey(0)
                if k == ord('q'):
                    break
            if k == ord('q'):
                    break
    
# --- Model ----------------------------------------------------------------------------------------

def get_model():
    if MODEL == 'unet':
        def conv(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            
        def up_conv(in_channels, out_channels):
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
                nn.ReLU(inplace=True)
            )

        class UNet(nn.Module):
            def __init__(self, pretrained=True, out_channels=12):
                super().__init__()

                self.encoder = models.vgg16_bn(pretrained=pretrained).features
                self.block1 = nn.Sequential(*self.encoder[:6])
                self.block2 = nn.Sequential(*self.encoder[6:13])
                self.block3 = nn.Sequential(*self.encoder[13:20])
                self.block4 = nn.Sequential(*self.encoder[20:27])
                self.block5 = nn.Sequential(*self.encoder[27:34])

                self.bottleneck = nn.Sequential(*self.encoder[34:])
                self.conv_bottleneck = conv(512, 1024)

                self.up_conv6 = up_conv(1024, 512)
                self.conv6 = conv(512 + 512, 512)
                self.up_conv7 = up_conv(512, 256)
                self.conv7 = conv(256 + 512, 256)
                self.up_conv8 = up_conv(256, 128)
                self.conv8 = conv(128 + 256, 128)
                self.up_conv9 = up_conv(128, 64)
                self.conv9 = conv(64 + 128, 64)
                self.up_conv10 = up_conv(64, 32)
                self.conv10 = conv(32 + 64, 32)
                self.conv11 = nn.Conv2d(32, out_channels, kernel_size=1)
                
                self.sigm = nn.Sigmoid()
                
            def forward(self, x):
                block1 = self.block1(x)
                block2 = self.block2(block1)
                block3 = self.block3(block2)
                block4 = self.block4(block3)
                block5 = self.block5(block4)


                bottleneck = self.bottleneck(block5)
                x = self.conv_bottleneck(bottleneck)


                x = self.up_conv6(x)
                x = torch.cat([x, block5], dim=1)
                x = self.conv6(x)


                x = self.up_conv7(x)
                x = torch.cat([x, block4], dim=1)
                x = self.conv7(x)


                x = self.up_conv8(x)
                x = torch.cat([x, block3], dim=1)
                x = self.conv8(x)


                x = self.up_conv9(x)
                x = torch.cat([x, block2], dim=1)
                x = self.conv9(x)


                x = self.up_conv10(x)
                x = torch.cat([x, block1], dim=1)
                x = self.conv10(x)


                x = self.conv11(x)

                x = self.sigm(x)

                return x

            
        model = UNet(out_channels=1)
        
    return model

# --- Loss ----------------------------------------------------------------------------------------

# Define the Dice Loss
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets):
        #print(inputs.shape, targets.shape)
        smooth = 1e-1
        
        flat_inputs = inputs.contiguous().view(-1)
        flat_targets = targets.contiguous().view(-1)

        intersection = (flat_inputs * flat_targets).sum()
        union = flat_inputs.pow(2).sum() + flat_targets.pow(2).sum()
        dice_coefficient = (2. * intersection + smooth) / (union + smooth)

        return 1.0 - dice_coefficient

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
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.2)
loss_fn = DiceLoss()
metrics = []
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
            #print('shapes:',label_pred.shape, label_batch.shape, image_batch.shape)
            
            # Compute loss
            loss = loss_fn(label_pred, label_batch)
            if len(loss.shape) != 0:
                loss = torch.mean(loss)
            batch_losses.append(loss.to('cpu').detach().numpy())

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
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
    if val_loss_history[-1] < best_vloss and val_loss_history[-1] > 0:
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
        print(val_inputs.shape, val_outputs.shape, val_labels.shape)
       
        for (vii, val_input), (voi, val_outputs), (vli, val_labels) in zip(enumerate(val_inputs), enumerate(val_outputs), enumerate(val_labels)):
            
            #print(val_input.shape, val_outputs.shape, val_labels.shape)
            # TODO: Clip values [0,1]
            val_input_np = val_input.detach().cpu().permute(1,2,0).numpy()
            val_input_np = np.ascontiguousarray((val_input_np * 255), dtype=np.uint8)
            
            val_pred_np = val_outputs.detach().cpu().permute(1,2,0).numpy()
            val_pred_np = np.ascontiguousarray((val_pred_np * 255), dtype=np.uint8)
            
            val_gt_np = val_labels.detach().cpu().numpy()
            val_gt_np = np.ascontiguousarray((val_gt_np * 255), dtype=np.uint8)
            
            cv2.imwrite(str(eval_folder / f'{i}_{vii}_in.png'), val_input_np)
            cv2.imwrite(str(eval_folder / f'{i}_{vii}_pred.png'), val_pred_np)
            cv2.imwrite(str(eval_folder / f'{i}_{vii}_gt.png'), val_gt_np)
            cv2.imshow("image", cv2.hconcat([val_input_np, cv2.cvtColor(val_pred_np ,cv2.COLOR_GRAY2BGR), cv2.cvtColor(val_gt_np ,cv2.COLOR_GRAY2BGR)]))
            k = cv2.waitKey(0)
            if k == ord('q'):
                    break

