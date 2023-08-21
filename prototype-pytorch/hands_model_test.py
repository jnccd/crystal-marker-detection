# Imports
import cv2
import torch
import torch.hub
import torchvision.transforms as transforms

# Create the model
model = torch.hub.load(
    repo_or_dir='guglielmocamporese/hands-segmentation-pytorch', 
    model='hand_segmentor', 
    pretrained=True
)

# Read the image
image = cv2.imread('traindata-creator/dataset/yolov5-0-on-skin-valset-v3-ensemble-test/0.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
transform = transforms.Compose([
    transforms.ToTensor()
])
img_tensor = transform(image)

print(img_tensor.shape)

# Inference
model.eval()
preds = model(img_tensor).argmax(1) # [B, H, W]