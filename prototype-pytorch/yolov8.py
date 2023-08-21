import os
import torch
from ultralytics import YOLO

def main():
    os.system('mkdir ./.cache')
    os.environ['TORCH_HOME'] = './.cache'

    model = YOLO('yolov8s.pt')

    model.train(
        data='traindata-creator/dataset/yolov5-640-good-pics-v2-sgs-only/yolov5-640-good-pics-v2-sgs-only.yaml',
        epochs=2, 
        imgsz=640,
        project='training/yolov8',
        name='test-yv8',
        device= range(torch.cuda.device_count()))
    
    os.system('rm *.pt')
    
    
if __name__ == '__main__':
    main()