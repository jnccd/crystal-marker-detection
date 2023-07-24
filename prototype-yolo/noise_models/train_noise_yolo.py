import os
from pathlib import Path
import torch
from ultralytics import YOLO

def main():
    root_dir = Path(__file__).resolve().parent
    
    os.system('mkdir ./.cache')
    os.environ['TORCH_HOME'] = './.cache'

    for i in range(2):
        model = YOLO('yolov8s.pt')
        model.model = torch.load(root_dir / f'test_noise_{i}.pt')
        model.train(
            data='traindata-creator/dataset/yolov5-640-good-pics-v2-sgs-only/yolov5-640-good-pics-v2-sgs-only.yaml',
            epochs=30, 
            imgsz=640,
            project=root_dir,
            name=f'test-yv8-noise-{i}')
    
if __name__ == '__main__':
    main()