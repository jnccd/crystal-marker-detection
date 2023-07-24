import os
from pathlib import Path
import torch
from ultralytics import YOLO

def main():
    root_dir = Path(__file__).resolve().parent
    
    os.system('mkdir ./.cache')
    os.environ['TORCH_HOME'] = './.cache'

    for i in range(2):
        model = YOLO('yolov8s.pt').model
        with torch.no_grad():
            for param in model.parameters():
                param.add_(torch.randn(param.size()) * 0.03)
        torch.save(model, root_dir / f'test_noise_{i}.pt')
    
    #os.system('rm *.pt')
    
if __name__ == '__main__':
    main()