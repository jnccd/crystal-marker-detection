import os
from pathlib import Path
import torch
from ultralytics import YOLO

def main():
    root_dir = Path(__file__).resolve().parent
    
    os.system('mkdir ./.cache')
    os.environ['TORCH_HOME'] = './.cache'

    for i in range(2):
        #model = YOLO('batch_train/models/yolov5s.pt').model
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=f'yolov5s.pt')
        #next(model.parameters()).to(torch.device("cuda:0"))
        with torch.no_grad():
            for param in model.parameters():
                param.to(torch.device("cuda:0"))
                print(param, param.device == torch.device("cuda:0"))
                param[3][0].add(torch.randn(param.size()[2:]) * 0.03)
        torch.save(model, root_dir / f'test_noise_{i}.pt')
    
    #os.system('rm *.pt')
    
if __name__ == '__main__':
    main()