import ast
import torch
from geomloss import SamplesLoss

from utils import *

root_dir = Path(__file__).resolve().parent
dataset_dir = root_dir/'..'/'traindata-creator/dataset/pet-0-man-pet-v2-ordered-val'
dataset_train_dir = dataset_dir / 'train'
dataset_val_dir = dataset_dir / 'val'
input_folder = create_dir_if_not_exists(root_dir / 'output/to-rect')
eval_folder = create_dir_if_not_exists(input_folder / 'eval')

loss = SamplesLoss(loss="sinkhorn", p=2, blur=.05)

val_pointss = []
pred_pointss = []
for pred_points_file, val_points_file in zip(get_files_from_folders_with_ending([input_folder], '_p.txt'), get_files_from_folders_with_ending([dataset_val_dir], '_p.txt')):
    val_points_file = Path(val_points_file)
    pred_points_file = Path(pred_points_file)
    
    val_points = ast.literal_eval(read_textfile(val_points_file))
    pred_points = ast.literal_eval(read_textfile(pred_points_file))
    
    val_pointss.append(val_points)
    pred_pointss.append(pred_points)
    
val_tensor = torch.Tensor(val_pointss)
pred_tensor = torch.Tensor(pred_pointss)

print(pred_pointss)
loss_out = loss(pred_tensor, val_tensor)

print(loss_out)
print(torch.mean(loss_out))

print('---')

mAPD_score = mAPD_2D(pred_tensor, val_tensor)
print(f'mAPD: {mAPD_score}')