import os
from evaluation.utility import *

# --- First run
# def aug_ensemble_workflow(aug_token: str, aug_name: str, aug_arg: str):
#     os.system(f'with_gpu -n 1 sudo mip-docker-run --gpus \'"device=$CUDA_VISIBLE_DEVICES"\' ncarstensen/pcmd:0.11 python mip_create_ensemble_datasets.py -n gpv2-{aug_token} -op "-tf /data/pcmd/dataseries/af-the_good_pics_for_nn2_s1/ /data/pcmd/dataseries/af-the_good_pics_for_nn2_s2/ -r 0.2 -t yolov5 -s 640 -{aug_arg}"')
#     os.system(f'python3 mip_worker_batch_train.py -n 6 -c "python batch_train/yolov5.py -d /data/pcmd/dataset/_noise-ensemble-gpv2-{aug_token}/ -t /data/pcmd/dataset/yolov5-640-on-skin-valset-v3-ensemble-test/ -e 300 -snr -o /data/pcmd/training/yolov5s-{aug_token}-ensemble/"')
#     os.system(f'bash mip_worker_await.sh')
#     os.system(f'with_gpu -n 1 sudo mip-docker-run --gpus \'"device=$CUDA_VISIBLE_DEVICES"\' ncarstensen/pcmd:0.11 python evaluation/plot_ensemble.py -n yolov5s-noise-{aug_token} -r /data/pcmd/training/yolov5s-{aug_token}-ensemble/ -pi 5 -ci 4 -cu 10% -rnp \'.*yolo5aug$\' -t "mAP scores for a given chance of {aug_name} augmentation in the dataset"')

# --- After yolov5s aug hyp opt
opt_aug_params = "-asgsc 0.16536455065004735 -apldc 0.3406092762590903 -apc 0.9026868390392013 -aps 0.44744759491769104 -arc 0.8397534486075489 -ars 269.5297433583759 -andrc 0.4779575209987885 -arcc 0.07811140975400804 -arc2c 0.09484458554495206 -almc 0.3281069403856025 -alm2c 0.04946212899649677 -abdc 0.5932304638260782 -alcc 0.09002886339102176 -agnc 0.2850102716939429 -agns 147.09472782083168"
epochs = 219
docker_image = 'ncarstensen/pcmd:0.13'
def aug_ensemble_workflow(aug_token: str, aug_name: str, aug_arg: str):
    # Get all param values that are not being changed in this ensemble
    other_aug_params_list = list(filter(lambda x: x[0] != f'-{aug_arg}', unflatten(opt_aug_params.split(' '), 2)))
    other_aug_params_str = ' '.join(flatten(other_aug_params_list))
    
    train_folder_name = f'yolov5s-hypsear-params-ensemble-{aug_token}'
    
    os.system(f'with_gpu -n 1 sudo mip-docker-run --gpus \'"device=$CUDA_VISIBLE_DEVICES"\' {docker_image} python mip_create_ensemble_datasets.py -n gpv2-{aug_token} -op "-tf /data/pcmd/dataseries/af-the_good_pics_for_nn2_s1/ /data/pcmd/dataseries/af-the_good_pics_for_nn2_s2/ -r 0.2 -t yolov5 -s 640 {other_aug_params_str} -{aug_arg}"')
    os.system(f'python3 mip_worker_batch_train.py -n 6 -c "python batch_train/yolov5.py -d /data/pcmd/dataset/_noise-ensemble-gpv2-{aug_token}/ -t /data/pcmd/dataset/yolov5-640-on-skin-valset-v3-ensemble-test/ -e {epochs} -snr -o /data/pcmd/training/{train_folder_name}/"')
    os.system(f'bash mip_worker_await.sh')
    os.system(f'with_gpu -n 1 sudo mip-docker-run --gpus \'"device=$CUDA_VISIBLE_DEVICES"\' {docker_image} python evaluation/plot_ensemble.py -n {train_folder_name} -r /data/pcmd/training/{train_folder_name}/ -pi 5 -ci 4 -cu 10% -rnp \'.*yolo5aug$\' -t "mAP scores for a given chance of {aug_name} augmentation in the dataset"')
    
os.system(f'bash mip_worker_await.sh')

aug_ensemble_workflow(aug_token='persp', aug_name='perspective', aug_arg='apc')
aug_ensemble_workflow(aug_token='rot', aug_name='rotation', aug_arg='arc')
aug_ensemble_workflow(aug_token='sgs', aug_name='smart grid shuffling', aug_arg='asgsc')
aug_ensemble_workflow(aug_token='gn', aug_name='gauss noise', aug_arg='agnc')
aug_ensemble_workflow(aug_token='bd', aug_name='black dot', aug_arg='abdc')
aug_ensemble_workflow(aug_token='ld', aug_name='label dropout', aug_arg='apldc')
aug_ensemble_workflow(aug_token='ndr', aug_name='ninety deg rotation', aug_arg='andrc')
aug_ensemble_workflow(aug_token='lm', aug_name='label move', aug_arg='almc')
aug_ensemble_workflow(aug_token='lm2', aug_name='label move v2', aug_arg='alm2c')
aug_ensemble_workflow(aug_token='rc', aug_name='random crop', aug_arg='arcc')
aug_ensemble_workflow(aug_token='rc2', aug_name='random crop v2', aug_arg='arc2c')
