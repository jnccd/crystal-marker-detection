import os
from evaluation.utility import *

# --- First run
# def aug_ensemble_workflow(aug_token: str, aug_name: str, aug_arg: str):
#     os.system(f'with_gpu -n 1 sudo mip-docker-run --gpus \'"device=$CUDA_VISIBLE_DEVICES"\' ncarstensen/pcmd:0.11 python mip_create_ensemble_datasets.py -n gpv2-{aug_token} -op "-tf /data/pcmd/dataseries/af-the_good_pics_for_nn2_s1/ /data/pcmd/dataseries/af-the_good_pics_for_nn2_s2/ -r 0.2 -t yolov5 -s 640 -{aug_arg}"')
#     os.system(f'python3 mip_worker_batch_train.py -n 6 -c "python batch_train/yolov5.py -d /data/pcmd/dataset/_noise-ensemble-gpv2-{aug_token}/ -t /data/pcmd/dataset/yolov5-640-on-skin-valset-v3-ensemble-test/ -e 300 -snr -o /data/pcmd/training/yolov5s-{aug_token}-ensemble/"')
#     os.system(f'bash mip_worker_await.sh')
#     os.system(f'with_gpu -n 1 sudo mip-docker-run --gpus \'"device=$CUDA_VISIBLE_DEVICES"\' ncarstensen/pcmd:0.11 python evaluation/plot_ensemble.py -n yolov5s-noise-{aug_token} -r /data/pcmd/training/yolov5s-{aug_token}-ensemble/ -pi 5 -ci 4 -cu 10% -rnp \'.*yolo5aug$\' -t "mAP scores for a given chance of {aug_name} augmentation in the dataset"')

# --- After yolov5s aug hyp opt
# opt_aug_params = "-asgsc 0.6832841914530021 -apldc 0.7472412493690345 -apc 0.8965160681643762 -aps 0.14231038299950238 -arc 0.7194422300566737 -ars 105.83149978479527 -andrc 0.8340463734130839 -arcc 0.12321476086303013 -arc2c 0.20432659337780826 -almc 0.020587253796741117 -alm2c 0.017673002005261285 -abdc 0.8011914710474702 -alcc 0.21241785889630882 -agnc 0.3599594528673184 -agns 80.5545109790516"
# epochs = 368

# --- Basic ensemble
opt_aug_params = "-ars 180 -aps 0.3"
epochs = 300

# --- After yolov5s aug hyp opt 2
# opt_aug_params = "-asgsc 0.008982257070160649 -apldc 0.9450952906618159 -apc 0.6573674532026004 -aps 0.2804576787977128 -arc 0.5427787640299094 -ars 324.55179667010617 -andrc 0.6035650010036894 -arcc 0.6817057409388554 -arc2c 0.39548280581255923 -almc 0.4601560149315024 -alm2c 0.03810201058767085 -abdc 0.770387100335939 -alcc 0.5298416478024052 -agnc 0.5099200084102142 -agns 22.830199041806175"
# epochs = 347

docker_image = 'ncarstensen/pcmd:0.13'
def aug_ensemble_workflow(aug_token: str, aug_name: str, aug_arg: str):
    # Get all param values that are not being changed in this ensemble
    other_aug_params_list = list(filter(lambda x: x[0] != f'-{aug_arg}', unflatten(opt_aug_params.split(' '), 2)))
    other_aug_params_str = ' '.join(flatten(other_aug_params_list))
    
    train_folder = f'/data/pcmd/training/_augments-zero-sahi-ensemble-2-rerun/yolov5s-{aug_token}-ensemble/'
    plot_name = f'zero-based-{aug_token}-sahi-ensemble-2-rerun'
    
    os.system(f'with_gpu -n 1 sudo mip-docker-run --gpus \'"device=$CUDA_VISIBLE_DEVICES"\' {docker_image} python mip_create_ensemble_datasets.py -n gpv2-{aug_token} -op "-tf /data/pcmd/dataseries/af-the_good_pics_for_nn2_s1/ /data/pcmd/dataseries/af-the_good_pics_for_nn2_s2/ -r 0.2 -t yolov5 -s 640 {other_aug_params_str} -{aug_arg}"')
    os.system(f'python3 mip_worker_batch_train.py -n 6 -c "python batch_train/yolov5.py -d /data/pcmd/dataset/_noise-ensemble-gpv2-{aug_token}/ -t /data/pcmd/dataset/yolov5-0-on-skin-valset-v3-ensemble-test/ -e {epochs} -snr -us -o {train_folder}"')
    os.system(f'bash mip_worker_await.sh')
    os.system(f'with_gpu -n 1 sudo mip-docker-run --gpus \'"device=$CUDA_VISIBLE_DEVICES"\' {docker_image} python evaluation/plot_ensemble.py -n {plot_name} -r {train_folder} -da -bfl -xl "Chance of augmentation usage per sample" -pi 5 -ci 4 -cu 10% -rnp \'.*yolo5aug$\' -t "mAP Scores for a given Chance of {aug_name} Augmentation in the Dataset"')
    
os.system(f'bash mip_worker_await.sh')

aug_ensemble_workflow(aug_token='rot', aug_name='Rotation', aug_arg='arc')
aug_ensemble_workflow(aug_token='persp', aug_name='Perspective', aug_arg='apc')
# aug_ensemble_workflow(aug_token='sgs', aug_name='Smart Grid Shuffling', aug_arg='asgsc')
# aug_ensemble_workflow(aug_token='rc', aug_name='Random Crop', aug_arg='arcc')
# aug_ensemble_workflow(aug_token='rc2', aug_name='Random Crop v2', aug_arg='arc2c')
# aug_ensemble_workflow(aug_token='bd', aug_name='Black Dot', aug_arg='abdc')
# aug_ensemble_workflow(aug_token='gn', aug_name='Gauss Noise', aug_arg='agnc')
# aug_ensemble_workflow(aug_token='ld', aug_name='Label Dropout', aug_arg='apldc')
# aug_ensemble_workflow(aug_token='ndr', aug_name='Ninety Degree Rotation', aug_arg='andrc')
# aug_ensemble_workflow(aug_token='lm', aug_name='Label Move', aug_arg='almc')
# aug_ensemble_workflow(aug_token='lm2', aug_name='Label Move v2', aug_arg='alm2c')
