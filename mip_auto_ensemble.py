import os

def aug_ensemble_workflow(aug_token: str, aug_name: str, aug_arg: str):
    os.system(f'with_gpu -n 1 sudo mip-docker-run --gpus \'"device=$CUDA_VISIBLE_DEVICES"\' ncarstensen/pcmd:0.11 python mip_create_ensemble_datasets.py -n gpv2-{aug_token} -op "-tf /data/pcmd/dataseries/af-the_good_pics_for_nn2_s1/ /data/pcmd/dataseries/af-the_good_pics_for_nn2_s2/ -r 0.2 -t yolov5 -s 640 -{aug_arg}"')
    os.system(f'python3 mip_worker_batch_train.py -n 6 -c "python batch_train/yolov5.py -d /data/pcmd/dataset/_noise-ensemble-gpv2-{aug_token}/ -t /data/pcmd/dataset/yolov5-640-on-skin-valset-v3-ensemble-test/ -e 300 -snr -o /data/pcmd/training/yolov5s-{aug_token}-ensemble/"')
    os.system(f'bash mip_worker_await.sh')
    os.system(f'with_gpu -n 1 sudo mip-docker-run --gpus \'"device=$CUDA_VISIBLE_DEVICES"\' ncarstensen/pcmd:0.11 python evaluation/plot_ensemble.py -n yolov5s-noise-{aug_token} -r /data/pcmd/training/yolov5s-{aug_token}-ensemble/ -pi 5 -ci 4 -cu 10% -rnp \'.*yolo5aug$\' -t "mAP scores for a given chance of {aug_name} augmentation in the dataset"')
    
os.system(f'bash mip_worker_await.sh')

aug_ensemble_workflow(aug_token='ld', aug_name='label dropout', aug_arg='apldc')
aug_ensemble_workflow(aug_token='ndr', aug_name='ninety deg rotation', aug_arg='andrc')
aug_ensemble_workflow(aug_token='persp', aug_name='perspective', aug_arg='apc')
aug_ensemble_workflow(aug_token='rc', aug_name='random crop', aug_arg='arcc')
aug_ensemble_workflow(aug_token='rc2', aug_name='random crop v2', aug_arg='arc2c')
aug_ensemble_workflow(aug_token='lm', aug_name='label move', aug_arg='almc')
aug_ensemble_workflow(aug_token='lm2', aug_name='label move v2', aug_arg='alm2c')
aug_ensemble_workflow(aug_token='lm2', aug_name='label move v2', aug_arg='alm2c')