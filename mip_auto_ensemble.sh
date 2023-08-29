# label-dropout
with_gpu -n 1 sudo mip-docker-run --gpus '"device=$CUDA_VISIBLE_DEVICES"' ncarstensen/pcmd:0.11 python mip_create_ensemble_datasets.py -n gpv2-ld -op "-tf /data/pcmd/dataseries/af-the_good_pics_for_nn2_s1/ /data/pcmd/dataseries/af-the_good_pics_for_nn2_s2/ -r 0.2 -t yolov5 -s 640 -apldc"
python3 mip_worker_batch_train.py -n 6 -c "python batch_train/yolov5.py -d /data/pcmd/dataset/_noise-ensemble-gpv2-ld/ -t /data/pcmd/dataset/yolov5-640-on-skin-valset-v3-ensemble-test/ -e 300 -snr -o /data/pcmd/training/yolov5s-ld-ensemble/"
bash mip_worker_await.sh
with_gpu -n 1 sudo mip-docker-run --gpus '"device=$CUDA_VISIBLE_DEVICES"' ncarstensen/pcmd:0.11 python evaluation/plot_ensemble.py -n yolov5s-noise-ld -r /data/pcmd/training/yolov5s-ld-ensemble/ -pi 5 -ci 4 -cu 10% -rnp '.*yolo5aug$' -t "mAP scores for a given chance of label dropout augmentation in the dataset"

# ninety-deg-rotation
with_gpu -n 1 sudo mip-docker-run --gpus '"device=$CUDA_VISIBLE_DEVICES"' ncarstensen/pcmd:0.11 python mip_create_ensemble_datasets.py -n gpv2-ndr -op "-tf /data/pcmd/dataseries/af-the_good_pics_for_nn2_s1/ /data/pcmd/dataseries/af-the_good_pics_for_nn2_s2/ -r 0.2 -t yolov5 -s 640 -andrc"
python3 mip_worker_batch_train.py -n 6 -c "python batch_train/yolov5.py -d /data/pcmd/dataset/_noise-ensemble-gpv2-ndr/ -t /data/pcmd/dataset/yolov5-640-on-skin-valset-v3-ensemble-test/ -e 300 -snr -o /data/pcmd/training/yolov5s-ndr-ensemble/"
bash mip_worker_await.sh
with_gpu -n 1 sudo mip-docker-run --gpus '"device=$CUDA_VISIBLE_DEVICES"' ncarstensen/pcmd:0.11 python evaluation/plot_ensemble.py -n yolov5s-noise-ndr -r /data/pcmd/training/yolov5s-ndr-ensemble/ -pi 5 -ci 4 -cu 10% -rnp '.*yolo5aug$' -t "mAP scores for a given chance of ninety deg rotation augmentation in the dataset"

# ...