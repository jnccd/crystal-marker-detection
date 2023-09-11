exit 0
# This file is not meant to be executed but rather as a lookup table for frequently used commands

# Create dataseries
python traindata-creator/createArucoFrameDataseries.py -if N:\\Downloads\\Archives\\FabioBilder\\the_good_pics_for_nn -lirf -ng 
python traindata-creator/createArucoFrameDataseries.py -if traindata-creator/raw/webcam-images-203d3683-7c91-4429-93b6-be24a28f47bf/ -ng
python traindata-creator/createArucoFrameDataseries.py -if traindata-creator/raw/webcam-images-1312ecab-04e7-4f45-a714-07365d8c0dae/ -ng
python traindata-creator/createArucoFrameDataseries.py -if traindata-creator/raw/webcam-images-f50ec0b7-f960-400d-91f0-c42a6d44e3d0/ -ng
python traindata-creator/createArucoFrameDataseries.py -if N:\\Downloads\\Archives\\FabioBilder\\the_good_pics_for_nn2_s1 -lirf -ng
python traindata-creator/createArucoFrameDataseries.py -if N:\\Downloads\\Archives\\FabioBilder\\the_good_pics_for_nn2_s2 -lirf -ng
python traindata-creator/createArucoFrameDataseries.py -if N:\\Downloads\\Archives\\FabioBilder\\the_good_pics_for_nn2_s1 -ng -lirf && python traindata-creator/createArucoFrameDataseries.py -if N:\\Downloads\\Archives\\FabioBilder\\the_good_pics_for_nn2_s2 -ng -lirf
python traindata-creator/createManualDataseries.py -if N:\\Downloads\\Archives\\FabioBilder\\fPCS_on_skin
python traindata-creator/createArucoFrameDataseries.py -if N:\\Downloads\\Archives\\FabioBilder\\the_good_pics_for_nn3_s1
python traindata-creator/createArucoFrameDataseries.py -if N:\\Downloads\\Archives\\FabioBilder\\the_good_pics_for_nn3_s2
python traindata-creator/createArucoFrameDataseries.py -if N:\\Backup\\MasterArbeitPhotos\\good-zimmer-v1

# Create datasets from dataseries
python traindata-creator/createDataset.py -n red-rects -tf traindata-creator/dataseries/af-webcam-images-1312ecab-04e7-4f45-a714-07365d8c0dae/ traindata-creator/dataseries/af-webcam-images-f50ec0b7-f960-400d-91f0-c42a6d44e3d0/ -vf traindata-creator/dataseries/af-webcam-images-203d3683-7c91-4429-93b6-be24a28f47bf/ -t seg -s 640 -a -aim 4
python traindata-creator/createDataset.py -n good-pics-v1 -tf traindata-creator/dataseries/af-the_good_pics_for_nn/ -r 0.2 -t yolov5 -s 640 -a -aim 4
python traindata-creator/createDataset.py -n good-pics-v1 -tf traindata-creator/dataseries/af-the_good_pics_for_nn/ -r 0.2 -t seg -s 640 -a -aim 4
python traindata-creator/createDataset.py -n good-pics-v2 -tf traindata-creator/dataseries/af-the_good_pics_for_nn2_s1/ traindata-creator/dataseries/af-the_good_pics_for_nn2_s2/ -r 0.2 -t yolov5 -s 640 -a -aim 4
python traindata-creator/createDataset.py -n good-pics-v2 -tf traindata-creator/dataseries/af-the_good_pics_for_nn2_s1/ traindata-creator/dataseries/af-the_good_pics_for_nn2_s2/ -r 0.2 -t seg -s 640 -a -aim 4
# New sets
python traindata-creator/createDataset.py -n good-pics-v1-no-aug -tf traindata-creator/dataseries/af-the_good_pics_for_nn/ -r 0.2 -t yolov5 -s 640
python traindata-creator/createDataset.py -n good-pics-v2-no-aug -tf traindata-creator/dataseries/af-the_good_pics_for_nn2_s1/ traindata-creator/dataseries/af-the_good_pics_for_nn2_s2/ -r 0.2 -t yolov5 -s 640
python traindata-creator/createDataset.py -n good-pics-v1-sgs-only -tf traindata-creator/dataseries/af-the_good_pics_for_nn/ -r 0.2 -t yolov5 -s 640 -a -aim 4 -asgsc 1
python traindata-creator/createDataset.py -n good-pics-v2-sgs-only -tf traindata-creator/dataseries/af-the_good_pics_for_nn2_s1/ traindata-creator/dataseries/af-the_good_pics_for_nn2_s2/ -r 0.2 -t yolov5 -s 640 -a -aim 4 -asgsc 1
python traindata-creator/createDataset.py -n good-pics-v1-pld-only -tf traindata-creator/dataseries/af-the_good_pics_for_nn/ -r 0.2 -t yolov5 -s 640 -a -aim 4 -apldc 1
python traindata-creator/createDataset.py -n good-pics-v2-pld-only -tf traindata-creator/dataseries/af-the_good_pics_for_nn2_s1/ traindata-creator/dataseries/af-the_good_pics_for_nn2_s2/ -r 0.2 -t yolov5 -s 640 -a -aim 4 -apldc 1
python traindata-creator/createDataset.py -n good-pics-v2-slight-mat-only -tf traindata-creator/dataseries/af-the_good_pics_for_nn2_s1/ traindata-creator/dataseries/af-the_good_pics_for_nn2_s2/ -r 0.2 -t yolov5 -s 640 -a -aim 4 -apc 0.4 -aps 0.04 -arc 0.4 -ars 20
python traindata-creator/createDataset.py -n good-pics-v2-def-aug -tf traindata-creator/dataseries/af-the_good_pics_for_nn2_s1/ traindata-creator/dataseries/af-the_good_pics_for_nn2_s2/ -r 0.2 -t yolov5 -s 640 -a -aim 4 -asgsc 0.9 -apldc 0.6 -apc 0.6 -aps 0.08 -arc 0.9 -ars 45 -andrc 0.7 -arc2c 0.6 -almc 0 -alm2c 0 -agnc 0
python traindata-creator/createDataset.py -n good-pics-v2-rc-only -tf traindata-creator/dataseries/af-the_good_pics_for_nn2_s1/ traindata-creator/dataseries/af-the_good_pics_for_nn2_s2/ -r 0.2 -t yolov5 -s 640 -a -aim 4 -arc2c 1
python traindata-creator/createDataset.py -n good-pics-v2-lm-only-test -tf traindata-creator/dataseries/af-the_good_pics_for_nn2_s1/ traindata-creator/dataseries/af-the_good_pics_for_nn2_s2/ -r 0.2 -t yolov5 -s 640 -a -aim 4 -almc 1
python traindata-creator/createDataset.py -n good-pics-v2-lm2-only-test -tf traindata-creator/dataseries/af-the_good_pics_for_nn2_s1/ traindata-creator/dataseries/af-the_good_pics_for_nn2_s2/ -r 0.2 -t yolov5 -s 640 -a -aim 4 -alm2c 1
# Create Valsets
python traindata-creator/createDataset.py -n on-skin-valset -vf traindata-creator/dataseries/man-fPCS_on_skin/ -t yolov5 -s 640
python traindata-creator/createDataset.py -n on-skin-valset-v2 -vf traindata-creator/dataseries/man-fPCS_on_skin/ traindata-creator/dataseries/man-on_skin_v2/ -t yolov5 -s 640
python traindata-creator/createDataset.py -n on-skin-valset-v3-ensemble-test -vf traindata-creator/dataseries/man-on_skin_v3_ensemble_test/ -t yolov5 -s 640
python traindata-creator/createDataset.py -n on-skin-valset-v3-testset -vf traindata-creator/dataseries/man-on_skin_v3_test/ -t yolov5 -s 640
python traindata-creator/createDataset.py -n on-skin-valset-v3-ensemble-test -vf traindata-creator/dataseries/man-on_skin_v3_ensemble_test/ -t yolov5 -s 0
python traindata-creator/createDataset.py -n on-skin-valset-v3-testset -vf traindata-creator/dataseries/man-on_skin_v3_test/ -t yolov5 -s 0
# Good pics v3
python traindata-creator/createDataset.py -n good-pics-v3-no-aug -tf traindata-creator/dataseries/af-the_good_pics_for_nn2_s1/ traindata-creator/dataseries/af-the_good_pics_for_nn2_s2/ traindata-creator/dataseries/af-the_good_pics_for_nn3_s1/ traindata-creator/dataseries/af-the_good_pics_for_nn3_s2/ traindata-creator/dataseries/af-good-zimmer-v1/ -r 0.1 -t yolov5 -s 640
python traindata-creator/createDataset.py -n good-pics-v3-def-aug -tf traindata-creator/dataseries/af-the_good_pics_for_nn2_s1/ traindata-creator/dataseries/af-the_good_pics_for_nn2_s2/ traindata-creator/dataseries/af-the_good_pics_for_nn3_s1/ traindata-creator/dataseries/af-the_good_pics_for_nn3_s2/ traindata-creator/dataseries/af-good-zimmer-v1/ -r 0.1 -t yolov5 -s 640 -a -aim 4 -asgsc 0.9 -apldc 0.6 -apc 0.6 -aps 0.08 -arc 0.9 -ars 45 -andrc 0.7 -arc2c 0.6 -almc 0 -alm2c 0 -agnc 0
python traindata-creator/createDataset.py -n good-pics-v3-def2-aug -tf traindata-creator/dataseries/af-the_good_pics_for_nn2_s1/ traindata-creator/dataseries/af-the_good_pics_for_nn2_s2/ traindata-creator/dataseries/af-the_good_pics_for_nn3_s1/ traindata-creator/dataseries/af-the_good_pics_for_nn3_s2/ traindata-creator/dataseries/af-good-zimmer-v1/ -r 0.1 -t yolov5 -s 640 -a -aim 4 -asgsc 0.3 -apldc 0.3 -apc 0.3 -aps 0.05 -arc 0.4 -ars 128 -andrc 0.7 -arc2c 0.3 -almc 0 -alm2c 0 -agnc 0.05
# Good pics compare builds
python traindata-creator/createDataset.py -n good-pics-v1-no-aug -tf traindata-creator/dataseries/af-the_good_pics_for_nn/ -r 0.2 -t yolov5 -s 640
python traindata-creator/createDataset.py -n good-pics-v1-def-aug -tf traindata-creator/dataseries/af-the_good_pics_for_nn/ -r 0.2 -t yolov5 -s 640 -a -aim 4 -asgsc 0.9 -apldc 0.6 -apc 0.6 -aps 0.08 -arc 0.9 -ars 45 -andrc 0.7 -arc2c 0.6 -almc 0 -alm2c 0 -agnc 0
python traindata-creator/createDataset.py -n good-pics-v1-def2-aug -tf traindata-creator/dataseries/af-the_good_pics_for_nn/ -r 0.2 -t yolov5 -s 640 -a -aim 4 -asgsc 0.3 -apldc 0.3 -apc 0.3 -aps 0.05 -arc 0.4 -ars 128 -andrc 0.7 -arc2c 0.3 -almc 0 -alm2c 0 -agnc 0.05
python traindata-creator/createDataset.py -n good-pics-v2-no-aug -tf traindata-creator/dataseries/af-the_good_pics_for_nn2_s1/ traindata-creator/dataseries/af-the_good_pics_for_nn2_s2/ -r 0.2 -t yolov5 -s 640
python traindata-creator/createDataset.py -n good-pics-v2-def-aug -tf traindata-creator/dataseries/af-the_good_pics_for_nn2_s1/ traindata-creator/dataseries/af-the_good_pics_for_nn2_s2/ -r 0.2 -t yolov5 -s 640 -a -aim 4 -asgsc 0.9 -apldc 0.6 -apc 0.6 -aps 0.08 -arc 0.9 -ars 45 -andrc 0.7 -arc2c 0.6 -almc 0 -alm2c 0 -agnc 0
python traindata-creator/createDataset.py -n good-pics-v2-def2-aug -tf traindata-creator/dataseries/af-the_good_pics_for_nn2_s1/ traindata-creator/dataseries/af-the_good_pics_for_nn2_s2/ -r 0.2 -t yolov5 -s 640 -a -aim 4 -asgsc 0.3 -apldc 0.3 -apc 0.3 -aps 0.05 -arc 0.4 -ars 128 -andrc 0.7 -arc2c 0.3 -almc 0 -alm2c 0 -agnc 0.05

# ensemble run using cmd_tf
python -m cmd_tf -df traindata-creator/dataset/seg-red-rects/ -r xunet-red-rects -e 75
python -m cmd_tf -df traindata-creator/dataset/seg-red-rects/ -r sm-unet-red-rects -bs 8 -e 75
python -m cmd_tf -df traindata-creator/dataset/seg-good-pics-ratio-val/ -r sm-unet-aruco -bs 8 -e 100
python -m cmd_tf -df traindata-creator/dataset/seg-good-pics-ratio-val/ -r sm-linknet-aruco -bs 4 -e 100
python -m cmd_tf -df traindata-creator/dataset/seg-good-pics-ratio-val/ -r sm-fpn-aruco -bs 4 -e 100
python -m cmd_tf -df traindata-creator/dataset/seg-good-pics-ratio-val/ -r sm-psnet-aruco -bs 4 -e 100

# Train yolo
#python repos/yolov5/train.py --img 320 --batch 16 --epochs 50 --data traindata-creator/dataset/yolov5-red-rects/dataset/yolov5-red-rects.yaml --weights yolov5s.pt
python repos/yolov5/train.py --img 640 --batch 8 --epochs 100 --data traindata-creator/dataset/yolov5-good-pics-v2/yolov5-good-pics-v2.yaml --weights yolov5s.pt --hyp hyp.scratch.yaml 
# Test yolo
python repos/yolov5_evaluate.py -r test_old_aug_gpv2_whyp -t traindata-creator/dataset/yolov5-640-on-skin-valset/
#python prototype-yolov5/test_yolo_on_pics.py -m repos/yolov5/runs/train/exp/weights/best.pt -d N:\\Downloads\\Archives\\FabioBilder\\fPCS_on_skin
# Do it all yolo
python repos/yolov5_train_loop.py -n test-py-loop -d traindata-creator/dataset/yolov5-640-good-pics-v2-slight-mat-only/ --no-aug -e 2
python repos/yolov5_train_loop.py -n test-py-loop-3 -d /data/pcmd/dataset/yolov5-640-good-pics-v2-slight-mat-only/ -v /data/pcmd/dataset/yolov5-640-on-skin-valset-v2/ --no-aug -e 2
python repos/yolov5_train_loop.py -n test-py-loop-rw -d traindata-creator/dataset/yolov5-640-good-pics-v2-slight-mat-only/ -v traindata-creator/dataset/yolov5-640-on-skin-valset-v2/ --no-aug -e 2 -rw

# Update run outputs
python -m cmd_tf -df traindata-creator/dataset/seg-good-pics-ratio-val/ -r sm-unet-aruco -bs 8 -e 0
python -m cmd_tf -df traindata-creator/dataset/seg-good-pics-ratio-val/ -r sm-linknet-aruco -bs 4 -e 0
python -m cmd_tf -df traindata-creator/dataset/seg-good-pics-ratio-val/ -r sm-fpn-aruco -bs 4 -e 0

# Analyze valdata
python evaluation/analyze.py -av sm-fpn-aruco
python evaluation/analyze.py -av repos/evaldata/yolov5/test-yolov5-640-good-pics-v2-slight-mat-only-1/

# Docker
docker build -t pcmd:0.1 .
docker run -it -w /src -v ./traindata-creator:/data -- pcmd:0.1
python repos/yolov5_train_loop.py -n test-py-loop -d /data/dataset/yolov5-640-good-pics-v2-slight-mat-only/ --no-aug -e 2

# Classic
python prototype-cv/classic_pipe/classic.py 

# Batch train
python batch_train/yolov5.py -d /data/pcmd/dataset/ -t /data/pcmd/dataset/yolov5-640-on-skin-valset-v2/
python batch_train/yolov5.py -d /data/pcmd/dataset/noise-sgss/ -t /data/pcmd/dataset/yolov5-640-on-skin-valset-v2/ -o training/noise-sgs-ensemble -e 300
python batch_train/yolov5.py -d traindata-creator/dataset/ -t traindata-creator/dataset/yolov5-640-on-skin-valset-v2/

python -m cmd_tf -t -td traindata-creator/dataset/seg-640-on-skin-valset-v2/ -r sm-unet-aruco

# Just train everything a bit
#python -m cmd_tf -df traindata-creator/dataset/seg-red-rects/ -r sm-unet-red-rects -bs 8 -e 75
#python -m cmd_tf -df traindata-creator/dataset/seg-good-pics-ratio-val/ -r sm-unet-aruco -bs 8 -e 100
#python -m cmd_tf -df traindata-creator/dataset/seg-good-pics-ratio-val/ -r sm-linknet-aruco -bs 4 -e 100
#python -m cmd_tf -df traindata-creator/dataset/seg-good-pics-ratio-val/ -r sm-fpn-aruco -bs 4 -e 100
#python -m cmd_tf -df traindata-creator/dataset/seg-good-pics-ratio-val/ -r sm-psnet-aruco -bs 4 -e 100

# Test if output stays somewhat the same 
# python -m cmd_tf -df traindata-creator/dataset/seg-good-pics-ratio-val/ -r sm-unet-aruco-same1 -bs 8 -e 5
# python -m cmd_tf -df traindata-creator/dataset/seg-good-pics-ratio-val/ -r sm-unet-aruco-same2 -bs 8 -e 5

# Create ensemble datasets
for i in `seq 0 0.05 1`; do 
    python traindata-creator/createDataset.py -n gpv2-sgs-$i -tf traindata-creator/dataseries/af-the_good_pics_for_nn2_s1/ traindata-creator/dataseries/af-the_good_pics_for_nn2_s2/ -r 0.2 -t yolov5 -s 640 -a -aim 4 -asgsc $i -taf traindata-creator/dataset/sgss; 
done
#smart grid ensemble, 0 to 1 chance in 0.1 steps, 10 sets per step => 100 sets
for i in `seq 0 0.1 1`; do 
    for j in `seq 0 1 10`; do 
        python traindata-creator/createDataset.py -n gpv2-sgs-$i-p$j -tf traindata-creator/dataseries/af-the_good_pics_for_nn2_s1/ traindata-creator/dataseries/af-the_good_pics_for_nn2_s2/ -r 0.2 -t yolov5 -s 640 -a -aim 4 -asgsc $i -agnc 1 -taf traindata-creator/dataset/noise-sgss
    done
done
#rotation ensemble, 0 to 360deg in 18 steps (20 rot steps), 5 sets per step => 100 sets
for i in `seq 0 18 360`; do 
    for j in `seq 0 1 5`; do 
        python traindata-creator/createDataset.py -n gpv2-rot-$i-p$j -tf traindata-creator/dataseries/af-the_good_pics_for_nn2_s1/ traindata-creator/dataseries/af-the_good_pics_for_nn2_s2/ -r 0.2 -t yolov5 -s 640 -a -aim 4 -arc 1 -ars $i -agnc 1 -taf traindata-creator/dataset/_noise-rots
    done
done
#gp compare on server, 9 datasets, 10 noise sets per dataset => 90 sets
for i in `seq 0 1 5`; do 
    target_folder=/data/pcmd/dataset/_noise-gp-compare-v2
    no_aug="-a -aim 2 -agnc 1"
    weak_aug="-a -aim 4 -asgsc 0.3 -apldc 0.3 -apc 0.2 -aps 0.05 -arc 0.6 -ars 128 -andrc 0.4 -arc2c 0.3 -alm2c 0.05 -abdc 0.2 -agnc 0.2"
    strong_aug="-a -aim 4 -asgsc 0.9 -apldc 0.6 -apc 0.6 -aps 0.08 -arc 0.9 -ars 360 -andrc 0.7 -arc2c 0.6 -alm2c 0.2 -abdc 0.6 -agnc 0.2"
    
    # v1
    python traindata-creator/createDataset.py -n good-pics-v1-p$i-1-no-aug -tf /data/pcmd/dataseries/af-the_good_pics_for_nn/ -r 0.2 -t yolov5 -s 640 $no_aug -taf $target_folder
    python traindata-creator/createDataset.py -n good-pics-v1-p$i-3-strong-aug -tf /data/pcmd/dataseries/af-the_good_pics_for_nn/ -r 0.2 -t yolov5 -s 640 $strong_aug -taf $target_folder
    python traindata-creator/createDataset.py -n good-pics-v1-p$i-2-weak-aug -tf /data/pcmd/dataseries/af-the_good_pics_for_nn/ -r 0.2 -t yolov5 -s 640 $weak_aug -taf $target_folder
    # v2
    python traindata-creator/createDataset.py -n good-pics-v2-p$i-1-no-aug -tf /data/pcmd/dataseries/af-the_good_pics_for_nn2_s1/ /data/pcmd/dataseries/af-the_good_pics_for_nn2_s2/ -r 0.2 -t yolov5 -s 640 $no_aug -taf $target_folder
    python traindata-creator/createDataset.py -n good-pics-v2-p$i-3-strong-aug -tf /data/pcmd/dataseries/af-the_good_pics_for_nn2_s1/ /data/pcmd/dataseries/af-the_good_pics_for_nn2_s2/ -r 0.2 -t yolov5 -s 640 $strong_aug -taf $target_folder
    python traindata-creator/createDataset.py -n good-pics-v2-p$i-2-weak-aug -tf /data/pcmd/dataseries/af-the_good_pics_for_nn2_s1/ /data/pcmd/dataseries/af-the_good_pics_for_nn2_s2/ -r 0.2 -t yolov5 -s 640 $weak_aug -taf $target_folder
    # v3
    python traindata-creator/createDataset.py -n good-pics-v3-p$i-1-no-aug -tf /data/pcmd/dataseries/af-the_good_pics_for_nn2_s1/ /data/pcmd/dataseries/af-the_good_pics_for_nn2_s2/ /data/pcmd/dataseries/af-the_good_pics_for_nn3_s1/ /data/pcmd/dataseries/af-the_good_pics_for_nn3_s2/ /data/pcmd/dataseries/af-good-zimmer-v1/ -r 0.1 -t yolov5 -s 640 $no_aug -taf $target_folder
    python traindata-creator/createDataset.py -n good-pics-v3-p$i-3-strong-aug -tf /data/pcmd/dataseries/af-the_good_pics_for_nn2_s1/ /data/pcmd/dataseries/af-the_good_pics_for_nn2_s2/ /data/pcmd/dataseries/af-the_good_pics_for_nn3_s1/ /data/pcmd/dataseries/af-the_good_pics_for_nn3_s2/ /data/pcmd/dataseries/af-good-zimmer-v1/ -r 0.1 -t yolov5 -s 640 $strong_aug -taf $target_folder
    python traindata-creator/createDataset.py -n good-pics-v3-p$i-2-weak-aug -tf /data/pcmd/dataseries/af-the_good_pics_for_nn2_s1/ /data/pcmd/dataseries/af-the_good_pics_for_nn2_s2/ /data/pcmd/dataseries/af-the_good_pics_for_nn3_s1/ /data/pcmd/dataseries/af-the_good_pics_for_nn3_s2/ /data/pcmd/dataseries/af-good-zimmer-v1/ -r 0.1 -t yolov5 -s 640 $weak_aug -taf $target_folder
done

# Full ensemble workflow for black dot ensemble
#black_dot ensemble on mip server, 0 to 1 chance in 0.2 steps, 5 sets per step => 25 sets (IN DOCKER ENV)
for i in `seq 0 0.2 1`; do 
    for j in `seq 0 1 5`; do 
        python traindata-creator/createDataset.py -n gpv2-bd-$i-p$j -tf /data/pcmd/dataseries/af-the_good_pics_for_nn2_s1/ /data/pcmd/dataseries/af-the_good_pics_for_nn2_s2/ -r 0.2 -t yolov5 -s 640 -a -aim 3 -abdc $i -agnc 1 -taf /data/pcmd/dataset/_noise-bds
    done
done
python3 mip_worker_batch_train.py -n 6 -c "python batch_train/yolov5.py -d /data/pcmd/dataset/_noise-bds/ -t /data/pcmd/dataset/yolov5-640-on-skin-valset-v3-ensemble-test/ -e 300 -snr -o /data/pcmd/training/yolov5s-bd-ensemble/"
bash mip_worker_await.sh
with_gpu -n 1 sudo mip-docker-run --gpus '"device=$CUDA_VISIBLE_DEVICES"' ncarstensen/pcmd:0.11 python evaluation/plot_ensemble.py -n yolov5-noise-bd -r /data/pcmd/training/yolov5s-bd-ensemble/ -pi 5 -ci 4 -cu 10% -rnp '.*yolo5aug$' -t "mAP scores for a given chance of black dot augmentation in the dataset"

# Ensemble workflow for label curving
with_gpu -n 1 sudo mip-docker-run --gpus '"device=$CUDA_VISIBLE_DEVICES"' ncarstensen/pcmd:0.11 python mip_create_ensemble_datasets.py -n gpv2-curving -op "-tf /data/pcmd/dataseries/af-the_good_pics_for_nn2_s1/ /data/pcmd/dataseries/af-the_good_pics_for_nn2_s2/ -r 0.2 -t yolov5 -s 640 -alcc"
python3 mip_worker_batch_train.py -n 6 -c "python batch_train/yolov5.py -d /data/pcmd/dataset/_noise-ensemble-gpv2-curving/ -t /data/pcmd/dataset/yolov5-640-on-skin-valset-v3-ensemble-test/ -e 300 -snr -o /data/pcmd/training/yolov5s-curving-ensemble/"
bash mip_worker_await.sh
with_gpu -n 1 sudo mip-docker-run --gpus '"device=$CUDA_VISIBLE_DEVICES"' ncarstensen/pcmd:0.11 python evaluation/plot_ensemble.py -n yolov5s-noise-curving -r /data/pcmd/training/yolov5s-curving-ensemble/ -pi 5 -ci 4 -cu 10% -rnp '.*yolo5aug$' -t "mAP scores for a given chance of curving augmentation in the dataset"

# Advanced inference
python evaluation/fusion_inference.py -r evaluation/from-server/yolov5s-rot-ensemble/ -rnp yolov5-640-gpv2-rot-234-p[0-9]*-yolo5aug
python evaluation/analyze.py -av evaluation/from-server/yolov5s-rot-ensemble/yolov5-640-gpv2-rot-234-p0-yolo5aug/test_fused/
python evaluation/fusion_inference.py -r evaluation/from-server/noise-sgs-ensemble/ -rnp yolov5-640-gpv2-sgs-10-p[0-9]*-yolo5aug
python evaluation/analyze.py -av evaluation/from-server/noise-sgs-ensemble/yolov5-640-gpv2-sgs-10-p0-yolo5aug/test_fused/
python repos/yolov5_evaluate.py -r evaluation/from-server/yolov5s-rot-ensemble/yolov5-640-gpv2-rot-234-p1-yolo5aug/ -t traindata-creator/dataset/yolov5-0-on-skin-valset-v3-testset/ -us
python repos/yolov5_evaluate.py -r evaluation/from-server/yolov5s-rot-ensemble/yolov5-640-gpv2-rot-234-p1-yolo5aug/ -t traindata-creator/dataset/yolov5-0-on-skin-valset-v3-ensemble-test/ -dbo -us -bis 0.2 -sqt 0.6
python batch_train/yolov8_evaluate.py -r evaluation/from-server/yolov8s-runs/yolov5-640-good-pics-v2-slight-mat-only/ -t traindata-creator/dataset/yolov5-0-on-skin-valset-v3-ensemble-test/ -dbo -us -bis 0.2 -sqt 0.6 -tn test-script

# Run datasets ensemble on remote
python batch_train/yolov5.py -d /data/pcmd/dataset/sgss/ -t /data/pcmd/dataset/yolov5-640-on-skin-valset-v2/ -e 300 -o training/yolov5s-sgs-ensemble-test

# Eval plotting
python evaluation/plot_ensemble.py -r evaluation/from-server/first-yolov5s-runs/ -n yolov5s-runs -rns -1
python evaluation/plot_ensemble.py -r evaluation/from-server/yolov5m-runs/ -n yolov5m-runs
python evaluation/plot_ensemble.py -n yolov5s-sgs-ensemble -r evaluation/from-server/yolov5s-sgs-ensemble-test/
python evaluation/plot_ensemble.py -n yolov5s-sgs-ensemble-yoloaug -r evaluation/from-server/yolov5s-sgs-ensemble-test/ -rnp '.*yolo5aug$'
python evaluation/plot_ensemble.py -n yolov5s-sgs-ensemble-yoloaug -r evaluation/from-server/yolov5s-sgs-ensemble-test/ -rnp '.*yolo5aug$' -bfl -ci 4 -cu % -xl 'Probability for each image in the dataset to be augmented by smart grid shuffeling'
python evaluation/plot_ensemble.py -n yolov5s-sgs-ensemble-no-yoloaug -r evaluation/from-server/yolov5s-sgs-ensemble-test/ -rnp '.*(?<!yolo5aug)$'
python evaluation/plot_ensemble.py -n yolov5-noise-sgs-ensemble -r evaluation/from-server/noise-sgs-ensemble/ -pi 5 -ci 4 -rnp '.*(?<!yolo5aug)$'
python evaluation/plot_ensemble.py -n yolov5-noise-sgs-ensemble-yolov5aug -r evaluation/from-server/noise-sgs-ensemble/ -pi 5 -ci 4 -rnp '.*yolo5aug$'
python evaluation/plot_ensemble.py -n yolov5-noise-sgs-ensemble-yolov5aug -r evaluation/from-server/noise-sgs-ensemble/ -pi 5 -ci 4 -rnp '.*yolo5aug$' -t "mAP scores for a given chance of smart grid shuffle augmentation in the dataset"
python evaluation/plot_ensemble.py -n yolov5-noise-sgs-ensemble-yolov5aug -r evaluation/from-server/noise-sgs-ensemble/ -pi 5 -ci 4 -rnp '.*yolo5aug$' -t "mAP scores for a given chance of smart grid shuffle augmentation in the dataset" -bfl -cu 10% -xl 'Probability for each image in the dataset to be augmented by smart grid shuffeling' -ct box
python evaluation/plot_ensemble.py -n yolov5-noise-gp-tv3old -r evaluation/from-server/noise-sgs-ensemble/ -pi 5 -rnp '.*yolo5aug$' -t "mAP scores for a given chance of smart grid shuffle augmentation in the dataset"

# Worker ensemble
with_gpu -n 1 sudo mip-docker-run --rm --gpus '"device=$CUDA_VISIBLE_DEVICES"' ncarstensen/pcmd:0.1 python batch_train/yolov5.py -d /data/pcmd/dataset/ -t /data/pcmd/dataset/yolov5-640-on-skin-valset-v2/ -e 10 -o /data/pcmd/training/worker_test/ -wi 0 -wc 2
python3 mip_worker_batch_train.py -c "python batch_train/yolov5.py -d /data/pcmd/dataset/ -t /data/pcmd/dataset/yolov5-640-on-skin-valset-v2/ -e 10 -o /data/pcmd/training/worker_test/"
python3 mip_worker_batch_train.py -n 6 -c "python batch_train/yolov5.py -d /data/pcmd/dataset/_noise-rots/ -t /data/pcmd/dataset/yolov5-640-on-skin-valset-v2/ -e 300 -snr -o /data/pcmd/training/yolov5s-rot-ensemble/"
python3 mip_worker_batch_train.py -n 9 -c "python batch_train/yolov5.py -d /data/pcmd/dataset/_gp-compare/ -t /data/pcmd/dataset/yolov5-640-on-skin-valset-v2/ -e 300 -snr -o /data/pcmd/training/yolov5s-gp-ensemble/"
python3 mip_worker_batch_train.py -n 6 -c "python batch_train/yolov5.py -d /data/pcmd/dataset/_noise-gp-compare/ -t /data/pcmd/dataset/yolov5-640-on-skin-valset-v3-ensemble-test/ -e 300 -snr -o /data/pcmd/training/yolov5s-gp-ensemble-tv3/"

# hyp param search
python hyperparam_search.py -t /data/pcmd/dataset/yolov5-0-on-skin-valset-v3-ensemble-test/ -df /data/pcmd/dataset/hyp-param-search -tf /data/pcmd/training/hyp-param-search -mine 200 -maxe 400 -ds " -tf /data/pcmd/dataseries/af-the_good_pics_for_nn2_s1/ /data/pcmd/dataseries/af-the_good_pics_for_nn2_s2/ -r 0.2" -emax 300
python traindata-creator/createDataset.py -n best-yolov5s-hyp-search-set -s 640 -tf traindata-creator/dataseries/af-the_good_pics_for_nn2_s1/ traindata-creator/dataseries/af-the_good_pics_for_nn2_s2/ -sd 8338 -r 0.2 -t yolov5 -s 640 -a -aim 4 -asgsc 0.16536455065004735 -apldc 0.3406092762590903 -apc 0.9026868390392013 -aps 0.44744759491769104 -arc 0.8397534486075489 -ars 269.5297433583759 -andrc 0.4779575209987885 -arcc 0.07811140975400804 -arc2c 0.09484458554495206 -almc 0.3281069403856025 -alm2c 0.04946212899649677 -abdc 0.5932304638260782 -alcc 
0.09002886339102176 -agnc 0.2850102716939429 -agns 147.09472782083168
