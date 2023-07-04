SECONDS=0

# Just train everything a bit
#python -m cmd_tf -df traindata-creator/dataset/seg-red-rects/ -r sm-unet-red-rects -bs 8 -e 75
#python -m cmd_tf -df traindata-creator/dataset/seg-good-pics-ratio-val/ -r sm-unet-aruco -bs 8 -e 100
#python -m cmd_tf -df traindata-creator/dataset/seg-good-pics-ratio-val/ -r sm-linknet-aruco -bs 4 -e 100
#python -m cmd_tf -df traindata-creator/dataset/seg-good-pics-ratio-val/ -r sm-fpn-aruco -bs 4 -e 100
#python -m cmd_tf -df traindata-creator/dataset/seg-good-pics-ratio-val/ -r sm-psnet-aruco -bs 4 -e 100

# Test if output stays somewhat the same 
# python -m cmd_tf -df traindata-creator/dataset/seg-good-pics-ratio-val/ -r sm-unet-aruco-same1 -bs 8 -e 5
# python -m cmd_tf -df traindata-creator/dataset/seg-good-pics-ratio-val/ -r sm-unet-aruco-same2 -bs 8 -e 5

for dir in traindata-creator/dataset/yolov5-*/; do
    dirname=$(basename "$dir")

    if [[ $dirname == *"-valset" ]]; then
        echo "I wont test $dirname"
        continue
    fi
    
    cat traindata-creator/dataset/$dirname/dataset-def.json 

    #python repos/yolov5_train_loop.py -n test-$dirname-1 -d $dirname -e 300 --no-aug
    #python repos/yolov5_train_loop.py -n test-$dirname-2 -d $dirname -e 300 --no-aug
    #python repos/yolov5_train_loop.py -n test-$dirname-3 -d $dirname -e 300 --no-aug
    python repos/yolov5_train_loop.py -n test-$dirname-yolo5aug-1 -d $dirname -e 300
    #python repos/yolov5_train_loop.py -n test-$dirname-yolo5aug-2 -d $dirname -e 300
    #python repos/yolov5_train_loop.py -n test-$dirname-yolo5aug-3 -d $dirname -e 300
 
done

duration=$SECONDS
echo "Batch training took $(($duration / 60)) minutes and $(($duration % 60)) seconds."