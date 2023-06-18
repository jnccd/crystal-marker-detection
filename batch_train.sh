# Just train everything a bit
#python -m cmd_tf -df traindata-creator/dataset-seg-red-rects/ -r sm-unet-red-rects -bs 8 -e 75
#python -m cmd_tf -df traindata-creator/dataset-seg-good-pics-ratio-val/ -r sm-unet-aruco -bs 8 -e 100
#python -m cmd_tf -df traindata-creator/dataset-seg-good-pics-ratio-val/ -r sm-linknet-aruco -bs 4 -e 100
#python -m cmd_tf -df traindata-creator/dataset-seg-good-pics-ratio-val/ -r sm-fpn-aruco -bs 4 -e 100
#python -m cmd_tf -df traindata-creator/dataset-seg-good-pics-ratio-val/ -r sm-psnet-aruco -bs 4 -e 100

# Test if output stays somewhat the same 
python -m cmd_tf -df traindata-creator/dataset-seg-good-pics-ratio-val/ -r sm-unet-aruco-same1 -bs 8 -e 5
python -m cmd_tf -df traindata-creator/dataset-seg-good-pics-ratio-val/ -r sm-unet-aruco-same2 -bs 8 -e 5