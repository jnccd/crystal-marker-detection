import yolov8_evaluate 
import pathlib
from ultralytics import YOLO

model = YOLO('../training/yolov8/yolov5-640-good-pics-v1-no-aug/weights/best.pt')
valset_path = pathlib.Path('../traindata-creator/dataset/yolov5-640-on-skin-valset-v2/')
out_testdata_path = pathlib.Path('../training/yolov8/yolov5-640-good-pics-v1-no-aug/test')

# Invoke directly
yolov8_evaluate.gen_evaldata(model, valset_path, out_testdata_path)