from sahi.predict import get_sliced_prediction
from sahi import AutoDetectionModel

detection_model = AutoDetectionModel.from_pretrained(model_type='yolov5', model_path='evaluation/from-server/yolov5s-rot-ensample/yolov5-640-gpv2-rot-234-p1-yolo5aug/weights/best.pt') # for YOLOv5 models

# get sliced prediction result
result = get_sliced_prediction(
    'traindata-creator/dataseries/man-on_skin_v2/PXL_20230707_080009486_in.png',
    detection_model,
    slice_height = 640,
    slice_width = 640,
    overlap_height_ratio = 0.2,
    overlap_width_ratio = 0.2
)

print(result)
result.export_visuals('prototype-yolo/sahi_export/')
for pred in result.object_prediction_list:
    print(pred.score.value, pred.bbox.minx, pred.bbox.miny, pred.bbox.maxx, pred.bbox.maxy)