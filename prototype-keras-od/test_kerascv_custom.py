import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers
import keras_cv
import numpy as np
from keras_cv import bounding_box
import os
import resource
from keras_cv import visualization
import tqdm
import tensorflow_datasets as tfds

BATCH_SIZE = 4
IMG_W = 320
IMG_H = 320

class_mapping = { 1: '1' }

def visualize_dataset(inputs, value_range, rows, cols, bounding_box_format):
    inputs = next(iter(inputs.take(1)))
    images, bounding_boxes = inputs["images"], inputs["bounding_boxes"]
    visualization.plot_bounding_box_gallery(
        images,
        value_range=value_range,
        rows=rows,
        cols=cols,
        y_true=bounding_boxes,
        scale=5,
        font_scale=0.7,
        bounding_box_format=bounding_box_format,
        class_mapping=class_mapping,
    )


def unpackage_raw_tfds_inputs(inputs, bounding_box_format):
    image = inputs["image"]
    boxes = keras_cv.bounding_box.convert_format(
        inputs["objects"]["bbox"],
        images=image,
        source="rel_yxyx",
        target=bounding_box_format,
    )
    bounding_boxes = {
        "classes": tf.cast(inputs["objects"]["label"], dtype=tf.float32),
        "boxes": tf.cast(boxes, dtype=tf.float32),
    }
    return {"images": tf.cast(image, tf.float32), "bounding_boxes": bounding_boxes}


def load_pascal_voc(split, dataset, bounding_box_format):
    ds = tfds.load(dataset, split=split, with_info=False, shuffle_files=True)
    ds = ds.map(
        lambda x: unpackage_raw_tfds_inputs(x, bounding_box_format=bounding_box_format),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return ds


train_ds = load_pascal_voc(split="train", dataset="tf_dataset", bounding_box_format="xywh")
eval_ds  = load_pascal_voc(split="test", dataset="tf_dataset", bounding_box_format="xywh")

print(len(train_ds))
print(len(eval_ds))
print(train_ds)

train_ds = train_ds.shuffle(BATCH_SIZE * 4)
eval_ds = eval_ds.shuffle(BATCH_SIZE * 4)

print(len(train_ds))
print(len(eval_ds))
print(train_ds)

train_ds = train_ds.ragged_batch(BATCH_SIZE, drop_remainder=True)
eval_ds = eval_ds.ragged_batch(BATCH_SIZE, drop_remainder=True)

print(len(train_ds))
print(len(eval_ds))
print(train_ds)

print("train_ds")
visualize_dataset(
    train_ds, 
    bounding_box_format="xywh",
    value_range=(0, 255),
    rows=2,
    cols=2,
)

print("eval_ds")
visualize_dataset(
    eval_ds,
    bounding_box_format="xywh",
    value_range=(0, 255),
    rows=2,
    cols=2,
)

# augmenter = keras.Sequential(
#     layers=[
#         keras_cv.layers.RandomFlip(mode="horizontal", bounding_box_format="xywh"),
#         keras_cv.layers.JitteredResize(
#             target_size=(IMG_W, IMG_H), scale_factor=(0.75, 1.3), bounding_box_format="xywh"
#         ),
#     ]
# )
# train_ds = train_ds.map(augmenter, num_parallel_calls=tf.data.AUTOTUNE)
# print("train_ds augmenter")
# visualize_dataset(
#     train_ds, bounding_box_format="xywh", value_range=(0, 255), rows=2, cols=2
# )

#inference_resizing = keras_cv.layers.Resizing(
#    IMG_W, IMG_H, bounding_box_format="xywh", pad_to_aspect_ratio=True
#)
#train_ds = train_ds.map(inference_resizing, num_parallel_calls=tf.data.AUTOTUNE)
#eval_ds = eval_ds.map(inference_resizing, num_parallel_calls=tf.data.AUTOTUNE)

def dict_to_tuple(inputs):
    return inputs["images"], bounding_box.to_dense(
        inputs["bounding_boxes"], max_boxes=32
    )

train_ds = train_ds.map(dict_to_tuple, num_parallel_calls=tf.data.AUTOTUNE)
eval_ds = eval_ds.map(dict_to_tuple, num_parallel_calls=tf.data.AUTOTUNE)

train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
eval_ds = eval_ds.prefetch(tf.data.AUTOTUNE)

base_lr = 0.005
# including a global_clipnorm is extremely important in object detection tasks
optimizer = tf.keras.optimizers.SGD(
    learning_rate=base_lr, momentum=0.9, global_clipnorm=10.0
)

coco_metrics = keras_cv.metrics.BoxCOCOMetrics(
    bounding_box_format="xywh", evaluate_freq=20
)

def print_metrics(result):
    maxlen = max([len(key) for key in result.keys()])
    print("Metrics:")
    print("-" * (maxlen + 1))
    for k, v in result.items():
        print(f"{k.ljust(maxlen+1)}: {v.numpy():0.2f}")
    
model = keras_cv.models.RetinaNet.from_preset(
    "resnet50_imagenet",
    num_classes=len(class_mapping),
    bounding_box_format="xywh",
)

model.compile(
    classification_loss="focal",
    box_loss="smoothl1",
    optimizer=optimizer,
    metrics=None,
)

model.fit(
    train_ds.take(5),
    validation_data=eval_ds.take(5),
    # Run for 10-35~ epochs to achieve good scores.
    epochs=1,
    #callbacks=[EvaluateCOCOMetricsCallback(eval_ds.take(20))],
)

model = keras_cv.models.RetinaNet.from_preset(
    "retinanet_resnet50_pascalvoc", bounding_box_format="xywh"
)

visualization_ds = eval_ds.unbatch()
visualization_ds = visualization_ds.ragged_batch(16)
visualization_ds = visualization_ds.shuffle(8)

def visualize_detections(model, dataset, bounding_box_format):
    images, y_true = next(iter(dataset.take(1)))
    y_pred = model.predict(images)
    y_pred = bounding_box.to_ragged(y_pred)
    visualization.plot_bounding_box_gallery(
        images,
        value_range=(0, 255),
        bounding_box_format=bounding_box_format,
        y_true=y_true,
        y_pred=y_pred,
        scale=4,
        rows=2,
        cols=4,
        show=True,
        font_scale=0.7,
        class_mapping=class_mapping,
    )

model.prediction_decoder = keras_cv.layers.MultiClassNonMaxSuppression(
    bounding_box_format="xywh",
    from_logits=True,
    iou_threshold=0.5,
    confidence_threshold=0.75,
)

visualize_detections(model, dataset=visualization_ds, bounding_box_format="xywh")

