import numpy as np
import tensorflow_datasets as tfds

(train_dataset, val_dataset), dataset_info = tfds.load(
    "coco/2017", split=["train", "validation"], with_info=True, data_dir="data"
)

print("train_dataset",train_dataset)
print("val_dataset",val_dataset)
print("train_dataset.shape",train_dataset.shape)
print("val_dataset.shape",val_dataset.shape)