import keras_retinanet
from keras_retinanet.preprocessing import csv_generator
from keras_retinanet.models import backbone
import keras_retinanet.losses as losses
from tensorflow import keras

train_csv_path = 'traindata-creator/dataset-csv-good-pics-ratio-val/train.csv'
val_csv_path = 'traindata-creator/dataset-csv-good-pics-ratio-val/val.csv'
classes_csv_path = 'traindata-creator/dataset-csv-good-pics-ratio-val/classes.csv'

train_gen = csv_generator.CSVGenerator(train_csv_path, classes_csv_path)
val_gen = csv_generator.CSVGenerator(val_csv_path, classes_csv_path)

model = backbone('resnet50').retinanet(num_classes=1)

model.compile(
    loss={
        'regression'    : losses.smooth_l1(),
        'classification': losses.focal()
    },
    optimizer=keras.optimizers.Adam(lr=1e-5, clipnorm=0.001)
)

model_out = model.fit(train_gen, epochs=5, validation_data=val_gen, verbose=1)
