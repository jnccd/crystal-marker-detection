"""tf_dataset dataset."""

import os
from pathlib import Path
import random
from numpy import float32, int64, uint8
import tensorflow_datasets as tfds
import uuid

class Builder(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for tf_dataset dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""

    return self.dataset_info_from_configs(
        features=tfds.features.FeaturesDict({
            'image': tfds.features.Image(shape=(320, 320, 3), encoding_format='png', dtype=uint8),
            'image/id': int64,
            'objects': tfds.features.Sequence({
                'area': int64,
                'bbox': tfds.features.BBoxFeature(),
                'id': int64,
                'label': tfds.features.ClassLabel(names=['1']),
    }),
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        supervised_keys=('image', 'bbox', 'label'),  # Set to `None` to disable
        homepage='https://dataset-homepage/',
    )
  
  def _get_all_filepaths(self, paths, file_ending):
    re = []
    for path in paths:
      re.extend(sorted(
          [
              os.path.join(path, fname)
              for fname in os.listdir(path)
              if fname.endswith(file_ending)
          ]
      ))
    return re

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""

    root_dir = Path(__file__).resolve().parent
    dataseries_t1_dir = root_dir/'..'/'dataseries-320-webcam-images-1312ecab-04e7-4f45-a714-07365d8c0dae'/'images_traindata'
    dataseries_t2_dir = root_dir/'..'/'dataseries-320-webcam-images-f50ec0b7-f960-400d-91f0-c42a6d44e3d0'/'images_traindata'
    dataseries_v1_dir = root_dir/'..'/'dataseries-320-webcam-images-203d3683-7c91-4429-93b6-be24a28f47bf'/'images_traindata'

    return {
        'train': self._generate_examples(self._get_all_filepaths([dataseries_t1_dir, dataseries_t2_dir], ".png")),
        'test': self._generate_examples(self._get_all_filepaths([dataseries_v1_dir], ".png")),
    }
  
  def _generate_examples(self, paths):
    """Yields examples."""
    
    counter = 0
    for img_spath in paths:

      img_path = Path(img_spath)
      txt_path = img_path.parent / (str(img_path.stem)+'_yolo.txt')

      objs = []
      with open(txt_path) as f:
          for row in f.readlines():
              (class_label, x, y, w, h) = row.removesuffix("\n").split(" ")
              objs.append( { 
                'label': class_label, 
                'area': float(w)*float(h), 
                'id': random.getrandbits(63),
                'bbox': tfds.features.BBox(ymin=float(y), 
                                           xmin=float(x), 
                                           ymax=float(y)+float(h), 
                                           xmax=float(x)+float(w)) } )

      yield counter, {
          'image': img_spath,
          'image/id': random.getrandbits(63),
          'objects': objs,
      }
      print(counter, img_spath)
      counter+=1
