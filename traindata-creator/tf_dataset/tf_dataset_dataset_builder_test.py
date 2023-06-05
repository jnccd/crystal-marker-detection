"""tf_dataset dataset."""

import tensorflow_datasets as tfds
from . import tf_dataset_dataset_builder


class TfDatasetTest(tfds.testing.DatasetBuilderTestCase):
  """Tests for tf_dataset dataset."""
  # TODO(tf_dataset):
  DATASET_CLASS = tf_dataset_dataset_builder.Builder
  SPLITS = {
      'train': 20,  # Number of fake train example
      'test': 5,  # Number of fake test example
  }

  # If you are calling `download/download_and_extract` with a dict, like:
  #   dl_manager.download({'some_key': 'http://a.org/out.txt', ...})
  # then the tests needs to provide the fake output paths relative to the
  # fake data directory
  # DL_EXTRACT_RESULT = {'some_key': 'output_file1.txt', ...}


if __name__ == '__main__':
  tfds.testing.test_main()
