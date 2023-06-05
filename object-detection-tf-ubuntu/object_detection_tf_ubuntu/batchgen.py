import numpy as np
from tensorflow import keras
from tensorflow.keras.utils import load_img

class ResnetBatchgen(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, img_size, input_img_paths, target_paths, cur_conf):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_paths = target_paths
        self.cur_conf = cur_conf

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        batch_target_paths = self.target_paths[i : i + self.batch_size]
        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        for j, path in enumerate(batch_input_img_paths):
            x[j] = load_img(path, target_size=self.img_size)
        y = np.zeros((self.batch_size,) + 5, dtype="uint8")
        for j, path in enumerate(batch_target_paths):
            
            
            
            y[j] = load_img(path, target_size=self.img_size)
        return x, y