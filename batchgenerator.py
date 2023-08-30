import numpy as np
import tensorflow as tf


class BatchGenerator(tf.keras.utils.Sequence):

    def __init__(self, x_col, y_col, batch_size=100, shuffle=False):
        self.x_col = x_col
        self.y_col = y_col
        self.batch_size = batch_size
        self.key_array = np.arange(self.x_col.shape[0])
        self.shuffle = shuffle

    def __len__(self):
        return len(self.key_array) // self.batch_size  # return number of batches

    def __getitem__(self, index):
        keys = self.key_array[index * self.batch_size: (index + 1) * self.batch_size]
        x = np.asarray(self.x_col[keys])
        y = np.asarray(self.y_col[keys])

        return x, y, keys

    def on_epoch_end(self):
        if self.shuffle:
            self.key_array = np.random.permutation(self.key_array)
