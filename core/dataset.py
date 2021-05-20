import numpy as np

import pandas as pd
import tensorflow as tf
import os

import faulthandler;
faulthandler.enable()


class DataLoader():
    def __init__(self, data_root, data_gen, input_shape=(224, 224, 1),normalized_output=True, batch_size=8, split='train'):
        self.input_shape= input_shape
        self.csv_path = os.path.join(data_root, split+'.csv')
        self.data_dir = os.path.join(data_root, split)
        self.batch_size = batch_size

        self.df = pd.read_csv(self.csv_path)
        self.normalized_output = normalized_output

        self.data_gen = data_gen
        self.batch_gen = None

    def normalize_output(self, scale_down: float):
        self.df['x'] = self.df['x']/scale_down
        self.df['y'] = self.df['y']/scale_down
        self.df['z'] = self.df['z']/scale_down

        return self.df

    def batch_loader(self):
        if self.normalized_output:
            self.df = self.normalize_output(100)

        self.batch_gen = self.data_gen.flow_from_dataframe(dataframe=self.df,
                                                       directory=self.data_dir,
                                                       x_col='img',
                                                       y_col=['x', 'y', 'z'],
                                                       target_size=(self.input_shape[0], self.input_shape[1]),
                                                       color_mode='grayscale',
                                                       class_mode='raw',
                                                       batch_size=self.batch_size
                                                       )
        return self.batch_gen


if __name__ == '__main__':
    data_gen = tf.keras.preprocessing.image.ImageDataGenerator()
    train_data = DataLoader('../IVSR_BINARY_DATA/50imperpose/data/', data_gen, split='train')
    val_data = DataLoader('../IVSR_BINARY_DATA/50imperpose/data/', data_gen, split='val')

    train_batch = train_data.batch_loader()
    val_batch = val_data.batch_loader()