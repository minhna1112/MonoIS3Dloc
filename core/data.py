import numpy as np
#import pandas as pd
import cudf
import os
import torch
from path import Path

def plot_3d():
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    train_path = "./data/train.csv"
    train_df = pd.read_csv(train_path)
    x = train_df["x"]
    y = train_df["y"]
    z = train_df["z"]

    surf = ax.scatter(x, y, z)

    val_path = "./data/val.csv"
    val_df = pd.read_csv(val_path)

    x = val_df["x"]
    y = val_df["y"]
    z = val_df["z"]

    # surf = ax.scatter(x, y, z, c="red")

    plt.show()


class DataLoader():
    def __init__(self, path_to_dataset, split='train'):
        self.df = cudf.read_csv(path_to_dataset)
        #self.df = pd.read_csv(path_to_dataset)

    def __length__(self):
        return len(self.df)

    def __getitem__(self, index):


if __name__ == '__main__':
    data_gen = tf.keras.preprocessing.image.ImageDataGenerator()
    train_data = DataLoader('../IVSR_BINARY_DATA/50imperpose/data/',split='train')
    val_data = DataLoader('../IVSR_BINARY_DATA/50imperpose/data/',  split='val')

    train_batch = train_data.batch_loader()
    val_batch = val_data.batch_loader()