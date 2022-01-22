import time

import tensorflow as tf
import pandas as pd
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description='Select between small or big data',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-d', '--data-size', type=str, choices=['big', 'small', 'real'], default='small')

class Dataset:
    def __init__(self, train_path: str, val_path: str,
                 img_directory: str, input_shape: tuple):

        self.MAX_VALUE = 30.0
        self.input_shape = input_shape
        self.image_dir = img_directory

        self.train_path = train_path
        self.train_df = pd.read_csv(train_path)
        self.train_df = self.preprocess_label(self.train_df)

        self.val_path = val_path
        self.val_df = pd.read_csv(val_path)
        self.val_df = self.preprocess_label(self.val_df)

        self.test_df = pd.read_csv(val_path)

        self.data_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

    def preprocess_label(self, train_df: pd.DataFrame):
        train_df["x"] = train_df["x"].div(self.MAX_VALUE)
        train_df["y"] = train_df["y"].div(self.MAX_VALUE)
        train_df["z"] = train_df["z"].div(self.MAX_VALUE)
        return train_df

    def generate_dataiterator(self, data_split='train'):
        if data_split=='train':
            df = self.train_df
        elif data_split=='val':
            df = self.val_df
        else: # For test set
            df = pd.read_csv(self.val_path)


        self.data_iterator = self.data_gen.flow_from_dataframe(
            dataframe=df,
            directory=self.image_dir,
            x_col="img",
            y_col=["x", "y", "z"],
            target_size=self.input_shape,
            color_mode="grayscale",
            class_mode="raw",
            batch_size=1, shuffle=True)

        # if parallel:
        #     data_generator = DataFrameGenerator(data_loader)
        #     tf_dataset = tf.data.Dataset.from_generator(data_generator.__iter__, output_types=(tf.float32, tf.float32))
        #     tf_dataset = tf_dataset.map(lambda x, y : (x, y), num_parallel_calls=4)
        #     return tf_dataset.it

        return self.data_iterator

class DataLoader:
    def __init__(self, dataset: Dataset, split='train'):
        self.dataset = dataset
        self.dataset.generate_dataiterator(split)

    def generator(self):
        for (images, labels) in self.dataset.data_iterator:
            yield images, labels

    def make(self, batch_size: int):
        tf_dataset = tf.data.Dataset.from_generator(self.generator, output_types=(tf.float32, tf.float32))
        tf_dataset = tf_dataset.map(lambda x, y: (tf.squeeze(x, axis=0), tf.squeeze(y, axis=0)), num_parallel_calls=8)
        tf_dataset = tf_dataset.batch(batch_size)
        return tf_dataset


if __name__ == '__main__':
    args = parser.parse_args()

    input_shape = (224, 398)

    if args.data_size=='big':
        train_path = "/home/ivsr/CV_Group/phuc/airsim/train.csv"
        val_path = "/home/ivsr/CV_Group/phuc/airsim/val.csv"
        img_directory = "/home/ivsr/CV_Group/phuc/airsim/data"
    else:
        train_path = "/home/ivsr/CV_Group/minh/train588_50.csv"
        val_path = "/home/ivsr/CV_Group/minh/val588_50.csv"
        img_directory = "/home/ivsr/CV_Group/minh/50imperpose/full"


    dataset = Dataset(train_path, val_path, img_directory, input_shape)
    train_loader = DataLoader(dataset, 'val')
    batch_loader = train_loader.make(32)

    begin = time.time()
    for batch_id, (images, labels) in enumerate(tqdm(batch_loader, colour='#c22c4e')):
        pass
    print(f'Process: {time.time()-begin} (s)')
    # train_loader = dataset.generate_dataloader('train')
    # val_loader = dataset.generate_dataloader('val')


    # next_batch = next(val_loader)
    # print(next_batch[0].shape) # batch_size, h, w, 1
    # print(next_batch[1].shape) # batch_size, 3
    #
    # next_batch = next(val_loader)
    # print(next_batch[0].shape) # batch_size, h, w, 1
    # print(next_batch[1].shape) # batch_size, 3

    # train_loader = DataLoader(dataset, 'val')
    # tf_dataset = tf.data.Dataset.from_generator(train_loader.generator, output_types=(tf.float32, tf.float32))
    # tf_dataset = tf_dataset.map(lambda x, y: (tf.squeeze(x, axis=0), tf.squeeze(y, axis=0)), num_parallel_calls=4)
    # tf_dataset = tf_dataset.batch(320)
    # for i, (X, y) in enumerate(tf_dataset):
    #     print(X.shape)
    #     print(i)
    #print(next(iter(train_loader.generator())))  # batch_size, h, w, 1; batch_size, 3