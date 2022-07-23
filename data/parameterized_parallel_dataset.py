import time
import random
import tensorflow as tf
import pandas as pd
from tqdm import tqdm
from path import Path

import argparse
import os
import warnings

from keras_preprocessing.image.utils import validate_filename

parser = argparse.ArgumentParser(description='Select between small or big data',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-d', '--data-size', type=str, choices=['big', 'small', 'real'], default='small')




class Dataset:
    def __init__(self, train_path: str,
                 img_directory: str, input_shape: tuple):

        self.input_shape = input_shape
        self.image_dir = Path(img_directory)

        self.train_path = train_path
        self.train_df = pd.read_csv(train_path)
        
        self.MAX_X = self.train_df["x"].max()
        self.MAX_Y = self.train_df["y"].max()
        self.MAX_Z = self.train_df["z"].max()
        self.MAX_VALUE = max([self.MAX_X, self.MAX_Y, self.MAX_Z ])
        print(f"Maximum values of x = {self.MAX_X}, y = {self.MAX_Y}, z = {self.MAX_Z}")
        
        self.train_df = self.preprocess_label(self.train_df)
        self.white_list_formats  = ('png', 'jpg', 'jpeg', 'bmp', 'ppm', 'tif', 'tiff')


    def preprocess_label(self, train_df: pd.DataFrame):      
        train_df["x"] = train_df["x"].div(self.MAX_VALUE)
        train_df["y"] = train_df["y"].div(self.MAX_VALUE)
        train_df["z"] = train_df["z"].div(self.MAX_VALUE)
        return train_df

    def __len__(self):
        return len(self.train_df)

    def __getitem__(self, index):
        X = self.image_dir / self.train_df['img'].iat[index]
        y = self.train_df[['x', 'y', 'z']].iloc[index]
        derived_label = self.train_df['derived_label'].iat[index]
        return X, y, derived_label

    
    def _filter_valid_filepaths(self, x_col):
        """Keep only dataframe rows with valid filenames

        # Arguments
            df: Pandas dataframe containing filenames in a column
            x_col: string, column in `df` that contains the filenames or filepaths

        # Returns
            absolute paths to image files
        """
        filepaths = self.train_df [x_col].map(
            lambda fname: os.path.join(self.image_dir, fname)
        )
        # print(filepaths.iloc[921])
        print("Validating filenames ... ... ...")
        mask = filepaths.apply(validate_filename, args=(self.white_list_formats,))
        # print(mask)
        n_invalid = (~mask).sum()
        print(filepaths[~mask])
        if n_invalid:
            warnings.warn(
                'Found {} invalid image filename(s) in x_col="{}". '
                'These filename(s) will be ignored.'
                .format(n_invalid, x_col)
            )
        self.train_df = self.train_df[mask]
        return self.train_df


class DataLoader:
    def __init__(self, dataset: Dataset, input_shape, batch_size: int, shuffle=True, num_parallel_calls = 4, validate=False):
        self.dataset = dataset
        self.input_shape = input_shape
        #self.dataset.generate_dataiterator(split)
        self.n = len(self.dataset)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_parallel_calls = num_parallel_calls

        if validate is True:
            self.dataset._filter_valid_filepaths('img')

    def generator(self):
        indices = range(len(self.dataset))
        if self.shuffle:
            indices = random.sample(indices, len(indices))
        for i in indices:
            # yield a tuple of image and corresponding positions labels
            yield self.dataset[i]

    def to_tensor(self, x, y, derived):
        image = tf.io.read_file(x)
        image = tf.io.decode_jpeg(image, channels=1)
        image = tf.image.resize(image, self.input_shape)
        image /= 255.0  # normalize to [0,1] range
        image = tf.convert_to_tensor(image, dtype=tf.float32)
        label =  tf.convert_to_tensor(y, dtype=tf.float32)
        derived = tf.one_hot(derived, depth=3)
                
        return image, label, derived

    def make_batch(self):

        tf_dataset = tf.data.Dataset.from_generator(self.generator, output_types=(tf.string, tf.float32, tf.int32))
        tf_dataset = tf_dataset.map(self.to_tensor, num_parallel_calls=self.num_parallel_calls)
        tf_dataset = tf_dataset.batch(self.batch_size)

        return tf_dataset


if __name__ == '__main__':
    args = parser.parse_args()

    input_shape = (224, 398)

    if args.data_size=='big':
        train_path = "./train_new.csv"
        val_path = "./val_new.csv"
        img_directory = "/media/data/teamAI/phuc/phuc/airsim/data"
    else:
        train_path = "/media/data/teamAI/phuc/airsim/20_train.csv"
        val_path = "/media/data/teamAI/phuc/airsim/20_val.csv"
        img_directory = "/media/data/teamAI/phuc/airsim/20"



    dataset = Dataset(val_path, img_directory, input_shape)
    
    sample = dataset[500]
    print(sample)
    # print(len(dataset.train_df))
    train_loader = DataLoader(dataset, input_shape=input_shape, batch_size=8, num_parallel_calls=8)
    # print(len(dataset.train_df))
    s_tensor = train_loader.to_tensor(sample[0], sample[1], sample[2])
    print(s_tensor)
    # batch_loader = train_loader.make_batch()
    #print(len(train_loader))
    #
    # begin = time.time()
    for batch_id, (images, labels, derived_label) in enumerate(tqdm(batch_loader, colour='#c22c4e')):
        pass
    # print(f'Process: {time.time()-begin} (s)')
    # # train_loader = dataset.generate_dataloader('train')
    # val_loader = dataset.generate_dataloader('val')


