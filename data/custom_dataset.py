import time

import tensorflow as tf
import pandas as pd
from tqdm import tqdm
from path import Path

import argparse



parser = argparse.ArgumentParser(description='Select between small or big data',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-d', '--data-size', type=str, choices=['big', 'small', 'real'], default='small')

class Dataset:
    def __init__(self, train_path: str,
                 img_directory: str, input_shape: tuple):

        self.MAX_VALUE = 30.0
        self.input_shape = input_shape
        self.image_dir = Path(img_directory)

        self.train_path = train_path
        self.train_df = pd.read_csv(train_path)
        self.train_df = self.preprocess_label(self.train_df)

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
        return X, y

class DataLoader:
    def __init__(self, dataset: Dataset, input_shape):
        self.dataset = dataset
        self.input_shape = input_shape
        #self.dataset.generate_dataiterator(split)

    def generator(self):
        for i in range(len(self.dataset)):
            yield self.dataset[i]

    def to_tensor(self, x, y):

        # image = tf.keras.preprocessing.image.load_img(x, color_mode='grayscale', target_size=self.input_shape)
        # image = tf.keras.preprocessing.image.img_to_array(image)
        # image_path = os.path.join(folder_path,raw_data['image_name'].numpy().decode('utf-8'))

        image = tf.io.read_file(x)
        image = tf.io.decode_jpeg(image, channels=1)
        image = tf.image.resize(image, self.input_shape)
        image /= 255.0  # normalize to [0,1] range

        image = tf.convert_to_tensor(image, dtype=tf.float32)
        label =  tf.convert_to_tensor(y, dtype=tf.float32)

        return image, label

    def make(self, batch_size: int):

        tf_dataset = tf.data.Dataset.from_generator(self.generator, output_types=(tf.string, tf.float32))
        tf_dataset = tf_dataset.map(self.to_tensor, num_parallel_calls=8)
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


    dataset = Dataset(train_path, img_directory, input_shape)
    sample = dataset[500]
    print(sample)
    train_loader = DataLoader(dataset, input_shape=input_shape)
    s_tensor = train_loader.to_tensor(sample[0], sample[1])
    print(s_tensor)
    batch_loader = train_loader.make(32)

    #
    begin = time.time()
    for batch_id, (images, labels) in enumerate(tqdm(batch_loader, colour='#c22c4e')):
        pass
    print(f'Process: {time.time()-begin} (s)')
    # # train_loader = dataset.generate_dataloader('train')
    # val_loader = dataset.generate_dataloader('val')


    # next_batch = next(batch_loader)
    # print(next_batch[0].shape) # batch_size, h, w, 1
    # print(next_batch[1].shape) # batch_size, 3

    # next_batch = next(val_loader)
    # print(next_batch[0].shape) # batch_size, h, w, 1
    # print(next_batch[1].shape) # batch_size, 3

    # train_loader = DataLoader(dataset, 'val')
    # tf_dataset = tf.data.Dataset.from_generator(train_loader.generator, output_types=(tf.float32, tf.float32))
    # tf_dataset = tf_dataset.map(lambda x, y: (tf.squeeze(x, axis=0), tf.squeeze(y, axis=0)), num_parallel_calls=4)
    # tf_dataset = tf_dataset.batch(320)
    # for i, (X, y) in enumerate(batch_loader):
    #     print(X.shape)
    #     print(y.shape)
    #print(next(iter(train_loader.generator())))  # batch_size, h, w, 1; batch_size, 3