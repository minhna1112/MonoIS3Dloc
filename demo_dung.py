import pandas as pd
# import torch
import tensorflow as tf
from path import Path
import random

from tqdm import tqdm


DATA_PATH = "./dung.csv"
IMG_PATH = "/media/data/teamAI/minh/kitti_out/semantic-0.4"

class Dataset():
    def __init__(self, datapath : str, imgpath : str , input_shape : int ): 
        self.datapath =  datapath
        self.input_shape = input_shape
        self.imgpath = Path(imgpath)
        self.df_train = pd.read_csv(datapath)
        # print(self.df_train)
        self.preprocess_label()

    def preprocess_label(self): 
        self.df_train.animal[self.df_train.loc[:,'animal'] == "dog"] = 0
        self.df_train.animal[self.df_train.loc[:,'animal'] == "chicken"] = 1
        self.df_train.animal[self.df_train.loc[:,'animal'] == "duck"] = 2
        self.df_train.animal[self.df_train.loc[:,'animal'] == "cat"] = 3
        
    def __getitem__(self, index):
        x = self.imgpath / self.df_train['img'].iat[index]
        y = self.df_train['animal'].iat[index]
        return x , y 

    def __len__(self):
        return len(self.df_train)


class DataLoader():
    def __init__(self, dataset : Dataset, input_shape : int, shuffle = True, batch_size =32 , num_parallel_calls = 8 ):
        self.input_shape = input_shape
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle 
        self.num_parallel_calls = num_parallel_calls
    def generator(self):
        indices = range(len(self.dataset))
        if self.shuffle : 
            indices = random.sample(indices, len(indices))
        for i in indices:
            yield self.dataset[i]
    def to_tensor(self, x,y):
        
        img = tf.io.read_file(x)
        try:
            img = tf.io.decode_png(img, channels=1)
        except:
            print(x)

        img = tf.image.resize(img ,self.input_shape)
        img /= 255.
        # label = tf.convert_to_tensor(y, dtype = tf.float32)
        label = tf.one_hot(y, depth = 4)
        
        return img, label 
        
    def make_batch(self):
        tf_data = tf.data.Dataset.from_generator(self.generator, output_types = (tf.string, tf.int32))
        tf_data = tf_data.map(self.to_tensor, num_parallel_calls = self.num_parallel_calls)
        tf_data = tf_data.batch(self.batch_size)

        return tf_data

def main():
    image_size = (416,416)
    dataset = Dataset(DATA_PATH, IMG_PATH,image_size)
    dataloader = DataLoader(dataset, image_size, batch_size=4)
    print(len(dataset))
    # print(dataset[0])
    for i, (img, label) in enumerate(tqdm(dataloader.make_batch())):
        pass
        # print(img.shape)
        # print(label.shape)

if __name__=='__main__':
    main()        

