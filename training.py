import numpy as np
import faulthandler;

import evaluate

faulthandler.enable()
import tensorflow as tf
from core import PPNet
import pandas as pd
#import matplotlib.pyplot as plt

def normalize_output(df):
    df['x'] = df['x'].div(100)
    df['y'] = df['y'].div(100)
    df['z'] = df['z'].div(100)

    return df

def train():
    input_shape = (224,224,1)
    train_path = './data/train.csv'
    train_df = pd.read_csv(train_path)
    train_df = normalize_output(train_df)

    val_path = './data/val.csv'
    val_df = pd.read_csv(val_path)
    val_df = normalize_output(val_df)

    #Data generator
    #data_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    data_gen = tf.keras.preprocessing.image.ImageDataGenerator()

    train_generator = data_gen.flow_from_dataframe(dataframe=train_df,
                                 directory='./data/train',
                                 x_col='img',
                                 y_col=['x', 'y', 'z'],
                                 target_size=(input_shape[0], input_shape[1]),
                                 color_mode='grayscale',
                                 class_mode='raw',
                                 batch_size=8
                                 )

    val_generator = data_gen.flow_from_dataframe(dataframe=val_df,
                                 directory='./data/val',
                                 x_col='img',
                                 y_col=['x', 'y', 'z'],
                                 target_size=(input_shape[0], input_shape[1]),
                                 color_mode='grayscale',
                                 class_mode='raw',
                                 batch_size=16
                                 )
    #Define number of iterations in each epoch = number of samples / batch_size
    STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
    STEP_SIZE_VAL=val_generator.n//val_generator.batch_size

    #Create model
    #model = PPNet.create_net(input_shape, num_csp_blocks=4)
    model = PPNet.create_simple_net(input_shape, init_num_fm=32)

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(loss=PPNet.CustomRMSE(), optimizer=optimizer)
    model.compile(loss='mse', optimizer=optimizer, metrics=[evaluate.RelativeError()])
    csv_logger = tf.keras.callbacks.CSVLogger('training_1505_downscaled_mse_bigset-1e-4init32.log')

    history = model.fit(train_generator,
                        steps_per_epoch=STEP_SIZE_TRAIN,
                        validation_data=val_generator,
                        validation_steps=STEP_SIZE_VAL,
                        validation_freq=1,
                        epochs=10,
                        callbacks=[csv_logger])

    model.save('./weights/training_1505_downscaled_mse_bigset-1e-4init32')

    return  history.history['loss'],  history.history['val_loss'], history.history['val_relative_error']


'''def plot_learning_curve(loss, val_loss):
    plt.figure()
    plt.plot(range(len(loss)), [l for l in loss])
    plt.plot(range(len(val_loss)), [l for l in val_loss])
    plt.show()
    plt.savefig('new_curve.png')
'''

if __name__ == '__main__':
    loss, val_loss, val_rel = train()
    #plot_learning_curve(loss, val_loss)