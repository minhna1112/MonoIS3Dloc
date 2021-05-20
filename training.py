import tensorflow as tf
from core import PPNet
from core import dataset
import evaluate

import datetime
import os

today = datetime.datetime.today()


def train(train_data, val_data, train_option='scratch', network_resolution=(224, 224, 1), pretrained_model_path=None, save_model_path=None,  log_path = None):
    STEP_SIZE_TRAIN=train_data.n//train_data.batch_size
    STEP_SIZE_VAL=val_data.n//val_data.batch_size

    append = False
    
    if save_model_path is None:
        save_model_path = './weights/{y}-{m}-{d}-{h}:{min}'.format(y = today.year, m = today.month, d = today.day, 
                                                                   h = today.hour, min = today.minute)
    if log_path is None:
        log_path = os.path.join('logs', save_model_path.split('/')[-1] + '.log')

    if train_option is not 'scratch':
        #Transfer or continue
        model = tf.keras.models.load_model(pretrained_model_path, compile=False)
        if train_option == 'continue':
            append = True
            log_path = os.path.join('logs', pretrained_model_path.split('/')[-1] + '.log')
    else:
        #Train from scratch
        model = PPNet.create_simple_net(network_resolution, init_num_fm=32, conv_dropout=0, fc_dropout=0.25, turn_off_later_bn=True, conv_activation='relu', fc_activation='relu')


    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(loss='mse', optimizer=optimizer, metrics=[evaluate.RelativeError()])
    model.summary()

    csv_logger = tf.keras.callbacks.CSVLogger(log_path, append=append)

    history = model.fit(train_data,
                        steps_per_epoch=STEP_SIZE_TRAIN,
                        validation_data=val_data,
                        validation_steps=STEP_SIZE_VAL,
                        validation_freq=1,
                        epochs=20,
                        callbacks=[csv_logger])



    model.save(save_model_path)

    return  history.history['loss'],  history.history['val_loss'], history.history['val_relative_error']

if __name__ == '__main__':
    
    data_gen = tf.keras.preprocessing.image.ImageDataGenerator()
    train_data = dataset.DataLoader('../IVSR_BINARY_DATA/10imperpose/data/', data_gen, split='train')
    val_data = dataset.DataLoader('../IVSR_BINARY_DATA/10imperpose/data/', data_gen, split='val')

    train_batch = train_data.batch_loader()
    val_batch = val_data.batch_loader()
    
    loss, val_loss, val_rel = train(train_data=train_batch, val_data=val_batch, train_option='continue', pretrained_model_path='./weights/training_1505_downscaled_mse_10impperpose-1e-4init32')
    