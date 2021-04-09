from core import PPNet
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt


def train():
    input_shape = (224,224,1)
    train_path = './data/train.csv'
    train_df = pd.read_csv(train_path)

    val_path = './data/val.csv'
    val_df = pd.read_csv(val_path)

    #Data generator
    data_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

    train_generator = data_gen.flow_from_dataframe(dataframe=train_df,
                                 directory='./data/train',
                                 x_col='path',
                                 y_col=['x_humancam', 'y_humancam', 'z_humancam'],
                                 target_size=(input_shape[0], input_shape[1]),
                                 color_mode='grayscale',
                                 class_mode='raw',
                                 batch_size=8
                                 )

    val_generator = data_gen.flow_from_dataframe(dataframe=val_df,
                                 directory='./data/val',
                                 x_col='path',
                                 y_col=['x_humancam', 'y_humancam', 'z_humancam'],
                                 target_size=(input_shape[0], input_shape[1]),
                                 color_mode='grayscale',
                                 class_mode='raw',
                                 batch_size=8
                                 )
    #Define number of iterations in each epoch = number of samples / batch_size
    STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
    STEP_SIZE_VAL=val_generator.n//val_generator.batch_size

    #Create model
    model = PPNet.create_net(input_shape, num_csp_blocks=4)
    model.compile(loss='mse', optimizer='adam')

    csv_logger = tf.keras.callbacks.CSVLogger('training4.log')

    history = model.fit(train_generator,
                        steps_per_epoch=STEP_SIZE_TRAIN,
                        validation_data=val_generator,
                        validation_steps=STEP_SIZE_VAL,
                        validation_freq=1,
                        epochs=40,
                        callbacks=[csv_logger])

    model.save('./weights/baseline4')

    return  history.history['loss'],  history.history['val_loss']


def plot_learning_curve(loss, val_loss):
    plt.figure()
    plt.plot(range(len(loss)), loss)
    plt.plot(range(len(val_loss)), loss)
    plt.show()

    plt.savefig('learning_curve.png')


if __name__ == '__main__':
    loss, val_loss = train()
    plot_learning_curve(loss, val_loss)