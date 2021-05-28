import math

from core import common
import tensorflow as tf
from tensorflow import keras


def PPNet(input_image, num_csp_blocks=1, num_dense_blocks=2, growth_rate=4,fc_activation='leaky', conv_activation='leaky', init_num_fms=32, final_num_fms=64, fc_dropout=0.5, conv_dropout=0.5, use_downsample=True):
    #Warm-up convolution: (H x W x 1 -> H x W x init_num_fms)
    conv = common.convolutional(input_image, (3,3,input_image.shape[-1],init_num_fms))

    #pass through multiple csp blocks:
    # H X W X init_num_fms -> H/(2^num_csp_blocks) x W/(2^num_csp_blocks) x (init_num_fm + num_csp_blocks*num_denseblocks*growth_rate)
    for i in range(num_csp_blocks):
        #Pass through a csp blocks: H x W x num_fm -> H x W x (num_fm + num_dense*growth_rate)
        num_fm = conv.shape[-1]
        conv = common.csp_block(conv, num_fm, growth_rate, num_dense_blocks, conv_activation, conv_dropout)
        #Downsampling: H x W -> H//2 x W//2
        num_fm = conv.shape[-1]
        conv = common.convolutional(conv, (3,3, num_fm, num_fm), use_downsample)

    #Final convolution to reduce number of feature maps > H x W x final_num_fms
    conv = common.convolutional(conv, (1,1,conv.shape[-1], final_num_fms))
    image_output = keras.layers.Flatten()(conv)
    #Fully connected for regression
    net_output = common.fully_connected(image_output, units=16, activate_type=fc_activation, dropout=fc_dropout)
    net_output = common.fully_connected(net_output, units=3, activate=False)

    return net_output


def create_net(image_shape=(416,416,3), num_csp_blocks=4, num_dense_blocks=2, growth_rate=4, fc_activation='leaky', conv_activation='leaky', init_num_fms=32, final_num_fms=64, fc_dropout=0.5, conv_dropout=0.5):
    input_tensor = keras.layers.Input(shape=image_shape, name='input_image')
    net_output = PPNet(input_tensor, num_csp_blocks=num_csp_blocks, num_dense_blocks=num_dense_blocks, growth_rate=growth_rate,
                       conv_activation=conv_activation, fc_activation=fc_activation,
                       conv_dropout=conv_dropout, fc_dropout=fc_dropout,
                       init_num_fms=init_num_fms, final_num_fms=final_num_fms)

    model = keras.Model(inputs=input_tensor, outputs=net_output, name='PPNet')

    return model

def create_simple_net(image_shape=(224,224,1),  num_conv=6, init_num_fm=32, use_preprocess_layer=True,
                      conv_dropout=0.25, turn_off_later_bn=True, turn_off_conv_dropout=True,fc_dropout=0.5,
                      conv_activation='leaky', fc_activation='relu'):
    input_tensor = keras.layers.Input(shape=image_shape, name='input_image')
    conv = input_tensor
    if use_preprocess_layer:
        conv = keras.layers.experimental.preprocessing.Rescaling(scale=1./127.5, offset=-1)(conv)

    conv = common.convolutional(conv, filters_shape=(3,3,1,init_num_fm), dropout=not turn_off_conv_dropout, activate_type=conv_activation)
    conv = tf.keras.layers.MaxPool2D()(conv)

    for i in range(num_conv-1):
        conv = common.convolutional(conv, filters_shape=(3, 3, math.pow(2, i)*init_num_fm, math.pow(2, (i+1))*init_num_fm), dropout=not turn_off_conv_dropout, bn = not turn_off_later_bn, activate_type=conv_activation)
        conv = tf.keras.layers.MaxPool2D()(conv)


    net_output = tf.keras.layers.Flatten()(conv)
    net_output = common.fully_connected(net_output, units=256 ,dropout=fc_dropout, activate_type=fc_activation)
    net_output = common.fully_connected(net_output, units=16, dropout=fc_dropout, activate_type=fc_activation)
    net_output = common.fully_connected(net_output, units=3, activate=False)


    model = keras.Model(inputs=input_tensor, outputs=net_output, name='SIMPNet')

    return model

class CustomRMSE(keras.losses.Loss):
    def call(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        distance = tf.reduce_sum(tf.square(y_true), axis=-1)  # Shape ()
        mse = tf.reduce_mean(tf.square(y_pred - y_true), axis=-1)  # Shape (1,)
        return tf.math.sqrt(tf.divide(mse, distance))  # shape (1,)

class AbsoluteRMSE(keras.losses.Loss):
    def call(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        #distance = tf.reduce_sum(tf.square(y_true), axis=-1)  # Shape ()
        mse = tf.reduce_mean(tf.square(y_pred - y_true), axis=-1)  # Shape (1,)
        return tf.math.sqrt(mse)  # shape (1,)

class DownScaledCustomRMSE(keras.losses.Loss):
    def call(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        y_true = tf.divide(y_true, 100)
        distance = tf.reduce_sum(tf.square(y_true), axis=-1)  # Shape ()
        mse = tf.reduce_mean(tf.square(y_pred - y_true), axis=-1)  # Shape (1,)
        return tf.math.sqrt(tf.divide(mse, distance))  # shape (1,)

class UpScaledCustomRMSE(keras.losses.Loss):

    def call(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        #y_true = tf.divide(y_true, 100.)
        y_pred = tf.multiply(y_pred, 100.)
        y_true = tf.multiply(y_true, 100.)
        distance = tf.reduce_sum(tf.square(y_true), axis=-1)  # Shape ()
        mse = tf.reduce_mean(tf.square(y_pred - y_true), axis=-1)  # Shape (1,)
        return tf.math.sqrt(tf.divide(mse, distance))  # shape (1,)

class UpScaledAbsoluteRMSE(keras.losses.Loss):

    def call(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        #y_true = tf.divide(y_true, 100.)
        y_pred = tf.multiply(y_pred, 100.)
        #distance = tf.reduce_sum(tf.square(y_true), axis=-1)  # Shape ()
        mse = tf.reduce_mean(tf.square(y_pred - y_true), axis=-1)  # Shape (1,)
        return tf.math.sqrt(mse)  # shape (1,)

