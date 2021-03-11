import tensorflow as tf
from tensorflow import keras


def fully_connected(input_layer, units, activate=True, bn=True, activate_type='leaky', dropout=None):
    dense = keras.layers.Dense(units)(input_layer)
    print(dense)
    if bn:
        dense = keras.layers.BatchNormalization()(dense)
    if activate:
        if activate_type == 'leaky':
            dense = tf.nn.leaky_relu(dense, alpha=0.1)
        elif activate_type == "mish":
            dense = mish(dense)
        elif activate_type == "relu":
            dense = tf.nn.relu(dense)
    if dropout is not None:
        dense = keras.layers.Dropout(dropout)(dense)
    print(dense)
    return dense


def convolutional(input_layer, filters_shape, downsample=False, activate=True, bn=True, activate_type='leaky',
                  dropout=None, flatten=False):
    strides = 1
    if downsample:
        padding = 'valid'
        if not flatten:
            strides = 2
            #input_layer = keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(input_layer)
    else:
        padding = 'same'

    conv = keras.layers.Conv2D(filters=filters_shape[-1], kernel_size=(filters_shape[0], filters_shape[1]), strides=strides,
                               padding=padding,
                               use_bias=not bn, kernel_regularizer=keras.regularizers.l2(0.0005),
                               kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                               bias_initializer=tf.constant_initializer(0.))(input_layer)

    if bn:
        conv = keras.layers.BatchNormalization()(conv)

    if activate:
        if activate_type == "leaky":
            conv = tf.nn.leaky_relu(conv, alpha=0.1)
        elif activate_type == "mish":
            conv = mish(conv)
        elif activate_type == "relu":
            conv = tf.nn.relu(conv)

    if dropout is not None:
        conv = keras.layers.Dropout(dropout)(conv)
    return conv


def mish(x):
    return x * tf.math.tanh(tf.math.softplus(x))


def residual_block(input_layer, input_channel, filter_num1, filter_num2, activate_type='leaky', dropout=0.1):
    short_cut = input_layer
    conv = convolutional(input_layer, filters_shape=(1, 1, input_channel, filter_num1), activate_type=activate_type,
                         dropout=dropout)
    conv = convolutional(conv, filters_shape=(3, 3, filter_num1, filter_num2), activate_type=activate_type,
                         dropout=dropout)

    residual_output = short_cut + conv
    return residual_output


def fully_conv_block(input_layer, input_channel, filter_num1, activate_type='leaky', dropout=0.5):
    conv = convolutional(input_layer,
                         filters_shape=(input_layer.shape[1], input_layer.shape[2], input_channel, input_channel//2),
                         activate_type=activate_type, downsample=True, flatten=True, dropout=dropout)
    conv = convolutional(conv, filters_shape=(1, 1, input_channel//2, filter_num1), activate_type=activate_type,
                         flatten=True, dropout=dropout)
    return conv

