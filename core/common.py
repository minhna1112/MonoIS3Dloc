import tensorflow as tf
from tensorflow import keras


def fully_connected(input_layer, units, activate=True, bn=True, activate_type='leaky', dropout=None):
    # A Densely connected layer : Dense -> BatchNorm -> Activation
    dense = keras.layers.Dense(units)(input_layer)
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
    return dense


def convolutional(input_layer, filters_shape, downsample=False, activate=True, bn=True, activate_type='leaky',
                  dropout=True, dropout_rate=0.5):
    #A Convolutional layer : Conv2D -> BatchNorm -> Activation
    strides = 1
    if downsample:
        padding = 'valid'
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

    if dropout:
        conv =keras.layers.Dropout(dropout_rate)(conv)
    return conv


def mish(x):
    return x * tf.math.tanh(tf.math.softplus(x))


def residual_block(input_layer, input_channel, filter_num1, filter_num2, activate_type='leaky', dropout=0.1):
    assert input_channel == filter_num2
    # Input shape: H x W X input_channel
    # Output shape: H x W X input_channel

    short_cut = input_layer
    conv = convolutional(input_layer, filters_shape=(1, 1, input_channel, filter_num1), activate_type=activate_type,
                         dropout=dropout)
    conv = convolutional(conv, filters_shape=(3, 3, filter_num1, filter_num2), activate_type=activate_type,
                         dropout=dropout)
    #The output from 2 convolutional layers is added to the original input (residual connection)
    residual_output = short_cut + conv
    return residual_output


def dense_block(input_layer, input_channel, growth_rate, activation_type='leaky', dropout=0.5):
    # Input shape: H x W X input_channel
    # Output shape: H x W X (input_channel + growth_rate)
   conv = residual_block(input_layer, input_channel, input_channel//2, input_channel, activation_type, dropout)
   conv = convolutional(conv, filters_shape=(1,1,input_channel, growth_rate), activate_type=activation_type, dropout=dropout)
   conv  = tf.concat([input_layer, conv], axis=-1)

   return conv


def csp_block(input_layer, input_channel, growth_rate, num_dense_blocks=2,activation_type='leaky', dropout=0.5):
    #split input feature maps into 2 halves (channel-wise):
    #One half is passed through multiple dense blocks
    #The other half is then concatenated with the output of the the dense blocks
    #shortcut, conv = tf.split(input_layer, num_or_size_splits=2, axis=-1) (REMOVED)

    #Input shape: H x W X input_channel
    #Output shape: H x W X (input_channel + num_dense_blocks*growth_rate)
    conv = input_layer
    for i in range(num_dense_blocks):
        num_fm = conv.shape[-1]
        conv = dense_block(conv, num_fm, growth_rate, activation_type, dropout)
    #conv = tf.concat([shortcut, conv], axis=-1) (REMOVED)

    return conv

