from core import common
import tensorflow as tf
from tensorflow import keras

def vector_route(input_layer, num_fc_blocks=1, increase=False, activation='leaky', dropout=0.5, init_num_units=8, final_units=64):
    dense = common.fully_connected(input_layer, init_num_units, activate_type=activation, dropout=dropout)

    for i in range(num_fc_blocks):
        hidden_units = init_num_units
        if increase:
            hidden_units = init_num_units*(i+1)
        dense = common.fully_connected(dense, hidden_units, activate_type=activation, dropout=dropout)

    dense = common.fully_connected(dense, final_units, activate_type=activation, dropout=dropout)

    return dense


def image_route(input_layer, num_res_blocks=3, activation='leaky', dropout=0.1, init_num_features_map=32, final_feature_maps=64, use_pooling=True, use_downsample=True):
    conv = common.convolutional(input_layer, filters_shape=(1,1, 3, init_num_features_map),activate_type=activation, dropout=dropout)
    conv = common.convolutional(input_layer, filters_shape=(3, 3, 3, 2*init_num_features_map), activate_type=activation, dropout=dropout, downsample=True)

    for i in range(num_res_blocks):
        num_fm = (2**(i+1))*init_num_features_map
        conv = common.residual_block(conv, num_fm, num_fm//2, num_fm, activate_type=activation, dropout=dropout)
        #conv = common.convolutional(conv, activate_type=activation, dropout=dropout, filters_shape=(1,1, num_fm, 2*num_fm), downsample=use_downsample)
        if conv.shape[1] > 1:
            if use_pooling:
                conv = keras.layers.MaxPool2D()(conv)

    conv = common.fully_conv_block(conv, init_num_features_map*(2**num_res_blocks), final_feature_maps, activate_type=activation, dropout=dropout)

    return conv


def PPNet(input_image, num_res_blocks=2, fc_activation='leaky', conv_activation='leaky', init_num_units=16, final_num_units=64, fc_dropout=0.5, conv_dropout=0.1, use_downsample=True):
    #vector_output = vector_route(input_vector, num_fc_blocks, activation=fc_activation, dropout=fc_dropout, init_num_units=init_num_units, final_units=final_num_units)
    image_output = image_route(input_image, num_res_blocks, conv_activation, conv_dropout, init_num_features_map=init_num_units, final_feature_maps=final_num_units, use_downsample=use_downsample)

    #print(vector_output.shape)
    #print(image_output.shape)

    #assert vector_output.shape[-1] == image_output.shape[-1]

    image_output = keras.layers.Flatten()(image_output)
    #net_output = tf.concat([vector_output, image_output], axis=-1)
    #print(net_output.shape)

    net_output = common.fully_connected(image_output, units=16, activate_type=fc_activation, dropout=fc_dropout)
    net_output = common.fully_connected(net_output, units=3, activate=False)

    return net_output

def create_net(image_shape=(416,416,3)):
    #input_vector = keras.layers.Input(shape=(4,), name='input_bbox_vector')
    input_tensor = keras.layers.Input(shape=image_shape, name='input_image')
    net_output = PPNet(input_tensor)

    model = keras.Model(inputs=input_tensor, outputs=net_output, name='PPNet')
    model.summary()

    return model

#def compute_loss(pred, label):
#    tf.nn.