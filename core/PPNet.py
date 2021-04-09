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
    model.summary()

    return model
