from tensorboard.plugins.hparams import api as hp
from core import PPNet
import tensorflow as tf
from tensorflow import keras

def create_tuning_net(hparams):
    net = PPNet.create_net(
        num_fc_blocks= hparams[HP_NUM_FC_BLOCKS],
        num_res_blocks= hparams[HP_NUM_RES_BLOCKS],
        fc_activation= hparams[HP_ACTIVATION],
        conv_activation=hparams[HP_ACTIVATION],
        init_num_units=hparams[HP_INIT_NUM_UNITS],
        final_num_units=hparams[HP_FINAL_NUM_UNITS],
        use_downsample=hparams[HP_USE_DOWNSAMPLE]
    )
    net.compile(
        optimizer='adam',
        loss='mse',
    )

    return net

def run_tuning_session

if __name__ == '__main__':
    HP_NUM_FC_BLOCKS = hp.HParam('num_fc_blocks', hp.Discrete([2, 5]))
    HP_NUM_RES_BLOCKS = hp.HParam('num_res_blocks', hp.Discrete([2, 4]))
    HP_ACTIVATION = hp.HParam('activation_type', hp.Discrete(['leaky', 'mish']))
    HP_INIT_NUM_UNITS = hp.HParam('init_num_units', hp.Discrete([8, 16, 32]))
    HP_FINAL_NUM_UNITS = hp.HParam('final_num_units', hp.Discrete([8, 16, 32]))
    HP_USE_DOWNSAMPLE = hp.HParam('use_downsample', hp.Discrete([True, False]))


    with tf.summary.create_file_writer('logs/hparam_tuning').as_default():
        hp.hparams_config(
            hparams=[HP_NUM_FC_BLOCKS,
                     HP_NUM_RES_BLOCKS,
                     HP_ACTIVATION,
                     HP_INIT_NUM_UNITS,
                     HP_FINAL_NUM_UNITS,
                     HP_USE_DOWNSAMPLE],

            metrics=[hp.Metric(tf.keras.metrics.MeanSquaredError(), display_name='MSE_LOSS')],

        )



