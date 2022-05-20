import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

from .trainer import Trainer

from tqdm import tqdm
from path import Path

class Tuner:
    def __init__(self, trainer: Trainer):
        self.trainer = trainer

    def run(self, max_epochs: int):
        return self.trainer.train(epochs=max_epochs, save_checkpoint=False)


HP_NUM_UNITS = hp.HParam('strides', hp.Discrete([8]))

with tf.summary.create_file_writer('../tune_logs/hparam_tuning').as_default():
    _ = hp.hparams_config(hparams=[HP_NUM_UNITS],
                          metrics=[hp.Metric('dist_err', display_name='Distance Error')])
