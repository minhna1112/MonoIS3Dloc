import numpy as np
import faulthandler; faulthandler.enable()
import tensorflow as tf
from core import PPNet
from core import dataset

class RelativeError(tf.keras.metrics.Metric):
    def __init__(self, name='relative_error', **kwargs):
        super(RelativeError, self).__init__(name=name, **kwargs)
        self.error = self.add_weight(name=name, initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        loss_obj = PPNet.CustomRMSE()
        self.error.assign(loss_obj(y_true, y_pred))

    def result(self):
        return self.error

def eval():
    data_gen = tf.keras.preprocessing.image.ImageDataGenerator()
    val_data = dataset.DataLoader('../IVSR_BINARY_DATA/50imperpose/data/', data_gen, split='val')
    val_generator = val_data.batch_loader()
    #model = tf.keras.models.load_model('./weights/baseline_2204_simp_zcentered')
    model = tf.keras.models.load_model('./weights/training_1405_downscaled_mse_subset-1e-4', compile=False)
    #model.compile(loss='mse', metrics=[UpscaledRelativeError()])
    model.compile(loss='mse', metrics=[tf.keras.losses.MeanSquaredError(),RelativeError()])
    model.evaluate(val_generator, batch_size=val_generator.batch_size)

if __name__ == '__main__':
    eval()