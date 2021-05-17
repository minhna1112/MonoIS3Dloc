import numpy as np
import faulthandler; faulthandler.enable()
import tensorflow as tf
import pandas as pd
from core import PPNet
from core.PPNet import custom_loss
from training import normalize_output

class CoordinateLoss(tf.keras.metrics.Metric):
    def __init__(self, name='coordinate_loss', **kwargs):
        super(CoordinateLoss, self).__init__(name=name, **kwargs)
        self.x_loss = self.add_weight(name="x_loss", initializer="zeros")
        self.y_loss = self.add_weight(name="y_loss", initializer="zeros")
        self.z_loss = self.add_weight(name="z_loss", initializer="zeros")

    def update_state(self, ground_truth, pred, sample_weight=None):
        print(ground_truth.shape, pred.shape)
        x_pred, y_pred, z_pred = pred[:, 0], pred[:, 1], pred[:, 2]
        x_true, y_true, z_true = ground_truth[:, 0], ground_truth[:, 1], ground_truth[:, 2]
        x_loss = tf.subtract(x_pred, x_true)
        y_loss = y_pred - y_true
        z_loss = z_pred - z_true

        #avg_loss = (x_loss + y_loss + z_loss) / 3
        self.x_loss.assign(x_loss)
        self.y_loss.assign(y_loss)
        self.z_loss.assign(z_loss)

    def result(self):
        return self.x_loss, self.y_loss, self.z_loss

class UpscaledRelativeError(tf.keras.metrics.Metric):
    def __init__(self, name='upscaled relative error', **kwargs):
        super(UpscaledRelativeError, self).__init__(name=name, **kwargs)
        self.error = self.add_weight(name=name, initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        loss_obj = PPNet.UpScaledCustomRMSE()
        self.error.assign(loss_obj(y_true, y_pred))

    def result(self):
        return self.error

class UpscaledAbsoluteError(tf.keras.metrics.Metric):
    def __init__(self, name='upscaled absolute error', **kwargs):
        super(UpscaledAbsoluteError, self).__init__(name=name, **kwargs)
        self.error = self.add_weight(name=name, initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        loss_obj = PPNet.UpScaledAbsoluteRMSE()
        self.error.assign(loss_obj(y_true, y_pred))

    def result(self):
        return self.error

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
    input_shape=(224,224)
    val_path = './old/data82/val.csv'
    val_df = pd.read_csv(val_path)
    val_df = normalize_output(val_df)
    #data_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255.)
    data_gen = tf.keras.preprocessing.image.ImageDataGenerator()

    val_generator = data_gen.flow_from_dataframe(dataframe=val_df,
                                                 directory='./old/data82/val',
                                                 x_col='img',
                                                 y_col=['x', 'y', 'z'],
                                                 target_size=(input_shape[0], input_shape[1]),
                                                 color_mode='grayscale',
                                                 class_mode='raw',
                                                 batch_size=8
                                                 )


    #model = tf.keras.models.load_model('./weights/baseline_2204_simp_zcentered')
    model = tf.keras.models.load_model('./weights/training_1405_downscaled_mse_subset-1e-4', compile=False)
    #model.compile(loss='mse', metrics=[UpscaledRelativeError()])
    model.compile(loss='mse', metrics=[tf.keras.losses.MeanSquaredError(),RelativeError()])
    model.evaluate(val_generator, batch_size=val_generator.batch_size)

if __name__ == '__main__':
    eval()