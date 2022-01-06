import tensorflow as tf
from model.model import DepthAwareNet
from model.loss import L2DepthLoss, L2NormRMSE

from solver.optimizer import OptimizerFactory
from solver.trainer import Trainer

import pandas as pd

MAX_VALUE = 30.0

input_shape = (398, 224, 1)

#train_path = "/home/ivsr/CV_Group/phuc/airsim/train.csv"
train_path = "/home/ivsr/CV_Group/phuc/airsim/train588_50.csv"
train_df = pd.read_csv(train_path)
train_df["x"] = train_df["x"].div(MAX_VALUE)
train_df["y"] = train_df["y"].div(MAX_VALUE)
train_df["z"] = train_df["z"].div(MAX_VALUE)
#val_path = "/home/ivsr/CV_Group/phuc/airsim/val.csv"
val_path = "/home/ivsr/CV_Group/phuc/airsim/val588_50.csv"
val_df = pd.read_csv(val_path)
val_df["x"] = val_df["x"].div(MAX_VALUE)
val_df["y"] = val_df["y"].div(MAX_VALUE)
val_df["z"] = val_df["z"].div(MAX_VALUE)

data_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_loader = data_gen.flow_from_dataframe(
      dataframe=train_df,
      directory="/home/ivsr/CV_Group/phuc/airsim/50imperpose/full",
      x_col="img",
      y_col=["x", "y", "z"],
      target_size=(input_shape[1], input_shape[0]),
      color_mode="grayscale",
      class_mode="raw",
      batch_size=32, shuffle=True)

val_loader = data_gen.flow_from_dataframe(
        dataframe=val_df,
        directory="/home/ivsr/CV_Group/phuc/airsim/50imperpose/full",
        x_col="img",
        y_col=["x", "y", "z"],
        target_size=(input_shape[1], input_shape[0]),
        color_mode="grayscale",
        class_mode="raw",
        batch_size=32)

STEP_SIZE_TRAIN = train_loader.n//train_loader.batch_size
STEP_SIZE_VAL = val_loader.n//val_loader.batch_size

net = DepthAwareNet()
net.build(input_shape=(None, 224, 398, 1))
#net.summary()

dist_loss_fn = L2NormRMSE()
depth_loss_fn = L2DepthLoss()

factory = OptimizerFactory(lr=1e-3, use_scheduler=True, staircase=True)
optimizer = factory.get_optimizer()

if __name__ == '__main__':

    trainer = Trainer(train_loader, val_loader=val_loader,
                      model=net, distance_loss_fn=dist_loss_fn, depth_loss_fn=depth_loss_fn,
                      optimizer=optimizer,
                      log_path='../ivsr_logs/log0601_20epochs.txt', savepath='../ivsr_weights/training_0601')

    trainer.train(20, True)
    trainer.save_model()

