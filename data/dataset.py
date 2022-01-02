import tensorflow as tf
import pandas as pd

MAX_VALUE = 30.0

input_shape = (398, 224, 1)

train_path = "/home/ivsr/CV_Group/phuc/airsim/train.csv"
train_df = pd.read_csv(train_path)
train_df["x"] = train_df["x"].div(MAX_VALUE)
train_df["y"] = train_df["y"].div(MAX_VALUE)
train_df["z"] = train_df["z"].div(MAX_VALUE)
val_path = "/home/ivsr/CV_Group/phuc/airsim/val.csv"
val_df = pd.read_csv(val_path)
val_df["x"] = val_df["x"].div(MAX_VALUE)
val_df["y"] = val_df["y"].div(MAX_VALUE)
val_df["z"] = val_df["z"].div(MAX_VALUE)

data_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_loader = data_gen.flow_from_dataframe(
      dataframe=train_df,
      directory="/home/ivsr/CV_Group/phuc/airsim/data",
      x_col="img",
      y_col=["x", "y", "z"],
      target_size=(input_shape[1], input_shape[0]),
      color_mode="grayscale",
      class_mode="raw",
      batch_size=32)

val_loader = data_gen.flow_from_dataframe(
        dataframe=val_df,
        directory="/home/ivsr/CV_Group/phuc/airsim/data",
        x_col="img",
        y_col=["x", "y", "z"],
        target_size=(input_shape[1], input_shape[0]),
        color_mode="grayscale",
        class_mode="raw",
        batch_size=1)

STEP_SIZE_TRAIN = train_loader.n//train_loader.batch_size
STEP_SIZE_VAL = val_loader.n//val_loader.batch_size

next_batch = next(iter(val_loader))
print(next_batch[0].shape) # batch_size, h, w, 1
print(next_batch[1].shape) # batch_size, 3