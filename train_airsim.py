import os
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.callbacks import Callback
from keras import backend as K
# physical_devices = tf.config.list_physical_devices("GPU")
# tf.config.experimental.set_memory_growth(physical_devices[0], True)
MAX_VALUE = 30.0
LOG_DIR = "log1"
os.makedirs(LOG_DIR, exist_ok=True)

checkpoint_path = "./training_21_12/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, 
    verbose=1, 
    save_weights_only=False,
    save_freq='epoch')





class PlotCallback(Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.loss = []
        self.val_loss = []
        self.logs = []
    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(self.i+1)
        self.loss.append(logs.get("loss"))
        self.val_loss.append(logs.get("val_loss")*MAX_VALUE)
        self.i += 1
        fig, ax = plt.subplots(1, 1)
        ax.plot(self.x, self.loss, label="loss")
        ax.plot(self.x, self.val_loss, label="val_loss", linestyle="--")
        plt.legend()
        plt.savefig(os.path.join(LOG_DIR, "loss_" + str(self.i) + ".png"))
        plt.show()
        plt.close(fig)
def euclidean_distance_loss(y_true, y_pred):
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))
def simple_net(image_shape=(398, 224, 1)):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation="relu", input_shape=image_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(256, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(256, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    
    model.add(Dense(512))
    model.add(Dense(64))
    model.add(Dense(3))
    model.add(Activation("tanh"))
    model.summary()
    #plot_model(model, to_file="model.png")
    return model
def train():
    # input_shape = (224, 224, 1)
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
    data_gen_args = dict(featurewise_center=True,
                         featurewise_std_normalization=True,
                         rescale=1./255,
                         rotation_range=90.,
                         width_shift_range=0.2,
                         height_shift_range=0.2,
                         zoom_range=0.2)
    data_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    train_generator = data_gen.flow_from_dataframe(
      dataframe=train_df,
      directory="/home/ivsr/CV_Group/phuc/airsim/data",
      x_col="img",
      y_col=["x", "y", "z"],
      target_size=(input_shape[1], input_shape[0]),
      color_mode="grayscale",
      class_mode="raw",
      batch_size=32)
    val_generator = data_gen.flow_from_dataframe(
        dataframe=val_df,
        directory="/home/ivsr/CV_Group/phuc/airsim/data",
        x_col="img",
        y_col=["x", "y", "z"],
        target_size=(input_shape[1], input_shape[0]),
        color_mode="grayscale",
        class_mode="raw",
        batch_size=32)
    STEP_SIZE_TRAIN = train_generator.n//train_generator.batch_size
    STEP_SIZE_VAL = val_generator.n//val_generator.batch_size
    model = simple_net(input_shape)
    #model.load_weights("/home/ivsr/CV_Group/phuc/airsim/training/cp-0005.ckpt")
    model.compile(loss=euclidean_distance_loss, optimizer="adam")
    #plot_callback = PlotCallback()
    csv_logger = tf.keras.callbacks.CSVLogger("training_21_12.log")
    model.save_weights(checkpoint_path.format(epoch=0))
    history = model.fit(train_generator,
                        steps_per_epoch=STEP_SIZE_TRAIN,
                        validation_data=val_generator,
                        validation_steps=STEP_SIZE_VAL,
                        validation_freq=1,
                        epochs=20,
                        callbacks=[csv_logger,cp_callback])
    

    model.save("./weights/baseline_21_12")
    return history.history["loss"], history.history["val_loss"]
def plot_learning_curve(loss, val_loss):
    plt.figure()
    plt.plot(range(len(loss)), loss)
    plt.plot(range(len(val_loss)), loss)
    plt.show()
    plt.savefig("learning_curve1.png")
if __name__ == "__main__":
  loss, val_loss = train()
  
