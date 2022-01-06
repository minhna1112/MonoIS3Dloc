import tensorflow as tf
import pandas as pd

from solver.evaluator import Evaluator

input_shape = (398, 224, 1)

test_path = "/home/ivsr/CV_Group/phuc/airsim/test588_50.csv"
test_df = pd.read_csv(test_path)

data_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
test_loader = data_gen.flow_from_dataframe(
        dataframe=test_df,
        directory="/home/ivsr/CV_Group/phuc/airsim/50imperpose/full",
        x_col="img",
        y_col=["x", "y", "z"],
        target_size=(input_shape[1], input_shape[0]),
        color_mode="grayscale",
        class_mode="raw",
        batch_size=1)

net = tf.keras.models.load_model('../ivsr_weights/training_0401_4/cp-9.cpkt')

evaluator  = Evaluator(test_loader, net, log_path='../ivsr_logs/test_training0401_4__cp-9.log')

evaluator.evaluate_on_datafrane(test_df)