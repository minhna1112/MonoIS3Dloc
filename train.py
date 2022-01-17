import tensorflow as tf

from data.dataset import Dataset

from model.model import DepthAwareNet
from model.loss import L2DepthLoss, L2NormRMSE

from solver.optimizer import OptimizerFactory
from solver.trainer import Trainer

import argparse

tf.keras.backend.clear_session()

parser = argparse.ArgumentParser(description='Select between small or big data',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-d', '--data-size', type=str, choices=['big', 'small', 'real'], default='small')

args = parser.parse_args()

input_shape = (224, 398)

################################
#   Define data and dataloader #
################################
if args.data_size == 'big':
    train_path = "/home/ivsr/CV_Group/phuc/airsim/train.csv"
    val_path = "/home/ivsr/CV_Group/phuc/airsim/val.csv"
    img_directory = "/home/ivsr/CV_Group/phuc/airsim/data"
else:
    train_path = "/home/ivsr/CV_Group/phuc/airsim/train588_50.csv"
    val_path = "/home/ivsr/CV_Group/phuc/airsim/val588_50.csv"
    img_directory = "/home/ivsr/CV_Group/phuc/airsim/50imperpose/full"

dataset = Dataset(train_path, val_path, img_directory, input_shape)
train_loader = dataset.generate_dataloader('train')
val_loader = dataset.generate_dataloader('val')

################
# Define model #
################
net = DepthAwareNet()
net.build(input_shape=(None, input_shape[0], input_shape[1], 1))

#######################
# Define loss function#
#######################
dist_loss_fn = tf.keras.losses.MeanSquaredError()
depth_loss_fn = tf.keras.losses.MeanSquaredError()

#######################
# Define optimizer#
#######################
factory = OptimizerFactory(lr=1e-3, use_scheduler=False)
optimizer = factory.get_optimizer()

if __name__ == '__main__':

    #trainer and train
    trainer = Trainer(train_loader, val_loader=val_loader,
                      model=net, distance_loss_fn=dist_loss_fn, depth_loss_fn=depth_loss_fn,
                      optimizer=optimizer,
                      log_path='../ivsr_logs/log0801_mse_noreg.txt', savepath='../ivsr_weights/training_0801_mse_no_reg')

    #trainer.train(20, True)
    #trainer.save_model()

