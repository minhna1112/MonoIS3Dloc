import tensorflow as tf

from data.parallel_dataset import Dataset, DataLoader

from model.model import DepthAwareNet
from model.loss import L2DepthLoss, L2NormRMSE

from solver.optimizer import OptimizerFactory
from solver.trainer import Trainer

import argparse

tf.keras.backend.clear_session()

parser = argparse.ArgumentParser(description='Select between small or big data',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-d', '--data-size', type=str, choices=['big', 'small', 'real'], default='small')
parser.add_argument('-b', '--batch-size', type=int, default=32)
parser.add_argument('-j', '--jobs', type=int,  default=8)

args = parser.parse_args()

input_shape = (180, 320)

################################
#   Define data and dataloader #
################################
if args.data_size == 'big':
    # train_path = "../data/train_big.csv"
    # val_path = "../data/val_big.csv"
    # img_directory = "../data/big/"

    train_path = "/home/ivsr/CV_Group/phuc/airsim/train.csv"
    val_path = "/home/ivsr/CV_Group/phuc/airsim/val.csv"
    img_directory = "/home/ivsr/CV_Group/phuc/airsim/data"
else:
    # train_path = "../data/train_small.csv"
    # val_path = "../data/val_small.csv"
    # img_directory = "../data/small/"

    train_path = "/home/ivsr/CV_Group/minh/train588_50.csv"
    val_path = "/home/ivsr/CV_Group/minh/val588_50.csv"
    img_directory = "/home/ivsr/CV_Group/minh/50imperpose/full"

train_dataset = Dataset(train_path, img_directory, input_shape)
val_dataset = Dataset(val_path, img_directory, input_shape)

train_loader = DataLoader(train_dataset, input_shape=input_shape, batch_size=args.batch_size, num_parallel_calls=args.jobs)
val_loader = DataLoader(val_dataset, input_shape=input_shape, batch_size=args.batch_size, num_parallel_calls=args.jobs)

# train_loader = dataset.generate_dataloader('train')
# val_loader = dataset.generate_dataloader('val')

################
# Define model #
################
net = DepthAwareNet(num_ext_conv=0)
net.build(input_shape=(None, input_shape[0], input_shape[1], 1))
net.summary()
#######################
# Define loss function#
#######################
USE_MSE = True
if USE_MSE:
    dist_loss_fn = tf.keras.losses.MeanSquaredError()
    depth_loss_fn = tf.keras.losses.MeanSquaredError()
else  :
    dist_loss_fn = L2NormRMSE()
    depth_loss_fn = L2DepthLoss()


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
                      log_path='../ivsr_logs/log2501_big_baseline_mse.txt', savepath='../ivsr_weights/training_2501_big_baseline_mse',
                      use_mse=USE_MSE)

    _  = trainer.train(200, False)
    #trainer.save_model()

