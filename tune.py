import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

from tqdm import tqdm
from path import Path

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

IMAGE_SHAPE = (1280, 720)
HP_STRIDES = hp.HParam('strides', hp.Discrete([8, 12, 16, 24, 32]))

with tf.summary.create_file_writer('logs/hparam_tuning_2001/').as_default():
    _ = hp.hparams_config(hparams=[HP_STRIDES],
                          metrics=[hp.Metric('train_err', display_name='Train Distance Error'),
                                   hp.Metric('val_err', display_name='Validation Distance Error')])

args = parser.parse_args()



def train_val_model(hparams):
    input_shape = (IMAGE_SHAPE[0] // hparams[HP_STRIDES],
                   IMAGE_SHAPE[1] // hparams[HP_STRIDES])

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
    net.summary()
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

    #trainer and train
    trainer = Trainer(train_loader, val_loader=val_loader,
                      model=net, distance_loss_fn=dist_loss_fn, depth_loss_fn=depth_loss_fn,
                      optimizer=optimizer,
                      log_path=Path('../ivsr_logs/')/str(hparams[HP_STRIDES]), savepath='../ivsr_weights/training_0801_mse_no_reg')

    train_loss, val_loss = trainer.train(5, False)

    return train_loss, val_loss

def run(run_dir, hparams):
  with tf.summary.create_file_writer(run_dir).as_default():
    hp.hparams(hparams)  # record the values used in this trial
    train_loss, val_loss = train_val_model(hparams)
    tf.summary.scalar('train_err', val_loss, step=2)
    tf.summary.scalar('val_err', train_loss, step=2)

session_num = 0

for stride in HP_STRIDES.domain.values:
      hparams = {
          HP_STRIDES: stride,
      }
      run_name = "run-%d" % session_num
      print('--- Starting trial: %s' % run_name)
      print({h.name: hparams[h] for h in hparams})
      run('logs/hparam_tuning_2001/' + run_name, hparams)
      session_num += 1
