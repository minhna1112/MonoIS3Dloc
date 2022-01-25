import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

from tqdm import tqdm
from path import Path

from data.parallel_dataset  import Dataset, DataLoader

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

IMAGE_SHAPE = (720, 1280)
HP_STRIDES = hp.HParam('strides', hp.Discrete([4, 8, 12, 16]))
HP_KSIZE = hp.HParam('kernel_size', hp.Discrete([3, 5, 7]))
HP_NCONV = hp.HParam('num_conv', hp.Discrete([4, 5, 6]))
HP_USE_MSE = hp.HParam('use_mse', hp.Discrete([True, False]))


with tf.summary.create_file_writer('/media/data/teamAI/minh/logs/hparam_tuning_2601/').as_default():
    _ = hp.hparams_config(hparams=[HP_STRIDES, HP_KSIZE, HP_NCONV, HP_USE_MSE],
                          metrics=[hp.Metric('train_err', display_name='Train Distance Error (m)'),
                                   hp.Metric('val_err', display_name='Validation Distance Error (m)'),
                                   hp.Metric('num_params', display_name='# Params')])

args = parser.parse_args()


def train_val_model(hparams, sess: int):
    input_shape = (IMAGE_SHAPE[0] // hparams[HP_STRIDES],
                   IMAGE_SHAPE[1] // hparams[HP_STRIDES])

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

    train_loader = DataLoader(train_dataset, input_shape=input_shape, batch_size=args.batch_size,
                              num_parallel_calls=args.jobs)
    val_loader = DataLoader(val_dataset, input_shape=input_shape, batch_size=args.batch_size,
                            num_parallel_calls=args.jobs)

    ################
    # Define model #
    ################
    net = DepthAwareNet(num_ext_conv=hparams[HP_NCONV]-4, ksize=hparams[HP_KSIZE])
    net.build(input_shape=(None, input_shape[0], input_shape[1], 1))
    net.summary()
    #######################
    # Define loss function#
    #######################

    if hparams[HP_USE_MSE] is True:
        dist_loss_fn = tf.keras.losses.MeanSquaredError()
        depth_loss_fn = tf.keras.losses.MeanSquaredError()
    else:
        dist_loss_fn = L2NormRMSE()
        depth_loss_fn = L2DepthLoss()

    #######################
    # Define optimizer#
    #######################
    factory = OptimizerFactory(lr=1e-3, use_scheduler=False)
    optimizer = factory.get_optimizer()

    #trainer and train
    trainer = Trainer(train_loader, val_loader=val_loader,
                      model=net, distance_loss_fn=dist_loss_fn, depth_loss_fn=depth_loss_fn,
                      optimizer=optimizer,
                      log_path=Path('../ivsr_logs')/'tuning2501'/str(sess)+'.txt',
                      savepath='../ivsr_weights/tuning_2501',
                      use_mse=hparams[HP_USE_MSE])

    train_loss, val_loss = trainer.train(13, save_checkpoint=False, early_stop=True)

    return input_shape, train_loss, val_loss, net.count_params()

def run(run_dir, hparams, sess: int):
  with tf.summary.create_file_writer(run_dir).as_default():
    hp.hparams(hparams)  # record the values used in this trial
    input_shape, train_loss, val_loss, num_params = train_val_model(hparams, sess)
    #tf.summary.text('input_size', f'{}input_shape, step=1)
    tf.summary.scalar('train_err', val_loss, step=1)
    tf.summary.scalar('val_err', train_loss, step=1)
    tf.summary.scalar('num_params', num_params, step=1)


session_num = 0

for use_mse in HP_USE_MSE.domain.values:
    for ksize in HP_KSIZE.domain.values:
        for stride in HP_STRIDES.domain.values:
            if stride < 12:
                for num_conv in HP_NCONV.domain.values:
                    hparams = {
                        HP_STRIDES: stride,
                        HP_NCONV: num_conv,
                        HP_KSIZE: ksize,
                        HP_USE_MSE: use_mse,
                    }
                    run_name = "run-%d" % session_num
                    print('--- Starting trial: %s' % run_name)
                    print({h.name: hparams[h] for h in hparams})
                    run('/media/data/teamAI/minh/logs/hparam_tuning_2601/' + run_name, hparams, session_num)
                    session_num += 1

            hparams = {
                HP_STRIDES: stride,
                HP_NCONV: 4,
                HP_KSIZE: ksize,
                HP_USE_MSE: use_mse,
              }
            run_name = "run-%d" % session_num
            print('--- Starting trial: %s' % run_name)
            print({h.name: hparams[h] for h in hparams})
            run('/media/data/teamAI/minh/logs/hparam_tuning_2601/' + run_name, hparams, session_num)
            session_num += 1
