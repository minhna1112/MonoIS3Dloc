import tensorflow as tf
from model.model import DepthAwareNet, ParameterizedNet, BackboneSharedParameterizedNet
from model.loss import L2DepthLoss, L2NormRMSE
from solver.optimizer import OptimizerFactory
import argparse

tf.keras.backend.clear_session()

parser = argparse.ArgumentParser(description='Select between small or big data',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-d', '--data-size', type=str, choices=['big', 'small', 'real'], default='small')
parser.add_argument('-m', '--training-mode', type=str, choices=['normal', 'parameterized', 'shared'], default='parameterized')
parser.add_argument('-b', '--batch-size', type=int, default=32)
parser.add_argument('-j', '--jobs', type=int,  default=8)

args = parser.parse_args()

input_shape = (180, 320)

################################
#   Define data and dataloader #
################################
if args.data_size == 'big':
    train_path = "./train_new.csv"
    val_path = "./val_new.csv"
    img_directory = "/media/data/teamAI/phuc/phuc/airsim/data"
else:
    train_path = "./train588_50_new.csv"
    val_path = "./val588_50_new.csv"
    img_directory = "/media/data/teamAI/phuc/phuc/airsim/50imperpose/full/"

if args.training_mode =='normal':
    from data.parallel_dataset import Dataset, DataLoader
else:
    from data.parameterized_parallel_dataset import Dataset, DataLoader
    
train_dataset = Dataset(train_path, img_directory, input_shape)
val_dataset = Dataset(val_path, img_directory, input_shape)

train_loader = DataLoader(train_dataset, input_shape=input_shape, batch_size=args.batch_size, num_parallel_calls=args.jobs)
val_loader = DataLoader(val_dataset, input_shape=input_shape, batch_size=args.batch_size, num_parallel_calls=args.jobs)

################
# Define model #
################
if args.training_mode =='parameterized':
    net = ParameterizedNet(num_ext_conv=1)
elif args.training_mode == 'shared':
    net = BackboneSharedParameterizedNet(num_ext_conv=1)
else:
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

#trainer and train
if args.training_mode =='normal':
    from solver.trainer import Trainer
else:
    from solver.parameterized_trainer import Trainer
    
trainer = Trainer(train_loader, val_loader=val_loader,
                    model=net, distance_loss_fn=dist_loss_fn, depth_loss_fn=depth_loss_fn,
                    optimizer=optimizer,
                    log_path='/media/data/teamAI/minh/ivsr-logs/training0905shared.txt', savepath='/media/data/teamAI/minh/ivsr_weights/training0905shared',
                    use_mse=USE_MSE)

_  = trainer.train(30, save_checkpoint=True, early_stop=True)
#trainer.save_model()

