import tensorflow as tf
import argparse
from data.parallel_dataset import Dataset, DataLoader
from solver.evaluator import Evaluator



parser = argparse.ArgumentParser(description='Select between small or big data',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-d', '--data-size', type=str, choices=['big', 'small', 'real'], default='big')
parser.add_argument('-b', '--batch-size', type=int, default=32)
parser.add_argument('-j', '--jobs', type=int,  default=16)

args = parser.parse_args()

################################
#   Define data and dataloader #
################################
if args.data_size == 'big':
    val_path = "./test_new.csv"
    img_directory = "/media/data/teamAI/phuc/phuc/airsim/data"
else:
    val_path = "./val588_50_new.csv"
    img_directory = "/media/data/teamAI/phuc/phuc/airsim/50imperpose/full/"

input_shape = (180, 320)
val_dataset = Dataset(val_path, img_directory, input_shape, preprocess_label=False)
val_loader = DataLoader(val_dataset, input_shape=input_shape, batch_size=args.batch_size, num_parallel_calls=args.jobs, 
                    validate=False, shuffle=False)

################
# Define model #
################
net = tf.keras.models.load_model('/media/data/teamAI/minh/ivsr_weights/training0505parameterized/cp-9.cpkt')
net.build(input_shape=(None, input_shape[0], input_shape[1], 1))
net.summary()
################
# evaluator #
################
evaluator = Evaluator(val_loader, model = net, log_path = '/media/data/teamAI/minh/ivsr-logs/evaluate0505parameterized.csv')
evaluator.evaluate_on_dataframe()