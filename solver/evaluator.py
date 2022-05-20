import tensorflow as tf
import os
from path import Path
import pandas as pd
from tqdm import tqdm
import numpy as np

class Evaluator:
    def __init__(self, dataloader, model: tf.keras.Model, log_path: str):
        self.dataloader  = dataloader
        self.model = model
        self.batch_size = self.dataloader.batch_size
        self.alpha = 0.1
        self.log_path = log_path
        self.metric_dict = {'img': [],
            'x_label': [], 'y_label': [], 'z_label': [],
            'x_pred': [], 'y_pred': [], 'z_pred': [],
            "real_distance": [], "pred_distance": [],
            'abs_x': [], 'abs_y': [], 'abs_z': [], 'abs_dist': [],
             'relative_dist_error': [], 'l2_err': [], 'rel_l2_err': []
            }
        # Write headers to csv file
        self.metric_frame = pd.DataFrame(self.metric_dict)
        self.metric_frame.to_csv(log_path, mode='w', sep=',')

    @tf.function
    def evaluate_on_batch(self, images,labels):
        
        # Groundtruth information
        x_label = labels[..., 0]
        y_label = labels[..., 1]
        z_label = labels[..., 2]
        real_distance = tf.sqrt(tf.reduce_sum(tf.square(labels), axis=-1))
        
        # Perform forward pass
        out_x, out_y, out_z, _ = self.model(images)
        out = tf.concat([out_x, out_y, out_z], axis=-1)
        out = out *  31.24
        # out = out * self.dataloader.MAX_VALUE # 31.24
        x_pred = out[..., 0] 
        y_pred = out[..., 1]
        z_pred = out[..., 2]
        pred_distance = tf.sqrt(tf.reduce_sum(tf.square(out), axis=-1))
        # Error calculation
        abs_err = tf.abs(real_distance - pred_distance)
        abs_out = tf.abs(out - labels)
        abs_err_x = abs_out[..., 0]
        abs_err_y = abs_out[..., 1]
        abs_err_z = abs_out[..., 2]
        relative_dist_error = tf.abs(real_distance - pred_distance) / 31.24

        l2_err =  tf.sqrt(tf.reduce_sum(tf.square(out - labels), axis=-1))
        relative_l2_err = l2_err / 31.24
        
        return tf.concat([i[:, np.newaxis] for i in [x_label, y_label, z_label, 
                                                    x_pred, y_pred, z_pred, 
                                                    real_distance, pred_distance, abs_err_x, abs_err_y, abs_err_z, abs_err, relative_dist_error, l2_err, relative_l2_err]], axis=-1)

    def evaluate_on_dataframe(self):
        batch_size = self.dataloader.batch_size
        for idx, (images, labels) in enumerate(tqdm(self.dataloader.make_batch(), colour='#01e1ec')):
            img_paths = []
            for b in range(batch_size):
                img_paths.append(str(self.dataloader.dataset[idx*batch_size+b][0]))
            img_paths = np.array(img_paths)[:, np.newaxis]
            # print(img_paths.shape)
            evaluation = self.evaluate_on_batch(images, labels)
            out = np.concatenate([img_paths, evaluation], axis=-1)
            pd.DataFrame(out).to_csv(self.log_path, mode='a', sep=',', header=None)

    # def export_to_csv(self):
    #     pd.DataFrame(self.metric_dict).to_csv(self.log_path, mode='w')
