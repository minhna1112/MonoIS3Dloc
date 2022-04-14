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
             'relative_dist_error': []
            }


    def calculate_distance_from_positions(self, positions):
        return float(tf.sqrt(tf.reduce_sum(tf.square(positions))))

    def evaluate_on_datafrane(self, source_df = pd.DataFrame):
        assert  len(source_df) == self.dataloader.n
        for idx, (images, labels) in enumerate(tqdm(self.dataloader, colour='#01e1ec')):
            if idx >= len(self.val_loader):
                break
            self.metric_dict['img'].append(source_df.iloc[idx].img)
            self.metric_dict['x_label'].append(float(labels[..., 0]))
            self.metric_dict['y_label'].append(float(labels[..., 1]))
            self.metric_dict['z_label'].append(float(labels[..., 2]))
            real_distance = self.calculate_distance_from_positions(labels)
            self.metric_dict['real_distance'].append(real_distance)

            # Perform forward pass
            out_x, out_y, out_z = self.model(images)
            out = tf.concat([out_x, out_y, out_z], axis=-1)
            out = out*30.0

            self.metric_dict['x_pred'].append(float(out[..., 0]))
            self.metric_dict['y_pred'].append(float(out[..., 1]))
            self.metric_dict['z_pred'].append(float(out[..., 2]))

            pred_distance = self.calculate_distance_from_positions(out)
            self.metric_dict['pred_distance'].append(pred_distance)

            abs_out = tf.abs(out - labels)
            self.metric_dict['abs_dist'].append(abs(real_distance - pred_distance))

            self.metric_dict['abs_x'].append(float(abs_out[..., 0]))
            self.metric_dict['abs_y'].append(float(abs_out[..., 1]))
            self.metric_dict['abs_z'].append(float(abs_out[..., 2]))
            self.metric_dict['relative_dist_error'].append(real_distance - pred_distance / 30.0)
            #print(self.metric_dict)
            self.export_to_csv()


    def export_to_csv(self):
        pd.DataFrame(self.metric_dict).to_csv(self.log_path, mode='w')
