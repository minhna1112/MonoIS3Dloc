import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from path import Path
import json
import os

tf.autograph.set_verbosity(0)

class Trainer:
    def __init__(self, dataloader, model: tf.keras.Model,
                 distance_loss_fn: tf.keras.losses.Loss, depth_loss_fn: tf.keras.losses.Loss,
                 optimizer: tf.keras.optimizers.Optimizer, val_loader,
                 log_path: str, savepath: str, use_mse=True):

        self.dataloader  = dataloader
        self.model = model
        self.distance_loss_fn = distance_loss_fn
        self.depth_loss_fn = depth_loss_fn
        self.optimizer = optimizer
        self.batch_size = self.dataloader.batch_size
        self.alpha = 0.0

        self.val_loader = val_loader

        self.savepath = Path(savepath)
        self.log_path = log_path
        self.loss_dict = { 'train_dist_loss': [], 'train_depth_loss': [], 'entropy_loss': [],
                           'val_dist_loss': [], 'val_depth_loss': [], 'val_entropy_loss': []}
        pd.DataFrame(self.loss_dict).to_csv(log_path)
        self.use_mse = use_mse

        self.crossentropy_loss = tf.keras.losses.CategoricalCrossentropy()

    @tf.function
    def train_on_batch(self, images,labels, derived_labels):
        with tf.GradientTape() as tape:
            #Perform forward pass 
            out_x, out_y, out_z, out_params = self.model(images)
            #Calculate loss
            if self.depth_loss_fn is not None:
                label_x = tf.reshape(labels[..., 0], out_x.shape)
                # label_x = labels[..., 0].reshape(out_x.shape)
                depth_loss = self.depth_loss_fn(label_x,out_x)
            else:
                depth_loss = 0.0

            out = tf.concat([out_x, out_y, out_z], axis=-1)
            distance_loss = self.distance_loss_fn(out, labels)
            entropy_loss = self.crossentropy_loss(out_params, derived_labels)

            depth_loss      = tf.cast(depth_loss, tf.float32)
            distance_loss   = tf.cast(distance_loss, tf.float32)
            alpha           = tf.cast(self.alpha, tf.float32)
            entropy_loss    = tf.cast(entropy_loss, tf.float32)           

            loss_values     = alpha * depth_loss + distance_loss + entropy_loss

        #Calculate backward gradients
        gradients = tape.gradient(loss_values, self.model.trainable_weights)
        #Update weights
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_weights))

        return loss_values, distance_loss, depth_loss, entropy_loss


    @tf.function
    def validate_on_batch(self, images,labels, derived_labels):
        # Perform forward pass
        out_x, out_y, out_z, out_params = self.model(images)
        # Calculate loss
        if self.depth_loss_fn is not None:
            label_x = tf.reshape(labels[..., 0], out_x.shape)
            # label_x = labels[..., 0].reshape(out_x.shape)
            depth_loss = self.depth_loss_fn(label_x, out_x)
        else:
            depth_loss = 0.0
        out = tf.concat([out_x, out_y, out_z], axis=-1)
        distance_loss = self.distance_loss_fn(out, labels)
        entropy_loss = self.crossentropy_loss(out_params, derived_labels)


        depth_loss      = tf.cast(depth_loss, tf.float32)
        distance_loss   = tf.cast(distance_loss, tf.float32)
        alpha           = tf.cast(self.alpha, tf.float32)
        entropy_loss    = tf.cast(entropy_loss, tf.float32)           

        loss_values     = alpha * depth_loss + distance_loss + entropy_loss

        return loss_values, distance_loss, depth_loss, entropy_loss

    @tf.function
    def convert_error_to_meter(self, err):
        if self.use_mse:
            err = tf.math.sqrt(err)
        err  = err * self.dataloader.dataset.MAX_VALUE
        return err


    def train(self, epochs: int, save_checkpoint=True,
              parallel=True, early_stop=True):
        patience = 3
        wait = 0
        best = self.dataloader.dataset.MAX_VALUE
        for e in range(epochs):
            print(f'Epoch: {e}: ........................................')
            running_train_loss = 0.0
            running_dist_loss = 0.0
            running_depth_loss = 0.0
            running_entropy_loss = 0.0

            #Train
            for batch_id, (images, labels, derived_labels) in enumerate(tqdm(self.dataloader.make_batch(), colour='#96c8a2')):
                loss, dist_loss, depth_loss, entropy_loss = self.train_on_batch(images, labels, derived_labels)
                # Log every 200 batches (6400 imgs).
                if batch_id % 200 == 0:
                    print(
                        "Training loss (for one batch) at step %d, epoch %d: %.4f; distance_loss: %.4f,  depth loss: %.4f, entropy loss: %.4f"
                        % (batch_id, e, float(loss), float(dist_loss), float(depth_loss), float(entropy_loss))
                    )
                running_train_loss += loss*len(labels)
                running_dist_loss  += dist_loss * len(labels)
                running_depth_loss += depth_loss * len(labels)
                running_entropy_loss += entropy_loss * len(labels)
                # if batch_id >= len(self.dataloader):
                #     break

            running_train_loss /= self.dataloader.n
            running_dist_loss /= self.dataloader.n
            running_depth_loss /= self.dataloader.n
            running_entropy_loss /= self.dataloader.n

            print(
                "Training loss (for one epoch) at epoch %d: %.4f; distance_loss: %.4f,  depth loss: %.4f, entropy loss: %.4f"
                % (e, float(running_train_loss), float(running_dist_loss), float(running_depth_loss), float(running_entropy_loss))
            )

            train_loss_in_meters = float(self.convert_error_to_meter(running_dist_loss))

            self.loss_dict['train_dist_loss'].append(float(running_dist_loss))
            self.loss_dict['train_depth_loss'].append(float(running_depth_loss))
            self.loss_dict['entropy_loss'].append(float(running_entropy_loss))

            running_val_loss = 0.0
            running_dist_loss = 0.0
            running_depth_loss = 0.0
            running_entropy_loss = 0.0


            #Validation
            for batch_id, (images, labels, derived_labels) in enumerate(tqdm(self.val_loader.make_batch(), colour='#c22c4e')):
                loss, dist_loss, depth_loss, entropy_loss = self.validate_on_batch(images, labels, derived_labels)
                # Log every 200 batches (6400 imgs).
                if batch_id % 200 == 0:
                    print(
                        "Validation loss (for one batch) at step %d, epoch %d: %.4f; distance_loss: %.4f,  depth loss: %.4f, entropy loss: %.4f"
                        % (batch_id, e, float(loss), float(dist_loss), float(depth_loss), float(entropy_loss))
                    )
                running_val_loss += loss * len(labels)
                running_dist_loss += dist_loss * len(labels)
                running_depth_loss += depth_loss * len(labels)
                running_entropy_loss += entropy_loss * len(labels)


                # if batch_id >= len(self.val_loader):
                #     break

            running_val_loss /= self.val_loader.n
            running_dist_loss /= self.val_loader.n
            running_depth_loss /= self.val_loader.n
            running_entropy_loss /= self.dataloader.n

            print(
                "Validation loss (for one epoch) at epoch %d: %.4f; distance_loss: %.4f,  depth loss: %.4f, entropy loss: %.4f"
                % (e, float(running_val_loss), float(running_dist_loss), float(running_depth_loss), float(running_entropy_loss))
            )

            val_loss_in_meters = float(self.convert_error_to_meter(running_dist_loss))
            print(f'Distance erros in meters: {val_loss_in_meters} (m)')

            
            self.loss_dict['val_dist_loss'].append(float(running_dist_loss))
            self.loss_dict['val_depth_loss'].append(float(running_depth_loss))
            self.loss_dict['val_entropy_loss'].append(float(running_entropy_loss))


            pd.DataFrame(self.loss_dict).to_csv(self.log_path)

            if save_checkpoint:
                print('Saving checkpoint....')
                self.model.save(self.savepath/f'cp-{e}.cpkt')
                print('Done')

            if early_stop:
                wait += 1
                if val_loss_in_meters < best:
                    best = val_loss_in_meters
                    wait = 0
                if wait >= patience:
                    print(f"Early stop at e = {e}")
                    break

            print('\n')

        return train_loss_in_meters, val_loss_in_meters

    def save_model(self):
        self.model.save(self.savepath/'last_checkpoint')
        print('Saved last checkpoint at' + self.savepath)