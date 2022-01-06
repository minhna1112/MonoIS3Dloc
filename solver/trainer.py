import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from path import Path
import json
import os

class Trainer():
    def __init__(self, dataloader, model: tf.keras.Model, distance_loss_fn: tf.keras.losses.Loss, depth_loss_fn: tf.keras.losses.Loss,optimizer: tf.keras.optimizers.Optimizer, val_loader, log_path: str, savepath: str):
        self.dataloader  = dataloader
        self.model = model
        self.distance_loss_fn = distance_loss_fn
        self.depth_loss_fn = depth_loss_fn
        self.optimizer = optimizer
        self.batch_size = self.dataloader.batch_size
        self.alpha = 0.1
        self.val_loader = val_loader

        self.savepath = Path(savepath)
        self.log_path = log_path
        self.loss_dict = {'train_loss': [], 'train_dist_loss': [], 'train_depth_loss': [],
                          'val_loss': [], 'val_dist_loss': [], 'val_depth_loss': []}
        pd.DataFrame(self.loss_dict).to_csv(log_path)
        
    def train_on_batch(self, images,labels):
        with tf.GradientTape() as tape:
            #Perform forward pass 
            out_x, out_y, out_z = self.model(images)
            #Calculate loss
            if self.depth_loss_fn is not None:
                label_x = labels[..., 0].reshape(out_x.shape)
                depth_loss = self.depth_loss_fn(label_x,out_x)
            else:
                depth_loss = 0.0
            out = tf.concat([out_x, out_y, out_z], axis=-1)
            distance_loss = self.distance_loss_fn(out, labels)
            loss_values = self.alpha*depth_loss + distance_loss
        #Calculate backward gradients
        gradients = tape.gradient(loss_values, self.model.trainable_weights)
        #Update weights
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_weights))
        return loss_values, distance_loss, depth_loss

    def validate_on_batch(self, images,labels):
        # Perform forward pass
        out_x, out_y, out_z = self.model(images)
        # Calculate loss
        if self.depth_loss_fn is not None:
            label_x = labels[..., 0].reshape(out_x.shape)
            depth_loss = self.depth_loss_fn(label_x, out_x)
        else:
            depth_loss = 0.0
        out = tf.concat([out_x, out_y, out_z], axis=-1)
        distance_loss = self.distance_loss_fn(out, labels)
        loss_values = self.alpha * depth_loss + distance_loss
        return loss_values, distance_loss, depth_loss



    def train(self, epochs: int, save_checkpoint=True):
        for e in range(epochs):
            print(f'Epoch: {e}: ........................................')
            running_train_loss = 0.0
            running_dist_loss = 0.0
            running_depth_loss = 0.0
            #Train
            for batch_id, (images, labels) in enumerate(tqdm(self.dataloader, colour='#96c8a2')):
                loss, dist_loss, depth_loss = self.train_on_batch(images, labels)
                # Log every 200 batches (6400 imgs).
                if batch_id % 200 == 0:
                    print(
                        "Training loss (for one batch) at step %d, epoch %d: %.4f; distance_loss: %.4f,  depth loss: %.4f"
                        % (batch_id, e, float(loss), float(dist_loss), float(depth_loss))
                    )
                running_train_loss += loss*len(labels)
                running_dist_loss  += dist_loss * len(labels)
                running_depth_loss += depth_loss * len(labels)
                if batch_id >= len(self.dataloader):
                    break

            running_train_loss /= self.dataloader.n
            running_dist_loss /= self.dataloader.n
            running_depth_loss /= self.dataloader.n

            print(
                "Training loss (for one epoch) at epoch %d: %.4f; distance_loss: %.4f,  depth loss: %.4f"
                % (e, float(running_train_loss), float(running_dist_loss), float(running_depth_loss))
            )

            self.loss_dict['train_loss'].append(float(running_train_loss))
            self.loss_dict['train_dist_loss'].append(float(running_dist_loss))
            self.loss_dict['train_depth_loss'].append(float(running_depth_loss))

            running_val_loss = 0.0
            running_dist_loss = 0.0
            running_depth_loss = 0.0

            #Validation
            for batch_id, (images, labels) in enumerate(tqdm(self.val_loader, colour='#c22c4e')):
                loss, dist_loss, depth_loss = self.validate_on_batch(images, labels)
                # Log every 200 batches (6400 imgs).
                if batch_id % 200 == 0:
                    print(
                        "Validation loss (for one batch) at step %d, epoch %d: %.4f; distance_loss: %.4f,  depth loss: %.4f"
                        % (batch_id, e, float(loss), float(dist_loss), float(depth_loss))
                    )
                running_val_loss += loss * len(labels)
                running_dist_loss += dist_loss * len(labels)
                running_depth_loss += depth_loss * len(labels)


                if batch_id >= len(self.val_loader):
                    break

            running_val_loss /= self.val_loader.n
            running_dist_loss /= self.val_loader.n
            running_depth_loss /= self.val_loader.n

            print(
                "Validation loss (for one epoch) at epoch %d: %.4f; distance_loss: %.4f,  depth loss: %.4f"
                % (e, float(running_val_loss), float(running_dist_loss), float(running_depth_loss))
            )

            self.loss_dict['val_loss'].append(float(running_val_loss))
            self.loss_dict['val_dist_loss'].append(float(running_dist_loss))
            self.loss_dict['val_depth_loss'].append(float(running_depth_loss))

            pd.DataFrame(self.loss_dict).to_csv(self.log_path)

            if save_checkpoint:
                print('Saving checkpoint....')
                self.model.save(self.savepath/f'cp-{e}.cpkt')
                print('Done')
    def save_model(self):
        self.model.save(self.savepath/'last_checkpoint')
        print('Saved last checkpoint at' + self.savepath)