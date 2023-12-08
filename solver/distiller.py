
import tensorflow as tf
import numpy as np
from .parameterized_trainer import Trainer
import pandas as pd
from tqdm import tqdm

class Distiller(Trainer):
    def __init__(self, dataloader, val_loader,
                 teacher: tf.keras.Model, student: tf.keras.Model,
                 distance_loss_fn: tf.keras.losses.Loss, depth_loss_fn: tf.keras.losses.Loss, distillation_loss_fn: tf.keras.losses.Loss,
                 optimizer: tf.keras.optimizers.Optimizer,
                 log_path: str, savepath: str,
                 alpha: float, temperature: float, 
                 use_mse=True):

        super(Distiller, self).__init__(dataloader = dataloader, val_loader = val_loader, optimizer = optimizer,
                                distance_loss_fn = distance_loss_fn, depth_loss_fn = depth_loss_fn,
                                log_path = log_path, savepath = savepath, use_mse = use_mse,
                                model=student)
        self.teacher = teacher
        # self.model = student
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha
        self.temperature = temperature
        self.loss_dict = { 'train_dist_loss': [], 'train_distill_loss': [], 'entropy_loss': [],
                           'val_dist_loss': [], 'val_depth_loss': [], 'val_entropy_loss': []}
        pd.DataFrame(self.loss_dict).to_csv(log_path)
        

    def train_on_batch(self, images,labels, derived_labels):
        # Forward pass of teacher
        teacher_x, teacher_y, teacher_z, _ = self.teacher(images, training=False)
        teacher_out = tf.concat([teacher_x, teacher_y, teacher_z], axis=-1)
        with tf.GradientTape() as tape:
            # Forward pass of student
            student_x, student_y, student_z, student_params = self.model(images, training=True)
            student_out = tf.concat([student_x, student_y, student_z], axis=-1)
        
            # Compute losses
            distance_loss = self.distance_loss_fn(student_out, labels)
            entropy_loss = self.crossentropy_loss(student_params, derived_labels)
            student_loss = distance_loss + entropy_loss

            # student_loss = self.model_loss_fn(y, student_out)
            # Compute scaled distillation loss from https://arxiv.org/abs/1503.02531
            # The magnitudes of the gradients produced by the soft targets scale
            # as 1/T^2, multiply them by T^2 when using both hard and soft targets.
            distillation_loss = (
                self.distillation_loss_fn(
                    tf.nn.softmax(teacher_out / self.temperature, axis=1),
                    tf.nn.softmax(student_out / self.temperature, axis=1),
                )
                * self.temperature**2
            )
            distance_loss   = tf.cast(distance_loss, tf.float32)
            entropy_loss    = tf.cast(entropy_loss, tf.float32)           
            student_loss   = tf.cast(student_loss, tf.float32)
            distillation_loss   = tf.cast(distillation_loss, tf.float32)
            alpha           = tf.cast(self.alpha, tf.float32)
            
            loss = alpha * student_loss + (1 - self.alpha) * distillation_loss

        # Compute gradients
        trainable_vars = self.model.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        return loss, distance_loss, distillation_loss, entropy_loss


    def train(self, epochs: int, save_checkpoint=True,
              parallel=True, early_stop=True):
        patience = 5
        wait = 0
        best = self.dataloader.dataset.MAX_VALUE
        for e in range(epochs):
            print(f'Epoch: {e}: ........................................')
            running_train_loss = 0.0
            running_dist_loss = 0.0
            running_distillation_loss = 0.0
            running_entropy_loss = 0.0
            

            #Train
            for batch_id, (images, labels, derived_labels) in enumerate(tqdm(self.dataloader.make_batch(), colour='#96c8a2')):
                loss, dist_loss, distillation_loss, entropy_loss = self.train_on_batch(images, labels, derived_labels)
                # Log every 200 batches (6400 imgs).
                if batch_id % 200 == 0:
                    print(
                        "Training loss (for one batch) at step %d, epoch %d: %.4f; distance_loss: %.4f,  distillation loss: %.4f, entropy loss: %.4f"
                        % (batch_id, e, float(loss), float(dist_loss), float(distillation_loss), float(entropy_loss))
                    )
                running_train_loss += loss*len(labels)
                running_dist_loss  += dist_loss * len(labels)
                running_distillation_loss += distillation_loss * len(labels)
                running_entropy_loss += entropy_loss * len(labels)
                # if batch_id >= len(self.dataloader):
                #     break

            running_train_loss /= self.dataloader.n
            running_dist_loss /= self.dataloader.n
            running_distillation_loss /= self.dataloader.n
            running_entropy_loss /= self.dataloader.n

            print(
                "Training loss (for one epoch) at epoch %d: %.4f; distill_loss: %.4f,  depth loss: %.4f, entropy loss: %.4f"
                % (e, float(running_train_loss), float(running_dist_loss), float(running_distillation_loss), float(running_entropy_loss))
            )

            train_loss_in_meters = float(self.convert_error_to_meter(running_dist_loss))

            self.loss_dict['train_dist_loss'].append(float(running_dist_loss))
            self.loss_dict['train_distill_loss'].append(float(running_distillation_loss))
            self.loss_dict['entropy_loss'].append(float(running_entropy_loss))

            running_val_loss = 0.0
            running_dist_loss = 0.0
            running_distillation_loss = 0.0
            running_entropy_loss = 0.0
            running_depth_loss = 0.0

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


