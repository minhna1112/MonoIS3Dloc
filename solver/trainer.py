import tensorflow as tf
from tqdm import tqdm

class Trainer():
    def __init__(self, dataloader, model: tf.keras.Model, distance_loss_fn: tf.keras.losses.Loss, depth_loss_fn: tf.keras.losses.Loss,optimizer, saved_checkpoints: str):
        self.dataloader  = dataloader
        self.model = model
        self.distance_loss_fn = distance_loss_fn
        self.depth_loss_fn = depth_loss_fn
        self.optimizer = optimizer
        self.batch_size = self.dataloader.batch_size
        self.alpha = 0.5
        
    def train_on_batch(self, images,labels):
        with tf.GradientTape() as tape:
            #Perform forward pass 
            out_x, out_y, out_z = self.model(images)
            #Calculate loss
            if self.depth_loss_fn is not None:
                label_x = labels[..., 0].reshape((self.batch_size, 1))
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

    def train(self, epochs: int):
        for e in range(epochs):
            print(f'Epoch: {e+1}: ...................')
            for batch_id, (images, labels) in enumerate(tqdm(self.dataloader, color='96c8a2')):
                loss, dist_loss, depth_loss= self.train_on_batch(images, labels)
                # Log every 200 batches.
                if batch_id % 200 == 0:
                    print(
                        "Training loss (for one batch) at step %d: %.4f"
                        % (batch_id, float(loss))
                    )