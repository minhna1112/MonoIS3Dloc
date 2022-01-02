import tensorflow as tf
import tensorflow.keras.backend as K

class L2NormRMSE(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        #distance = tf.reduce_sum(tf.square(y_true), axis=-1)  # Shape ()
        mse = tf.reduce_sum(tf.square(y_pred - y_true), axis=-1)  # Shape (1,)
        return tf.cast(tf.math.sqrt(mse), dtype=tf.float32)  # shape (1,)

class L2DepthLoss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        #distance = tf.reduce_sum(tf.square(y_true), axis=-1)  # Shape ()
        #y_pred = tf.expand_dims(y_pred, axis=-1)
        #y_true = tf.expand_dims(y_true, axis=-1)
        mse = tf.square(y_pred - y_true)  # Shape (1,)
        return tf.cast(tf.math.sqrt(mse), dtype=tf.float32)  # shape (1,)

if __name__ == '__main__':
    x = tf.ones(shape=(4,3))
    print(x[..., 0].shape)
    y = 3*tf.ones(shape=(4, 3))
    loss_fn = L2NormRMSE()
    depth_loss_fn = L2DepthLoss()
    print(loss_fn(x, y)+.5*depth_loss_fn(x[..., 0],y[..., 0])) #Should be equal to sqrt 12 + 0.5 sqrt 4