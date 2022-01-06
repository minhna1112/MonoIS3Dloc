import tensorflow as tf

class OptimizerFactory:
    def __init__(self, lr: float, use_scheduler=True, staircase=True):
        if use_scheduler:
            initial_learning_rate = lr
            lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate,
                decay_steps=6610,
                decay_rate=0.1,
                staircase=staircase)

            self.optimizer = tf.keras.optimizers.Adam(lr_schedule)
        else:
            self.optimizer = tf.keras.optimizers.Adam(lr)

    def get_optimizer(self):
        return self.optimizer