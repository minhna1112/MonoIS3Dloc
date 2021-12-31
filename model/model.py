import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.layers import LeakyReLU

class SimpleNet(tf.keras.Model):
    def __init__(self, image_shape, activation='relu',input_shape=(224,398,1)):
        super(SimpleNet, self).__init__()

        self.image_shape = image_shape
        self.activation  = activation

        def downsample_convolution(out_channels=32, input_shape=None, name: str):
            if input_shape is not None:
                conv = Conv2D(out_channels, (3, 3), activation=self.activation, input_shape=input_shape)
            else:
                conv = Conv2D(out_channels, (3, 3), activation=self.activation)
            pool = MaxPooling2D(pool_size=(2, 2))

            return Sequential([conv, pool], name=name)

        self.conv1 = downsample_convolution(32, self.image_shape)
        self.conv2 = downsample_convolution(64)
        self.conv3 = downsample_convolution(128)
        self.conv4 = downsample_convolution(256)
        self.conv5 = downsample_convolution(256)

        self.flat = Flatten()
        self.dense1 = Dense(512, activation=self.activation)
        self.dense2 = Dense(64, activation=self.activation)

    def call(self, inputs, training=None, mask=None):
        out = self.conv1(inputs)
        out = self.conv2(out)
        out= self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)

        out = self.flat(out)
        out = self.dense1(out)
        out = (out)
        out = Dense(3, activation='tanh')(out)
        return out


def simple_net(image_shape=(224, 398, 1)):
    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation="relu", input_shape=image_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(256, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(256, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())

    model.add(Dense(512))
    model.add(Dense(64))
    model.add(Dense(3))
    model.add(Activation("tanh"))

    # plot_model(model, to_file="model.png")
    return model

if __name__ == '__main__':
    model1 = simple_net()
    model1.summary()

    model2 = SimpleNet(image_shape=(224,398,3))
    model2.build(input_shape=(None,224,398,3))
    model2.summary()