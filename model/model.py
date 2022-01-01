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

        def downsample_convolution(out_channels=32, input_shape=None, name='conv'):
            if input_shape is not None:
                conv = Conv2D(out_channels, (3, 3), activation=self.activation, input_shape=input_shape)
            else:
                conv = Conv2D(out_channels, (3, 3), activation=self.activation)
            pool = MaxPooling2D(pool_size=(2, 2))

            return Sequential([conv, pool], name=name)

        self.conv1 = downsample_convolution(32, name='conv1')
        self.conv2 = downsample_convolution(64, name='conv2')
        self.conv3 = downsample_convolution(128, name='conv3')
        self.conv4 = downsample_convolution(256, name='conv4')
        self.conv5 = downsample_convolution(256, name='conv5')
        self.conv6 = downsample_convolution(256, name='conv6')

        self.flat = Flatten()
        self.dense1 = Dense(512, activation=self.activation, name='dense1')
        self.dense2 = Dense(64, activation=self.activation, name='dense2')
        self.dense3 = Dense(3, activation='tanh', name='out')

    def call(self, inputs, training=None, mask=None):
        out = self.conv1(inputs)
        out = self.conv2(out)
        out= self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        #print(out.shape)
        out = self.conv6(out)
        #print(out.shape)
        out = self.flat(out)
        #print(out.shape)
        out = self.dense1(out)
        out = self.dense2(out)
        out = self.dense3(out)
        return out

class DepthAwareNet(tf.keras.Model):
    def __init__(self, activation='relu',input_shape=(224,398,1)):
        super().__init__()
        self.activation  = activation

        def downsample_convolution(out_channels=32, input_shape=None, name='conv'):
            if input_shape is not None:
                conv = Conv2D(out_channels, (3, 3), activation=self.activation, input_shape=input_shape)
            else:
                conv = Conv2D(out_channels, (3, 3), activation=self.activation)
            pool = MaxPooling2D(pool_size=(2, 2))

            return Sequential([conv, pool], name=name)

        self.conv1 = downsample_convolution(32, name='conv1')
        self.conv2 = downsample_convolution(64, name='conv2')
        self.conv3 = downsample_convolution(128, name='conv3')
        self.conv4 = downsample_convolution(256, name='conv4')
        self.conv5 = downsample_convolution(256, name='conv5')
        self.conv6 = downsample_convolution(256, name='conv6')

        self.flat = Flatten()
        self.dense1 = Dense(512, activation=self.activation, name='dense1')
        self.dense2 = Dense(64, activation=self.activation, name='dense2')
        self.dense3 = Dense(3, name='dense3')
        self.sigmoid = tf.keras.layers.Activation('sigmoid')
        self.tanh = tf.keras.layers.Activation('tanh')

    def call(self, inputs, training=None, mask=None):
        out = self.conv1(inputs)
        out = self.conv2(out)
        out= self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        #print(out.shape)
        out = self.conv6(out)
        #print(out.shape)
        out = self.flat(out)
        #print(out.shape)
        out = self.dense1(out)
        out = self.dense2(out)
        out = self.dense3(out)
        x, y, z = tf.split(value=out, num_or_size_splits=3, axis=-1)
        x = self.sigmoid(x)
        y = self.tanh(y)
        z = self.tanh(z)
        return x,y,z #[(batch_size, 1), (batch_size, 1), (batch_size, 1)]


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
    #model1 = simple_net()
    #model1.summary()

    # model2 = SimpleNet(image_shape=(224,398,1))
    # model2.build(input_shape=(None,224,398,1))
    # model2.summary()
    #
    #
    # x = tf.ones(shape=(4, 224,398, 1))
    # #assert model2(x).shape == model1(x).shape
    # print(model2(x).shape) #(Batch_size, 3)
    #

    model3 = DepthAwareNet()
    model3.build(input_shape=(None, 224, 398, 1))
    model3.summary()

    x = tf.ones(shape=(4, 224, 398, 1))
    # assert model2(x).shape == model1(x).shape
    print(model3(x))  # [(Batch_size, 1), (Batch_size, 1),(Batch_size, 1)[