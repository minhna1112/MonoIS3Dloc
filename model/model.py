import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.layers import LeakyReLU


class PredictionHead(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.sigmoid = tf.keras.layers.Activation('sigmoid')
        self.tanh = tf.keras.layers.Activation('tanh')

    def call(self, input_tensor):
        x, y, z = tf.split(value=input_tensor, num_or_size_splits=3, axis=-1)
        x = self.sigmoid(x)
        y = self.tanh(y)
        z = self.tanh(z)
        return x, y, z

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

class BackBone(tf.keras.Model):
    def __init__(self, activation='relu',input_shape=(224,398,1), num_ext_conv = 0, ksize=3, num_branch=1):
        super().__init__()
        self.activation  = activation
        self.num_ext_conv  = num_ext_conv
        self.ksize = ksize
        self.num_branch = num_branch

        def downsample_convolution(out_channels=32,  name='conv'):

            conv = Conv2D(out_channels, (self.ksize, self.ksize), activation=self.activation, padding='same')
            pool = MaxPooling2D(pool_size=(2, 2))

            return Sequential([conv, pool], name=name)

        self.conv_base = Sequential([
            downsample_convolution(32, name='conv1'),
            downsample_convolution(64, name='conv2'),
            downsample_convolution(128, name='conv3'),
            # downsample_convolution(256, name='conv4')
        ], name='conv_base')

        if self.num_ext_conv > 0 :
            if num_branch <= 1:
                self.conv_ext = Sequential([
                    downsample_convolution(256, name=f'conv4_{i}')  for i in range(self.num_ext_conv)
                ], name='conv_ext')
            else: 
                self.conv_ext = []
                for j in range(self.num_branch):
                    self.conv_ext.append(
                        Sequential([ downsample_convolution(256, name=f'conv{4+i}')  for i in range(self.num_ext_conv)
                        ], name=f'branch{j}')
                    )

    
    def call(self, inputs, training=None, mask=None):
        out = self.conv_base(inputs)
        if self.num_ext_conv > 0:
            if self.num_branch <= 1:
                out = self.conv_ext(out)
            else:
                final_out = []
                for branch in self.conv_ext:
                    final_out.append(branch(out))
                out = final_out

        return out

class ParameterizedNet(tf.keras.Model):
    def __init__(self, activation='relu',input_shape=(224,398,1), num_ext_conv = 0, ksize=3, num_params = 3):
        super().__init__()
        self.activation  = activation
        self.num_ext_conv  = num_ext_conv
        self.ksize = ksize
        self.backbone = BackBone(activation, input_shape, num_ext_conv=1, ksize=3, num_branch=2)
        self.flatten1 = tf.keras.layers.Flatten()
        self.flatten2 = tf.keras.layers.Flatten()
        self.dense1 = Dense(128, activation=self.activation, name='dense1')
        self.dense2 = Dense(128, activation=self.activation, name='dense2')
        self.dense3 = Dense(3,  name='dense3')
        self.dense4 = Dense(3,  name='dense4')
        self.parameterized_layer = tf.keras.layers.Activation('softmax')
        self.prediction_head = PredictionHead()

    def call(self, inputs, training=None, mask=None):
        [out_1, out_2] = self.backbone(inputs)
        out_1 = self.flatten1(out_1)
        out_1 = self.dense1(out_1)
        out_1 = self.dense3(out_1)
        out_1 = self.parameterized_layer(out_1)

        # out_2 = self.backbone2(inputs)
        out_2 = self.flatten2(out_2)
        out_2 = self.dense2(out_2)
        out_2 = tf.concat([out_1, out_2], axis=-1)
        out_2 = self.dense4(out_2)
        
        x, y, z = self.prediction_head(out_2)
        
        return x, y, z, out_1

class BackboneSharedParameterizedNet(tf.keras.Model):
    def __init__(self, activation='relu',input_shape=(224,398,1), num_ext_conv = 0, ksize=3, num_params = 3):
        super().__init__()
        self.activation  = activation
        self.num_ext_conv  = num_ext_conv
        self.ksize = ksize
        self.backbone1 = BackBone(activation, input_shape, num_ext_conv, ksize)
        self.flatten1 = tf.keras.layers.Flatten()

        self.dense1 = Dense(128, activation=self.activation, name='dense1')
        self.dense2 = Dense(128, activation=self.activation, name='dense2')
        self.dense3 = Dense(3,  name='dense3')
        self.dense4 = Dense(3,  name='dense4')
        self.parameterized_layer = tf.keras.layers.Activation('softmax')
        self.prediction_head = PredictionHead()
        

    def call(self, inputs, training=None, mask=None):

        out_1 = self.backbone1(inputs)
        flattened = self.flatten1(out_1)

        out_1 = self.dense1(flattened)
        out_1 = self.dense3(out_1)
        out_1 = self.parameterized_layer(out_1)

        out_2 = self.dense2(flattened)
        out_2 = tf.concat([out_1, out_2], axis=-1)
        out_2 = self.dense4(out_2)
        
        x, y, z = self.prediction_head(out_2)
        
        return x, y, z, out_1

    
    

class DepthAwareNet(tf.keras.Model):
    def __init__(self, activation='relu',input_shape=(224,398,1), num_ext_conv = 0, ksize=3):
        super().__init__()
        self.activation  = activation
        self.num_ext_conv  = num_ext_conv
        self.ksize = ksize

        def downsample_convolution(out_channels=32,  name='conv'):

            conv = Conv2D(out_channels, (self.ksize, self.ksize), activation=self.activation, padding='same')
            pool = MaxPooling2D(pool_size=(2, 2))

            return Sequential([conv, pool], name=name)

        self.conv_base = Sequential([
            downsample_convolution(32, name='conv1'),
            downsample_convolution(64, name='conv2'),
            downsample_convolution(128, name='conv3'),
            downsample_convolution(256, name='conv4')
        ], name='conv_base')

        if self.num_ext_conv > 0 :
            self.conv_ext = Sequential([
                downsample_convolution(256, name=f'conv{4+i}') for i in range(self.num_ext_conv)
            ], name='conv_ext')

        self.flat = Flatten()
        self.dense1 = Dense(128, activation=self.activation, name='dense1')
        #self.dense2 = Dense(64, activation=self.activation, name='dense2')
        self.dense3 = Dense(3, name='dense3')
        self.sigmoid = tf.keras.layers.Activation('sigmoid')
        self.tanh = tf.keras.layers.Activation('tanh')

    def call(self, inputs, training=None, mask=None):

        out = self.conv_base(inputs)

        if self.num_ext_conv > 0:
            out = self.conv_ext(out)

        out = self.flat(out)
        out = self.dense1(out)
        out = self.dense3(out)
        # x, y, z = tf.split(value=out, num_or_size_splits=3, axis=-1)
        # x = self.sigmoid(x)
        # y = self.tanh(y)
        # z = self.tanh(z)

        x, y, z = self.prediction_head(out)

        return x,y,z #[(batch_size, 1), (batch_size, 1), (batch_size, 1)]


if __name__ == '__main__':
    
    # model3 = DepthAwareNet(num_ext_conv=0)
    # model3.build(input_shape=(None, 224, 398, 1))
    # model3.summary()
    # print(f'Total params: {model3.count_params()}')


    model3 = ParameterizedNet(num_ext_conv=1)
    model3.build(input_shape=(None, 180, 320, 1))
    model3.summary()
    print(f'Total params: {model3.count_params()}')

    # model3 = BackboneSharedParameterizedNet(num_ext_conv=1)
    # model3.build(input_shape=(None, 180, 320, 1))
    # model3.summary()
    # print(f'Total params: {model3.count_params()}')
