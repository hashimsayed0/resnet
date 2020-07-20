import tensorflow as tf
import sys

class ResNet():
    def __init__(self, training=True):
        self.training = training

    def add_residual_block(self, input, block_number, in_channels, out_channels):
        block_number = str(block_number) # used for providing a unique name to each layer.
        skip = tf.identity(input)
        
        down = 1
        if in_channels != out_channels:
            # performs 1x1 convolution to match output dimensions for skip connection
            skip = tf.keras.layers.Conv2D(filters=out_channels, kernel_size=1, strides=2, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False))(skip)
            skip = tf.keras.layers.BatchNormalization()(skip)
            down = 2

        # one residual block (Convolution, batch_norm, relu)
        input = tf.keras.layers.Conv2D(filters=out_channels, kernel_size=3, strides=down, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False))(input)
        input = tf.keras.layers.BatchNormalization()(input)
        input = tf.nn.relu(input)

        input = tf.keras.layers.Conv2D(filters=out_channels, kernel_size=3, strides=1, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False))(input)
        input = tf.keras.layers.BatchNormalization()(input)

        #skip connection and ReLU the output
        input = input + skip
        return tf.nn.relu(input)

    def forward(self, data):
        # 64 7x7 convolutions followed by batchnorm, relu, 3x3 maxpool with stride 2
        data = self.add_convolution(data, 'conv_layer_0', 7, 32, 64, 'same')(data)
        data = tf.keras.layers.BatchNormalization()(data)
        data = tf.keras.layers.Activation('relu')(data)
        data = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), padding='same', strides=2)(data)
        # adds residual blocks of the appropriate size. 
        data = self.add_residual_block(data, 1, 64, 64)
        data = self.add_residual_block(data, 2, 64, 64)
        data = self.add_residual_block(data, 3, 64, 128)
        data = self.add_residual_block(data, 4, 128, 128)
        data = self.add_residual_block(data, 5, 128, 256)
        data = self.add_residual_block(data, 6, 256, 256)
        data = self.add_residual_block(data, 7, 256, 512)
        data = self.add_residual_block(data, 8, 512, 512)

        print(data.shape)
        # performs global average pooling on each feature map to get 4 output channels
        data = tf.keras.layers.GlobalAveragePooling2D()(data)
        logits = tf.keras.layers.Dense(4)(data)
        return logits

    def add_convolution(self,
                        input,
                        name,
                        filter_size,
                        input_channels,
                        output_channels,
                        padding):
        # implements a convolutional layer with the above specifications
        return tf.keras.layers.Conv2D(filters=output_channels, kernel_size=filter_size, strides=2, padding=padding, name=name, input_shape=(input_channels, input_channels, 3), kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False))



        
