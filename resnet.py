import tensorflow as tf
import sys

class ResNet():
    def __init__(self, training=True):
        self.training = training

    def add_residual_block(self, input, block_number, in_channels, out_channels):
        block_number = str(block_number) #This was used for providing a unique name to each layer.
        skip = tf.identity(input)
        
        down = 1
        if in_channels != out_channels:
            # TODO: perform 1x1 convolution to match output dimensions for skip connection
            skip = tf.keras.layers.Conv2D(filters=out_channels, kernel_size=1, strides=2, padding='same')(skip)
            skip = tf.keras.layers.BatchNormalization()(skip)
            down = 2

        #TODO: Implement one residual block (Convolution, batch_norm, relu)
        input = tf.keras.layers.Conv2D(filters=out_channels, kernel_size=3, strides=down, padding='same')(input)
        input = tf.keras.layers.BatchNormalization()(input)
        input = tf.nn.relu(input)

        input = tf.keras.layers.Conv2D(filters=out_channels, kernel_size=3, strides=1, padding='same')(input)
        input = tf.keras.layers.BatchNormalization()(input)

        #TODO: Add the skip connection and ReLU the output
        input = input + skip
        return tf.nn.relu(input)

    def forward(self, data):
        #TODO: 64 7x7 convolutions followed by batchnorm, relu, 3x3 maxpool with stride 2
        data = tf.keras.layers.Conv2D(filters=64, name='conv_layer_0', kernel_size=7, strides=2, input_shape=(32, 32, 3,), padding="same")(data)
        data = tf.keras.layers.BatchNormalization()(data)
        data = tf.keras.layers.Activation('relu')(data)
        data = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), padding='same', strides=2)(data)

        #TODO: Add residual blocks of the appropriate size. See the diagram linked in the README for more details on the architecture.
        # Use the add_residual_block helper function
        data = self.add_residual_block(data, 1, 64, 64)
        data = self.add_residual_block(data, 2, 64, 64)
        data = self.add_residual_block(data, 3, 64, 128)
        data = self.add_residual_block(data, 4, 128, 128)
        data = self.add_residual_block(data, 5, 128, 256)
        data = self.add_residual_block(data, 6, 256, 256)
        data = self.add_residual_block(data, 7, 256, 512)
        data = self.add_residual_block(data, 8, 512, 512)

        #TODO: perform global average pooling on each feature map to get 4 output channels
        data = tf.keras.layers.GlobalAveragePooling2D()(data)
        logits = tf.keras.layers.Dense(4)(data)
        logits = tf.nn.softmax(logits)
        return logits

    def add_convolution(self,
                        input,
                        name,
                        filter_size,
                        input_channels,
                        output_channels,
                        padding):
        #TODO: Implement a convolutional layer with the above specifications
        return tf.nn.conv2d(input, filter=filter_size, padding=padding, strides=None, filters=output_channels, name=name, input_shape=input_channels)



        
