from tensorflow import keras
from tensorflow.keras.layers import Dense, Conv2D, Dropout, BatchNormalization, Activation, \
    Add, concatenate, GlobalAveragePooling2D, ZeroPadding2D
from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D, Flatten
from tensorflow.keras.regularizers import l2


def dense_block(x, blocks):
    """A dense block. """
    for i in range(blocks):
        x = conv_block(x, 32)
    return x


def transition_block(x, reduction):
    """A transition block."""
    bn_axis = 3
    x = BatchNormalization(epsilon=1.001e-5)(x)
    x = Activation('relu')(x)
    x = Conv2D(int(keras.backend.int_shape(x)[bn_axis] * reduction), 1, kernel_regularizer=l2(1e-5), use_bias=False)(x)
    x = AveragePooling2D(2, strides=2)(x)
    return x


def conv_block(x, growth_rate):
    """A building block for a dense block."""
    x1 = BatchNormalization(epsilon=1.001e-5)(x)
    x1 = Activation('relu')(x1)
    x1 = Conv2D(4 * growth_rate, 1, kernel_regularizer=l2(1e-5), use_bias=False)(x1)
    x1 = BatchNormalization(epsilon=1.001e-5)(x1)
    x1 = Activation('relu')(x1)
    x1 = Conv2D(growth_rate, 3, kernel_regularizer=l2(1e-5), padding='same', use_bias=False)(x1)
    x = concatenate([x, x1])
    return x


class AlexDense:

    def __init__(self, input_data):
        self.input_data = input_data

    def get_output(self):
        return self.alex_dense()

    def alex_dense(self):
        """Instantiates the DenseNet architecture.
        # Returns
            A Keras model instance.
        """
        blocks = [6, 12, 24, 16]

        x = ZeroPadding2D(padding=((3, 3), (3, 3)))(self.input_data)
        x = Conv2D(64, 7, strides=2, use_bias=False)(x)
        x = BatchNormalization(epsilon=1.001e-5)(x)
        x = Activation('relu')(x)
        x = ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
        x = MaxPooling2D(3, strides=2)(x)

        x = dense_block(x, blocks[0])
        x = transition_block(x, 0.5)
        x = dense_block(x, blocks[1])
        x = transition_block(x, 0.5)
        x = dense_block(x, blocks[2])
        x = transition_block(x, 0.5)
        x = dense_block(x, blocks[3])

        x = BatchNormalization(epsilon=1.001e-5)(x)
        x = Activation('relu')(x)

        x = GlobalAveragePooling2D()(x)
        model_output = Dense(10, activation='softmax')(x)

        return model_output


class AlexNet:

    def __init__(self, input_data):
        self.input_data = input_data

    def get_output(self):
        return self.alex_net()

    def alex_net(self, num_classes=10):

        # Conv1
        x = Conv2D(96, kernel_size=11, padding='same', strides=2, kernel_regularizer=l2(1e-4),
                   activation='relu')(self.input_data)
        x = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)
        # Conv2
        x = Conv2D(256, kernel_size=5, padding='same', strides=1, kernel_regularizer=l2(1e-4), activation='relu')(x)
        x = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)
        # Conv3
        x = Conv2D(384, kernel_size=3, padding='same', strides=1, kernel_regularizer=l2(1e-4), activation='relu')(x)
        x = Conv2D(384, kernel_size=3, padding='same', strides=1, kernel_regularizer=l2(1e-4), activation='relu')(x)
        x = Conv2D(256, kernel_size=3, padding='same', strides=1, kernel_regularizer=l2(1e-4), activation='relu')(x)
        x = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)
        # Flatten
        y = Flatten()(x)
        # Dense1
        y = Dropout(0.5)(Dense(1024, activation='relu')(y))
        # Dense2
        y = Dropout(0.5)(Dense(1024, activation='relu')(y))
        # Output
        model_output = Dense(num_classes, activation='softmax')(y)

        return model_output


class AlexVGG:

    def __init__(self, input_data):
        self.input_data = input_data

    def get_output(self):
        return self.alex_net()

    def alex_net(self, num_classes=10):
        # Block 1
        x = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(self.input_data)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

        # Block 2
        x = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

        # Block 3
        x = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

        # Flatten
        y = Flatten()(x)
        # Dense1
        y = Dropout(0.5)(Dense(4096, kernel_regularizer=l2(1e-4), kernel_initializer='he_normal', activation='relu')(y))
        # Dense2
        y = Dropout(0.5)(Dense(4096, kernel_regularizer=l2(1e-4), kernel_initializer='he_normal', activation='relu')(y))
        # Output
        model_output = Dense(num_classes, kernel_regularizer=l2(1e-4), kernel_initializer='he_normal', activation='softmax')(y)

        return model_output


class AlexResidual:

    def __init__(self, input_data):
        self.input_data = input_data

    def get_output(self):
        return self.alex_residual()

    def alex_residual(self, num_filters=64, num_blocks=4, num_sub_blocks=2, num_classes=10):
        x = Conv2D(num_filters, kernel_size=7, padding='same', strides=2, kernel_initializer='he_normal',
                   kernel_regularizer=l2(1e-4))(self.input_data)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        # Instantiate residual blocks
        for i in range(num_blocks):
            for j in range(num_sub_blocks):
                strides = 1
                is_first_layer_but_not_first_block = j == 0 and i > 0
                if is_first_layer_but_not_first_block:
                    strides = 2
                y = Conv2D(num_filters, kernel_size=3, padding='same', strides=strides, kernel_initializer='he_normal',
                           kernel_regularizer=l2(1e-4))(x)
                y = BatchNormalization()(y)
                y = Activation('relu')(y)
                y = Conv2D(num_filters, kernel_size=3, padding='same', kernel_initializer='he_normal',
                           kernel_regularizer=l2(1e-4))(y)
                y = BatchNormalization()(y)
                if is_first_layer_but_not_first_block:
                    x = Conv2D(num_filters, kernel_size=1, padding='same', strides=2, kernel_initializer='he_normal',
                               kernel_regularizer=l2(1e-4))(x)
                x = keras.layers.add([x, y])
                x = Activation('relu')(x)
            num_filters = 2 * num_filters
        x = AveragePooling2D()(x)
        y = Flatten()(x)
        model_output = Dense(num_classes, activation='softmax', kernel_initializer='he_normal')(y)

        return model_output


class AlexWRN:
    
    def __init__(self, input_data):
        self.input_data = input_data
        self.weight_decay = 0.0005

    def get_output(self):
        return self.alex_wrn()

    def expand_conv(self, init, base, k, strides=1):
        x = Conv2D(base * k, (3, 3), padding='same', strides=strides, kernel_initializer='he_normal',
                   kernel_regularizer=l2(self.weight_decay), use_bias=False)(init)
        x = BatchNormalization(momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
        x = Activation('relu')(x)
        x = Conv2D(base * k, (3, 3), padding='same', kernel_initializer='he_normal',
                   kernel_regularizer=l2(self.weight_decay), use_bias=False)(x)
        skip = Conv2D(base * k, (1, 1), padding='same', strides=strides, kernel_initializer='he_normal',
                      kernel_regularizer=l2(self.weight_decay), use_bias=False)(init)
        m = Add()([x, skip])

        return m

    def conv1_block(self, input, k=1, dropout=0.0):
        init = input
        x = BatchNormalization(momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(input)
        x = Activation('relu')(x)
        x = Conv2D(16 * k, (3, 3), padding='same', kernel_initializer='he_normal',
                   kernel_regularizer=l2(self.weight_decay), use_bias=False)(x)
        if dropout > 0.0:
            x = Dropout(dropout)(x)
        x = BatchNormalization(momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
        x = Activation('relu')(x)
        x = Conv2D(16 * k, (3, 3), padding='same', kernel_initializer='he_normal',
                   kernel_regularizer=l2(self.weight_decay), use_bias=False)(x)
        m = Add()([init, x])
        return m

    def conv2_block(self, input, k=1, dropout=0.0):
        init = input
        x = BatchNormalization(momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(input)
        x = Activation('relu')(x)
        x = Conv2D(32 * k, (3, 3), padding='same', kernel_initializer='he_normal',
                   kernel_regularizer=l2(self.weight_decay), use_bias=False)(x)
        if dropout > 0.0:
            x = Dropout(dropout)(x)
        x = BatchNormalization(momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
        x = Activation('relu')(x)
        x = Conv2D(32 * k, (3, 3), padding='same', kernel_initializer='he_normal',
                   kernel_regularizer=l2(self.weight_decay), use_bias=False)(x)
        m = Add()([init, x])

        return m

    def conv3_block(self, input, k=1, dropout=0.0):
        init = input
        x = BatchNormalization(momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(input)
        x = Activation('relu')(x)
        x = Conv2D(64 * k, (3, 3), padding='same', kernel_initializer='he_normal',
                   kernel_regularizer=l2(self.weight_decay), use_bias=False)(x)
        if dropout > 0.0:
            x = Dropout(dropout)(x)
        x = BatchNormalization(momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
        x = Activation('relu')(x)
        x = Conv2D(64 * k, (3, 3), padding='same', kernel_initializer='he_normal',
                   kernel_regularizer=l2(self.weight_decay), use_bias=False)(x)
        m = Add()([init, x])

        return m

    def alex_wrn(self, nb_classes=10, N=4, k=10, dropout=0.0):
        """
        Creates a Wide Residual Network
        :param input: Input Keras object
        :param nb_classes: Number of output classes
        :param N: Repeate conv1,2,3 N-1 times. depth = 1 + 9 + 6(N-1) = 6*N + 4 .
        :param k: Width of the network.
        :param dropout: Adds dropout if value is greater than 0.0
        :return: outputs
        """
        x = Conv2D(64, kernel_size=3, padding='same', kernel_initializer='he_normal',
                   kernel_regularizer=l2(self.weight_decay), use_bias=False)(self.input_data)
        x = BatchNormalization(momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
        x = Activation('relu')(x)
        x = self.expand_conv(x, 16, k)

        for i in range(N - 1):
            x = self.conv1_block(x, k, dropout)
        x = BatchNormalization(momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
        x = Activation('relu')(x)

        x = self.expand_conv(x, 32, k, strides=2)

        for i in range(N - 1):
            x = self.conv2_block(x, k, dropout)
        x = BatchNormalization(momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
        x = Activation('relu')(x)

        x = self.expand_conv(x, 64, k, strides=2)

        for i in range(N - 1):
            x = self.conv3_block(x, k, dropout)
        x = BatchNormalization(momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
        x = Activation('relu')(x)

        x = AveragePooling2D((8, 8))(x)
        x = Flatten()(x)
        outputs = Dense(nb_classes, activation='softmax')(x)

        return outputs


def conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(1, 1)):
    """Helper function conv + BN + ReLU for inception."""

    x = Conv2D(filters, (num_row, num_col), kernel_regularizer=l2(1e-5), strides=strides, padding=padding, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x


class AlexInception:

    def __init__(self, input_data):
        self.input_data = input_data

    def get_output(self):
        return self.alex_inception()

    def alex_inception(self, classes=10):
        """Inception architecture."""

        x = conv2d_bn(self.input_data, 32, 3, 3, strides=1, padding='valid')
        x = conv2d_bn(x, 32, 3, 3, padding='valid')
        x = conv2d_bn(x, 64, 3, 3)
        x = MaxPooling2D((3, 3), strides=(2, 2))(x)

        # mixed 0, 1, 2: 35 x 35 x 256
        branch1x1 = conv2d_bn(x, 64, 1, 1)

        branch5x5 = conv2d_bn(x, 48, 1, 1)
        branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

        branch3x3dbl = conv2d_bn(x, 64, 1, 1)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

        branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 32, 1, 1)
        x = concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool])

        # mixed 1: 35 x 35 x 256
        branch1x1 = conv2d_bn(x, 64, 1, 1)

        branch5x5 = conv2d_bn(x, 48, 1, 1)
        branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

        branch3x3dbl = conv2d_bn(x, 64, 1, 1)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

        branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 64, 1, 1)
        x = concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool])

        # mixed 2: 35 x 35 x 256
        branch1x1 = conv2d_bn(x, 64, 1, 1)

        branch5x5 = conv2d_bn(x, 48, 1, 1)
        branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

        branch3x3dbl = conv2d_bn(x, 64, 1, 1)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

        branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 64, 1, 1)
        x = concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool])

        # mixed 3: 17 x 17 x 768
        branch3x3 = conv2d_bn(x, 384, 3, 3, strides=(2, 2), padding='valid')

        branch3x3dbl = conv2d_bn(x, 64, 1, 1)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3, strides=(2, 2), padding='valid')

        branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
        x = concatenate([branch3x3, branch3x3dbl, branch_pool])

        # mixed 4: 17 x 17 x 768
        branch1x1 = conv2d_bn(x, 192, 1, 1)

        branch7x7 = conv2d_bn(x, 128, 1, 1)
        branch7x7 = conv2d_bn(branch7x7, 128, 1, 7)
        branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

        branch7x7dbl = conv2d_bn(x, 128, 1, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 1, 7)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

        branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
        x = concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool])

        # mixed 5, 6: 17 x 17 x 768
        for i in range(2):
            branch1x1 = conv2d_bn(x, 192, 1, 1)

            branch7x7 = conv2d_bn(x, 160, 1, 1)
            branch7x7 = conv2d_bn(branch7x7, 160, 1, 7)
            branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

            branch7x7dbl = conv2d_bn(x, 160, 1, 1)
            branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
            branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 1, 7)
            branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
            branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

            branch_pool = AveragePooling2D(
                (3, 3), strides=(1, 1), padding='same')(x)
            branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
            x = concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool])

        # mixed 7: 17 x 17 x 768
        branch1x1 = conv2d_bn(x, 192, 1, 1)

        branch7x7 = conv2d_bn(x, 192, 1, 1)
        branch7x7 = conv2d_bn(branch7x7, 192, 1, 7)
        branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

        branch7x7dbl = conv2d_bn(x, 192, 1, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

        branch_pool = AveragePooling2D((3, 3), padding='same', strides=(1, 1))(x)
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
        x = concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool])

        # Classification block
        x = GlobalAveragePooling2D(name='avg_pool')(x)
        model_output = Dense(classes, activation='softmax', name='predictions')(x)

        return model_output
