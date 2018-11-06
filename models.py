from tensorflow import keras
from tensorflow.keras.layers import Dense, Conv2D, Dropout, BatchNormalization, Activation, Add
from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D, Flatten
from tensorflow.keras.regularizers import l2


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
        x = Conv2D(64, kernel_size=5, padding='same', kernel_initializer='he_normal',
                   kernel_regularizer=l2(self.weight_decay), use_bias=False)(self.input_data)
        x = BatchNormalization(momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
        x = Activation('relu')(x)
        x = self.expand_conv(x, 16, k)

        for i in range(N - 1):
            x = self.conv1_block(x, k, dropout)

        x = BatchNormalization(momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
        x = Activation('relu')(x)
        x = self.expand_conv(x, 32, k, strides=(2, 2))

        for i in range(N - 1):
            x = self.conv2_block(x, k, dropout)

        x = BatchNormalization(momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
        x = Activation('relu')(x)
        x = self.expand_conv(x, 64, k, strides=(2, 2))

        for i in range(N - 1):
            x = self.conv3_block(x, k, dropout)

        x = BatchNormalization(momentum=0.1, epsilon=1e-5, gamma_initializer='uniform')(x)
        x = Activation('relu')(x)
        x = AveragePooling2D((8, 8))(x)
        x = Flatten()(x)
        outputs = Dense(nb_classes, activation='softmax')(x)

        return outputs
