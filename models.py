from tensorflow import keras
from tensorflow.keras.layers import Dense, Conv2D, Dropout, BatchNormalization, Activation
from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D, Input, Flatten
from tensorflow.keras.regularizers import l2


def alex_net(input_data, num_classes=10):

    # Conv1
    x = Conv2D(96, kernel_size=11, padding='same', strides=2, kernel_regularizer=l2(1e-4), activation='relu')(input_data)
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


def alex_residual(input_data, num_filters=64, num_blocks=4, num_sub_blocks=2, num_classes=10):

    x = Conv2D(num_filters, kernel_size=7, padding='same', strides=2, kernel_initializer='he_normal',
               kernel_regularizer=l2(1e-4))(input_data)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Instantiate convolutional base (stack of blocks).
    for i in range(num_blocks):
        for j in range(num_sub_blocks):
            strides = 1
            is_first_layer_but_not_first_block = j == 0 and i > 0
            if is_first_layer_but_not_first_block:
                strides = 2
            y = Conv2D(num_filters,
                       kernel_size=3,
                       padding='same',
                       strides=strides,
                       kernel_initializer='he_normal',
                       kernel_regularizer=l2(1e-4))(x)
            y = BatchNormalization()(y)
            y = Activation('relu')(y)
            y = Conv2D(num_filters,
                       kernel_size=3,
                       padding='same',
                       kernel_initializer='he_normal',
                       kernel_regularizer=l2(1e-4))(y)
            y = BatchNormalization()(y)
            if is_first_layer_but_not_first_block:
                x = Conv2D(num_filters,
                           kernel_size=1,
                           padding='same',
                           strides=2,
                           kernel_initializer='he_normal',
                           kernel_regularizer=l2(1e-4))(x)
            x = keras.layers.add([x, y])
            x = Activation('relu')(x)
        num_filters = 2 * num_filters

    x = AveragePooling2D()(x)
    y = Flatten()(x)
    model_output = Dense(num_classes, activation='softmax', kernel_initializer='he_normal')(y)
    return model_output
