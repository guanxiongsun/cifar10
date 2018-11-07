"""DenseNet models for Keras.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow import keras
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Activation, \
    concatenate, GlobalAveragePooling2D, ZeroPadding2D
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
    x = BatchNormalization(epsilon=1.001e-4)(x)
    x = Activation('relu')(x)
    x = Conv2D(int(keras.backend.int_shape(x)[bn_axis] * reduction), 1, kernel_regularizer=l2(1e-4), use_bias=False)(x)
    x = AveragePooling2D(2, strides=2)(x)
    return x


def conv_block(x, growth_rate):
    """A building block for a dense block."""
    x1 = BatchNormalization(epsilon=1.001e-5)(x)
    x1 = Activation('relu')(x1)
    x1 = Conv2D(4 * growth_rate, 1, kernel_regularizer=l2(1e-4), use_bias=False)(x1)
    x1 = BatchNormalization(epsilon=1.001e-5)(x1)
    x1 = Activation('relu')(x1)
    x1 = Conv2D(growth_rate, 3, kernel_regularizer=l2(1e-4), padding='same', use_bias=False)(x1)
    x = concatenate([x, x1])
    return x


def DenseNet(blocks=[6, 12, 24, 16], classes=1000):
    """Instantiates the DenseNet architecture.
    # Returns
        A Keras model instance.
    """
    x = ZeroPadding2D(padding=((3, 3), (3, 3)))(img_input)
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
    x = Dense(classes, kernel_regularizer=l2(1e-4), activation='softmax')(x)

    return x
