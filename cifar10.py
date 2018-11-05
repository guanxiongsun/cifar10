from __future__ import print_function
from tensorflow import keras
from tensorflow.keras.layers import Dense, Conv2D, Dropout, BatchNormalization, Activation
from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D, Input, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
import numpy as np
import os
from random_eraser import get_random_eraser

model_names = ['alex_net', 'alex_residual', 'alex_dense']
model_name = model_names[1]

save_path = 'checkpoints'

random_erasing = True

# Training params.
batch_size = 16
epochs = 50
num_classes = 10

# Load the CIFAR10 data.
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Input image dimensions.
# We assume data format "channels_last".
img_rows = x_train.shape[1]
img_cols = x_train.shape[2]
channels = x_train.shape[3]
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, channels)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, channels)
input_shape = (img_rows, img_cols, channels)


# Normalize data.
def normalize(x):
    """
        argument
            - x: training data array
        return
            - x scaled to [0,1]
    """
    x = x.astype('float32') / 255
    # print(x.shape[0], ' samples')
    return x


x_train = normalize(x_train)
x_test = normalize(x_test)


def one_hot_encode(x, num):
    """
        argument
            - x: labels
            - num: number of classes
        return
            - one hot encoding matrix (number of labels, number of class)
    """
    x = keras.utils.to_categorical(x, num)
    return x


# Convert class vectors to binary class matrices.
y_train = one_hot_encode(y_train, num_classes)
y_test = one_hot_encode(y_test, num_classes)


# Start model definition.
inputs = Input(shape=input_shape)


def alex_net(input_data):
    # Conv1
    x = Conv2D(96, kernel_size=11, padding='same', strides=2, activation='relu')(input_data)
    x = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)

    # Conv2
    x = Conv2D(256, kernel_size=5, padding='same', strides=1, activation='relu')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)

    # Conv3
    x = Conv2D(384, kernel_size=3, padding='same', strides=1, activation='relu')(x)
    x = Conv2D(384, kernel_size=3, padding='same', strides=1, activation='relu')(x)
    x = Conv2D(256, kernel_size=3, padding='same', strides=1, activation='relu')(x)
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


def alex_residual(input_data):
    num_filters = 64
    num_blocks = 4
    num_sub_blocks = 2
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


if model_name == 'alex_residual':
    outputs = alex_residual(inputs)
elif model_name == 'alex_net':
    outputs = alex_net(inputs)
else:
    pass

model = Model(inputs=inputs, outputs=outputs)
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.001),
              metrics=['accuracy'])
model.summary()

# Prepare model saving directory.
save_dir = os.path.join(os.getcwd(), save_path)

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
file_path = os.path.join(save_dir, model_name + '.h5')

# Prepare callbacks for model saving and for learning rate decaying.
checkpoint = ModelCheckpoint(filepath=file_path,
                             verbose=1,
                             save_best_only=True)
lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)

# Add TensorBoard callbacks
tbCallBack = keras.callbacks.TensorBoard(log_dir='./logs_' + model_name, histogram_freq=0, write_graph=True, write_grads=True,
                                         write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)

callbacks = [checkpoint, lr_reducer, tbCallBack]


if random_erasing:
    data_generator = ImageDataGenerator(
        # rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False,  # randomly flip images
        preprocessing_function=get_random_eraser(v_l=0, v_h=1, pixel_level=True)
    )
else:
    data_generator = ImageDataGenerator(
        # rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False,  # randomly flip images
        # preprocessing_function=get_random_eraser(v_l=0, v_h=1, pixel_level=pixel_level)
    )

# Compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied).
data_generator.fit(x_train)

# Fit the model on the batches generated by datagen.flow().
model.fit_generator(data_generator.flow(x_train, y_train, batch_size=batch_size),
                    steps_per_epoch=x_train.shape[0] // batch_size,
                    validation_data=(x_test, y_test),
                    epochs=epochs, verbose=1, workers=4,
                    callbacks=callbacks)

# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
