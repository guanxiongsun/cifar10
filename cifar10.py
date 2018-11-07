from __future__ import print_function
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import Model
import numpy as np
import os
import data_generater
import models
import load_data as loader

model_names = {0: 'alex_net', 1: 'alex_vgg', 2: 'alex_residual',
               3: 'alex_wrn', 4: 'alex_inception', 5: 'alex_dense'}

# Select Models
model_name = model_names[3]

# Use RE or not
random_erasing = False

# Training params.
save_path = 'checkpoints'
batch_size = 256
epochs = 100
num_classes = 10

# Get cifar10 dataset
input_shape, x_train, x_test, y_train, y_test = loader.get_cifar10()

# Norm
x_train = loader.normalize(x_train)
x_test = loader.normalize(x_test)

# Convert vectors to one hot codes
y_train = loader.one_hot_encode(y_train, 10)
y_test = loader.one_hot_encode(y_test, 10)

# Start model definition.
inputs = keras.layers.Input(shape=input_shape)

if model_name == 'alex_residual':
    outputs = models.AlexResidual(inputs).get_output()
elif model_name == 'alex_net':
    outputs = models.AlexNet(inputs).get_output()
elif model_name == 'alex_vgg':
    outputs = models.AlexVGG(inputs).get_output()
elif model_name == 'alex_wrn':
    outputs = models.AlexWRN(inputs).get_output()
elif model_name == 'alex_inception':
    outputs = models.AlexInception(inputs).get_output()
elif model_name == 'alex_dense':
    outputs = models.AlexDense(inputs).get_output()
else:
    outputs = None

model = Model(inputs=inputs, outputs=outputs)
learn_rates = {'alex_net': 1e-3, 'alex_residual': 1e-3, 'alex_wrn': 1e-3,
               'alex_inception': 1e-3, 'alex_dense': 1e-3, 'alex_vgg': 1e-3}
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=learn_rates[model_name]), metrics=['accuracy'])
model.summary()

# Prepare model saving directory.
save_dir = os.path.join(os.getcwd(), save_path)
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
file_path = os.path.join(save_dir, model_name + '.h5')

# Prepare callbacks for model saving and for learning rate decaying.
checkpoint = ModelCheckpoint(filepath=file_path, verbose=1, save_best_only=True)
lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=10, min_lr=1e-6)

# Add TensorBoard callbacks
if random_erasing:
    model_name = model_name + '_re'
tbCallBack = keras.callbacks.TensorBoard(log_dir='./logs_' + model_name, write_grads=True, write_images=True)

callbacks = [checkpoint, lr_reducer, tbCallBack]

# Get data generator
data_gen = data_generater.get(is_random_erasing=random_erasing)
data_gen.fit(x_train)
model.fit_generator(data_gen.flow(x_train, y_train, batch_size=batch_size),
                    steps_per_epoch=x_train.shape[0] // batch_size,
                    validation_data=(x_test, y_test),
                    epochs=epochs, verbose=1, workers=4,
                    callbacks=callbacks)

# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
