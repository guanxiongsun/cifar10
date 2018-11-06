import sklearn
from sklearn.pipeline import Pipeline
from sklearn import svm
import pickle
import numpy as np
import matplotlib.pyplot as plt

def load_data(file):
    with open(file, 'rb') as f:
        data = pickle.load(f)
    return data


data = load_data("cifar-10-batches-py/data_batch_1")

images = data['data']
# Reshape to go from length 3072 vector to 32x32 rgb images
# order='F' deals with specifics of how the data is laid out
images = images.reshape((-1, 32, 32, 3), order='F')
labels = np.array(data['labels'])


def get_classes(classes=[0, 1, 2], per_class=100):
    # Array of indices i where labels[i] is in classes
    indices = np.concatenate([np.where(labels == c)[0][:per_class] for c in classes])
    return images[indices], labels[indices]

# For speed, let's consider only 2 classes, 100 images per class for now


classes = [0, 1, 2, 3]
X, Y = get_classes(classes, 100)

for c in classes:
    plt.imshow(images[labels == c][0], interpolation='nearest')
    plt.show()