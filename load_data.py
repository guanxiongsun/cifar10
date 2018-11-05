from tensorflow.keras.datasets import cifar10
from tensorflow import keras


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


def get_cifar10():
    """
        argument
            - x: labels
            - num: number of classes
        return
            - input_shapes: 32,32,3
            - x_train 50000, 32, 32, 3
            - x_test 10000, 32, 32, 3
            - y_train 50000, 10
            - y_test 10000, 10
    """
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    img_rows = x_train.shape[1]
    img_cols = x_train.shape[2]
    channels = x_train.shape[3]
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, channels)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, channels)
    input_shape = (img_rows, img_cols, channels)

    x_train = normalize(x_train)
    x_test = normalize(x_test)

    # Convert vectors to one hot codes
    y_train = one_hot_encode(y_train, 10)
    y_test = one_hot_encode(y_test, 10)

    return input_shape, x_train, x_test, y_train, y_test
