from tensorflow.keras.preprocessing.image import ImageDataGenerator
from utils.random_eraser import get_random_eraser


def get(is_random_erasing):
    """
        argument
            - is_random_erasing: Bool type, use RE or not
        return
            - data_generator
    """
    if is_random_erasing:
        data_generator = ImageDataGenerator(
            rotation_range=10,  # degrees, 0 to 180
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False,  # randomly flip images
            preprocessing_function=get_random_eraser(v_l=0, v_h=1, pixel_level=True)
        )
    else:
        data_generator = ImageDataGenerator(
            rotation_range=10,  # degrees, 0 to 180
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False,  # randomly flip images
        )

    return data_generator
