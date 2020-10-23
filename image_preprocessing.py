import cv2
from keras import utils


def img_to_mnist(path):
    """
    Reads the image from the path argument and converts it to the normalized numpy array.
    :param path: String, path to the image.
    :return: Numpy array normalized in range 0-1
    """
    img_array = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img_array = cv2.bitwise_not(img_array)

    size = (28, 28)
    new_array = cv2.resize(img_array, size)

    img_normalized = utils.normalize(new_array, axis=1)
    return img_normalized
