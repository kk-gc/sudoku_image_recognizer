import cv2
import numpy as np
from matplotlib import pyplot as plt


def thresholding(image: np.array, show_results: bool = False) -> dict[str, np.array]:
    """
    Function return dict of transformed images
    :param image: numpy array of an image to process
    :param show_results: show all transformations on the screen
    :return: dict = { threshold_transformation_name: image_as_numpy_array }
    """

    _image_grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _return = {
        # 'ORIGINAL': image,
        'GRAYSCALE': _image_grayscale,
        'BINARY': cv2.threshold(_image_grayscale, 127, 255, cv2.THRESH_BINARY)[1],
        'BINARY_INV': cv2.threshold(_image_grayscale, 127, 255, cv2.THRESH_BINARY_INV)[1],
        'TRUNC': cv2.threshold(_image_grayscale, 127, 255, cv2.THRESH_TRUNC)[1],
        'TOZERO': cv2.threshold(_image_grayscale, 127, 255, cv2.THRESH_TOZERO)[1],
        'TOZERO_INV': cv2.threshold(_image_grayscale, 127, 255, cv2.THRESH_TOZERO_INV)[1],
        'ADAPTIVE_MEAN': cv2.adaptiveThreshold(_image_grayscale, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                               cv2.THRESH_BINARY, 11, 2),
        'ADAPTIVE_GAUSSIAN': cv2.adaptiveThreshold(_image_grayscale, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                   cv2.THRESH_BINARY, 11, 2),
        'OTSU': cv2.threshold(_image_grayscale, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
        'OTSU_GAUSS': cv2.threshold(cv2.GaussianBlur(_image_grayscale, (5, 5), 0), 0, 255,
                                    cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    }

    # display all transformations on the screen
    if show_results:
        _index = 0
        for title_shown, image_shown in _return.items():
            plt.subplot(2, 6, _index + 1), plt.imshow(image_shown, 'gray', vmin=0, vmax=255)
            plt.title(title_shown)
            plt.xticks([]), plt.yticks([])
            _index += 1
        plt.show()

    return _return


if __name__ == '__main__':
    pass
