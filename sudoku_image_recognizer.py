import cv2
import numpy as np
import pytesseract
from typing import Optional, Any, Dict, Sequence

from cv2 import UMat

import thresholding


class SudokuImageRecognizer:

    pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

    def __init__(self, base_image, desired_output=None):
        self.base_image = cv2.imread(base_image)
        self.desired_output = desired_output
        self.images_thresholding = thresholding.thresholding(self.base_image, show_results=False)
        self.contours = self.get_contours()
        self.cells = self.get_cells()
        self.digits = self.get_digits()
        self.digits_match = self.get_digits_match()
        self.results = self.get_results()

    def get_contours(self) -> dict[Any, Sequence[UMat] | UMat]:
        """
        Method return contours extracted from
        all image transformations (thresholding)
        :return: dict = { transformation_name: contours }
        """
        _contours = {}
        for threshold_name, threshold_image in self.images_thresholding.items():
            _contours[threshold_name] = cv2.findContours(threshold_image,
                                                         cv2.RETR_EXTERNAL,
                                                         cv2.CHAIN_APPROX_SIMPLE)[0]
        return _contours

    def get_cells(self) -> dict[str, list]:
        """
        Method is trying to find 9 x 9 = 81 occurrences of contours
        with the contour area within some pre-defined parameters:
        maximum_box_area - cell can't be bigger than 1/9 to an image to fit all 9
        minimum_box_area - arbitrary but to filter big number of 'too small' contours
        :return: dict = { transformation_name: 81 cells coordinates or [] }
        """
        _all_cells = {}
        _image_height, _image_width, _ = self.base_image.shape

        for contours_name, contours_data in self.contours.items():

            # maximum_box_area is shorter edge, divided by 9 boxes squared
            maximum_box_area = (min(_image_height, _image_width) // 9) ** 2

            _minimum_allowed_ratio = .5
            # minimum_box_area is shorter edge times _minimum_allowed_ratio,
            # divided by 9 boxes squared
            minimum_box_area = (min(_image_height * _minimum_allowed_ratio,
                                    _image_width * _minimum_allowed_ratio) // 9) ** 2

            _cells = []
            for i in range(len(contours_data)):
                if maximum_box_area > cv2.contourArea(contours_data[i]) > minimum_box_area:
                    _xs, _ys = np.split(contours_data[i], 2, axis=2)
                    _cells.append({'_i': i,
                                   'x0': _xs.min(),
                                   'x1': _xs.max(),
                                   'y0': _ys.min(),
                                   'y1': _ys.max(),
                                   })

            # this double sorting is to have rows in line to the pixel
            # as they like to be off for one or two, it might (will)
            # cause a problem with angled images if this is ever implemented
            if len(_cells) == 81:
                _cells = sorted(_cells, key=lambda x: (x['y0'], x['x0']))
                for i in range(0, 81, 9):
                    for j in range(i, i + 9):
                        _cells[j]['y0'] = _cells[i]['y0']
                # sorted(_cells, key=lambda x: (x['y0'], x['x0']))
                _all_cells[contours_name] = sorted(_cells, key=lambda x: (x['y0'], x['x0']))

        return _all_cells

    def get_digits(self) -> dict[str, list]:
        """
        Method is returning extracted digits for all
        image transformations from `self.cells`
        :return: dict = { transformation_name: extracted_digits }
        """
        _all_digits = {}
        for cells_name, cells_data in self.cells.items():
            digits = []
            for cell in cells_data:
                digit = self._get_digit(cells_name, cell)
                digits.append(digit)
            _all_digits[cells_name] = digits
        return _all_digits

    def _get_digit(self, image_name: str, cell_coordinates: dict) -> int:
        """
        Method extracting digit from the given cell
        :param image_name:
        :param cell_coordinates:
        :return: extracted digit or 0 if unsuccessful [int]
        """
        y0 = cell_coordinates['y0']
        y1 = cell_coordinates['y1']
        x0 = cell_coordinates['x0']
        x1 = cell_coordinates['x1']

        # shrink cropped by 10% to avoid call frame interference
        shrink_ratio = .1
        y0 = int(y0 + shrink_ratio * (y1 - y0))
        y1 = int(y1 - shrink_ratio * (y1 - y0))
        x0 = int(x0 + shrink_ratio * (x1 - x0))
        x1 = int(x1 - shrink_ratio * (x1 - x0))

        cropped = self.images_thresholding[image_name][y0:y1, x0:x1]

        output_string = pytesseract.image_to_string(cropped, config=r'--oem 3 --psm 6 digits')

        try:
            _return = int(output_string)
            return _return
        except ValueError:
            return 0

    def get_digits_match(self) -> Optional[dict[str, bool]]:
        """
        If `self.desired_output` string is provided, method check readings from
        which transformation are matching 100%
        :return: dict = { transformation_name: bool }
        """
        if self.desired_output and isinstance(self.desired_output, str):
            _desired_output = [int(x) for x in self.desired_output]
            _digits_match = {}
            for digits_name, digits_data in self.digits.items():
                if digits_data == _desired_output:
                    _digits_match[digits_name] = True
                else:
                    _digits_match[digits_name] = False
            return _digits_match
        return None

    def get_results(self) -> dict[int, set]:
        """
        Method is comparing all results from `self.digits` and produce
        dict = { cell_no: extracted_digit(s) }:
        one extracted_digit(s) -> same result from all transformations of the image, safe to use
        zero and digit -> some recognition fail, still could use digit, kinda safe
        two or more digits -> unlikely, but should be analyzed as two (or more) separate cases
        :return: dict = { cell_no: extracted_digit(s) }
        """
        _result = {}
        if self.digits:
            for i in range(81):
                _result[i] = {value[i] for value in self.digits.values()}
        return _result


if __name__ == '__main__':

    desired_output = '530070000600195000098000060800060003400803001700020006060000280000419005000080079'
    sir = SudokuImageRecognizer('sudoku.png', desired_output)

