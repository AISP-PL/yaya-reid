'''
Created on 14 gru 2022

@author: spasz
'''
from PyQt5 import QtWidgets, QtCore

from helpers.colors import RedYellowGreenInterpolate
from PyQt5.QtGui import QColor


class FloatScaleTableWidgetItem(QtWidgets.QTableWidgetItem):
    # Float value
    value: float = 0
    # Max value
    maxVal: float = 100
    # Min value
    minVal: float = 0

    def __init__(self,
                 value: float = 0,
                 decimals: int = 2,
                 maxVal: float = 100,
                 minVal: float = 0,
                 ):
        ''' Constructor.'''
        super().__init__()
        # Set variables
        self.value = value

        # Set item data
        self.setData(QtCore.Qt.UserRole, value)
        self.setText(f'{self.value:2.{decimals}f}')
        # Set alignment center vertical and horizontal
        self.setTextAlignment(QtCore.Qt.AlignCenter)

        # Value : Scale to [0, 1]
        valueRGB = (self.value - minVal) / (maxVal - minVal)

        # Background color : Get RGB from similarity
        red, green, blue = RedYellowGreenInterpolate(valueRGB)
        self.setBackground(QColor(red, green, blue))

    def __lt__(self, other: QtWidgets.QTableWidgetItem):
        ''' Operation < for sorting.'''
        value = other.data(QtCore.Qt.UserRole)
        return (value is not None) and (self.value < value)
