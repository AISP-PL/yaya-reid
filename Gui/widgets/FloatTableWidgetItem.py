'''
Created on 14 gru 2022

@author: spasz
'''
from PyQt5 import QtWidgets, QtCore


class FloatTableWidgetItem(QtWidgets.QTableWidgetItem):
    # Float value
    value: float = 0

    def __init__(self,
                 value: float = 0,
                 decimals: int = 2
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

    def __lt__(self, other: QtWidgets.QTableWidgetItem):
        ''' Operation < for sorting.'''
        value = other.data(QtCore.Qt.UserRole)
        return (value is not None) and (self.value < value)
