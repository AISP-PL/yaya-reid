'''
Created on 14 gru 2022

@author: spasz
'''
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtGui import QIcon


class IdentityTableWidgetItem(QtWidgets.QTableWidgetItem):
    # value
    value: int = 0

    def __init__(self,
                 qicon: QIcon,
                 number: int = 0,
                 ):
        ''' Constructor.'''
        super().__init__(qicon, str(number))
        # Set variables
        self.value = number

        # Set item data
        self.setData(QtCore.Qt.UserRole, number)
        self.setText(f'{self.value}')

    def __lt__(self, other: QtWidgets.QTableWidgetItem):
        ''' Operation < for sorting.'''
        value = other.data(QtCore.Qt.UserRole)
        return (value is not None) and (self.value < value)
