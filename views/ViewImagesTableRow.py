'''
    View of images QTableWidget.
'''
import logging
from PyQt5.QtWidgets import QTableWidget, QTableWidgetItem, QAbstractItemView
from PyQt5 import QtCore
from PyQt5.QtCore import Qt
from Gui.widgets.FloatTableWidgetItem import FloatTableWidgetItem
from Gui.widgets.PercentTableWidgetItem import PercentTableWidgetItem
from Gui.widgets.RectTableWidgetItem import RectTableWidgetItem
from engine.AnnoterReid import Identity


class ViewImagesTableRow:

    @staticmethod
    def View(table: QTableWidget, rowIndex: int, identity: Identity):
        ''' View images in table.'''
        # Get translations
        _translate = QtCore.QCoreApplication.translate

        # Start from column zero
        colIndex = 0

        # Identitiy
        item = QTableWidgetItem(str(identity.number))
        item.setToolTip(str(identity.number))
        table.setItem(rowIndex, colIndex, item)
        colIndex += 1

        # Identitiy images count
        item = QTableWidgetItem(str(identity.images_count))
        item.setToolTip(str(identity.number))
        table.setItem(rowIndex, colIndex, item)
        colIndex += 1

        # Hue column
        item = FloatTableWidgetItem(identity.hue)
        item.setToolTip(str(identity.number))
        table.setItem(rowIndex, colIndex, item)
        colIndex += 1

        # Saturation column
        item = FloatTableWidgetItem(identity.saturation)
        item.setToolTip(str(identity.number))
        table.setItem(rowIndex, colIndex, item)
        colIndex += 1

        # Brightness column
        item = FloatTableWidgetItem(identity.brightness)
        item.setToolTip(str(identity.number))
        table.setItem(rowIndex, colIndex, item)
        colIndex += 1

        # Image hash column
        item = FloatTableWidgetItem(identity.imhash,
                                    decimals=7)
        item.setToolTip(str(identity.number))
        table.setItem(rowIndex, colIndex, item)
        colIndex += 1

        # Image features
        item = QTableWidgetItem(str(identity.features))
        item.setToolTip(str(identity.number))
        table.setItem(rowIndex, colIndex, item)
        colIndex += 1
