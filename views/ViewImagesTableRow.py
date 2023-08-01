'''
    View of images QTableWidget.
'''
import logging
from PyQt5.QtWidgets import QTableWidget, QTableWidgetItem, QAbstractItemView
from PyQt5 import QtCore
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from Gui.widgets.IdentityTableWidgetItem import IdentityTableWidgetItem
from Gui.widgets.IntTableWidgetItem import IntTableWidgetItem
from Gui.widgets.FloatTableWidgetItem import FloatTableWidgetItem
from Gui.widgets.PercentTableWidgetItem import PercentTableWidgetItem
from Gui.widgets.RectTableWidgetItem import RectTableWidgetItem
from engine.AnnoterReid import Identity


class ViewImagesTableRow:

    @staticmethod
    def View(table: QTableWidget,
             rowIndex: int,
             identity: Identity,
             similarity: float = 0,
             ):
        ''' View images in table.'''
        # Get translations
        _translate = QtCore.QCoreApplication.translate

        # Start from column zero
        colIndex = 0

        # Identitiy
        image = identity.image
        item = IdentityTableWidgetItem(QIcon(image.path),
                                       identity.number)
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

        # Image consistency
        item = FloatTableWidgetItem(identity.consistency)
        item.setToolTip(str(identity.number))
        table.setItem(rowIndex, colIndex, item)
        colIndex += 1

        # Identity similarity to selected identity (given as parameter)
        item = FloatTableWidgetItem(similarity)
        item.setToolTip(str(identity.number))
        table.setItem(rowIndex, colIndex, item)
        colIndex += 1

        # Image features
        item = IntTableWidgetItem(identity.features_binrepr)
        item.setToolTip(str(identity.number))
        table.setItem(rowIndex, colIndex, item)
        colIndex += 1
