'''
    View of images QTableWidget.
'''
from PyQt5.QtWidgets import QTableWidget, QTableWidgetItem, QAbstractItemView
from PyQt5.QtCore import Qt
from PyQt5 import QtCore

from views.ViewImagesTableRow import ViewImagesTableRow


class ViewImagesTable:

    @staticmethod
    def View(table: QTableWidget, identities: list):
        ''' View images in table.'''
        # Check : Invalid files list
        if (identities is None) or (len(identities) == 0):
            return

        # Get translations
        _translate = QtCore.QCoreApplication.translate

        # Update GUI data
        table.clear()
        labels = _translate('ViewImagesTable',
                            'Identity;Images;Hue;Saturation;Brightness;ImHash;' +
                            'Features').split(';')
        table.setColumnCount(len(labels))
        table.setHorizontalHeaderLabels(labels)
        table.setRowCount(len(identities))
        table.setColumnCount(len(labels))

        # View each row.
        for rowIndex, identity_number in enumerate(identities):
            identity = identities[identity_number]
            ViewImagesTableRow.View(table, rowIndex, identity)

        # GUI - Enable sorting again
        table.setIconSize(QtCore.QSize(64, 64))
        table.setSortingEnabled(True)
        table.resizeColumnsToContents()
        table.resizeRowsToContents()
