'''
    View of images QTableWidget.
'''
from PyQt5.QtWidgets import QTableWidget, QLabel, QListWidget, QListWidgetItem
from engine.AnnoterReid import Identity
from PyQt5 import QtCore
from PyQt5.QtGui import QIcon, QColor

from helpers.colors import RedYellowGreenInterpolate


class ViewIdentity:

    @staticmethod
    def View(preview: QLabel,
             desc: QLabel,
             identity: Identity,
             selectedImageIndex: int,
             ):
        ''' View images in table.'''
        # Get translations
        _translate = QtCore.QCoreApplication.translate

        # Check : Selected image index invalid
        if (selectedImageIndex is None):
            return

        # Check : Selected image index out of range
        if (selectedImageIndex < 0) or (selectedImageIndex >= len(identity.images)):
            return

        # Identity : Get Image
        image = identity.images[selectedImageIndex]

        # Preview : QLabel : Load QIcon from image.path and set as pixmap.
        preview.setPixmap(QIcon(image.path).pixmap(200, 200))

        # Preview : QLabel : Desc
        desc.setText(f'Identity {identity.number} / {image.name}')
