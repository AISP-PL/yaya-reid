'''
    View of images QTableWidget.
'''
from PyQt5.QtWidgets import QTableWidget, QLabel, QListWidget, QListWidgetItem
from engine.AnnoterReid import Identity
from PyQt5 import QtCore
from PyQt5.QtGui import QIcon, QColor

from helpers.colors import RedYellowGreenInterpolate


class ViewIdentityCorelations:

    @staticmethod
    def View(preview: QLabel,
             desc: QLabel,
             corelations: QListWidget,
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

        # Identitiy : Get Image corelations
        similarites = identity.ImageSimilarities(image)
        # Image and similarites : Group together and sort by similarites (descending)
        imagesAndSimilarites = list(zip(identity.images, similarites))
        imagesAndSimilarites.sort(key=lambda x: x[1], reverse=True)

        # Corelations Gallery : View each row.
        corelations.setViewMode(QListWidget.IconMode)
        corelations.setIconSize(QtCore.QSize(100, 100))
        corelations.setResizeMode(QListWidget.Adjust)
        corelations.clear()

        # Images and similarities : For each element
        for image, similarity in imagesAndSimilarites:
            # Item : Create
            item = QListWidgetItem(QIcon(image.path), f'{similarity:2.2f}')
            # Background color : Get RGB from similarity
            red, green, blue = RedYellowGreenInterpolate(similarity)
            item.setBackground(QColor(red, green, blue))

            corelations.addItem(item)
