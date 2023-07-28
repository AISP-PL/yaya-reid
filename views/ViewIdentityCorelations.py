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
    def View(corelations: QListWidget,
             identity1: Identity,
             imageIndex1: int,
             identity2: Identity,
             ):
        ''' View images in table.'''
        # Get translations
        _translate = QtCore.QCoreApplication.translate

        # Check : Selected image index invalid
        if (imageIndex1 is None):
            return

        # Check : Selected image index out of range
        if (imageIndex1 < 0) or (imageIndex1 >= len(identity1.images)):
            return

        # Identity : Get base Image
        image = identity1.images[imageIndex1]

        # Similarities : Get similarities between image and all images in identity2
        similarites = identity2.ImageSimilarities(image)

        # ImageNumber, Image and similarites : Group together and sort by similarites (descending)
        imageNumbers = list(range(len(identity2.images)))
        imagesAndSimilarites = list(
            zip(imageNumbers, identity2.images, similarites))
        imagesAndSimilarites.sort(key=lambda x: x[2], reverse=True)

        # Corelations Gallery : View each row.
        corelations.setViewMode(QListWidget.IconMode)
        corelations.setIconSize(QtCore.QSize(100, 100))
        corelations.setResizeMode(QListWidget.Adjust)
        corelations.clear()

        # Images and similarities : For each element
        for imageIndex, image, similarity in imagesAndSimilarites:
            # Item : Create
            item = QListWidgetItem(QIcon(image.path), f'{similarity:2.2f}')
            # Background color : Get RGB from similarity
            red, green, blue = RedYellowGreenInterpolate(similarity)
            item.setBackground(QColor(red, green, blue))
            item.setToolTip(str(imageIndex))

            corelations.addItem(item)
