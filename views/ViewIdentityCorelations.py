'''
    View of images QTableWidget.
'''
from PyQt5.QtWidgets import QTableWidget, QLabel, QListWidget, QListWidgetItem
from engine.AnnoterReid import Identity
from PyQt5 import QtCore
from PyQt5.QtGui import QIcon


class ViewIdentityCorelations:

    @staticmethod
    def View(preview: QLabel,
             corelations: QListWidget,
             identity: Identity):
        ''' View images in table.'''
        # Get translations
        _translate = QtCore.QCoreApplication.translate

        # Identity : Get Image
        image = identity.image

        # Preview : QLabel : Load QIcon from image.path and set as pixmap.
        preview.setPixmap(QIcon(image.path).pixmap(200, 200))

        # Identitiy : Get Image corelations
        similarites = identity.ImageSimilarities(image)

        # Corelations Gallery : View each row.
        corelations.setViewMode(QListWidget.IconMode)
        corelations.setIconSize(QtCore.QSize(200, 200))
        corelations.setResizeMode(QListWidget.Adjust)
        corelations.clear()

        # For each identity image : Add to corelations
        for index, image in enumerate(identity.images):
            # Item : Create
            item = QListWidgetItem(QIcon(image.path), str(similarites[index]))
            corelations.addItem(item)
