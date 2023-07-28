'''
    View of images QTableWidget.
'''
from PyQt5.QtWidgets import QTableWidget, QLabel, QListWidget, QListWidgetItem
from engine.AnnoterReid import Identity
from PyQt5 import QtCore
from PyQt5.QtGui import QIcon


class ViewIdentityGallery:

    @staticmethod
    def View(gallery: QListWidget,
             identity: Identity):
        ''' View images in table.'''
        # Get translations
        _translate = QtCore.QCoreApplication.translate

        # Gallery : View each row.
        gallery.setViewMode(QListWidget.IconMode)
        gallery.setIconSize(QtCore.QSize(200, 200))
        gallery.setResizeMode(QListWidget.Adjust)
        gallery.clear()

        # For each identity image : Add to gallery
        for image in identity.images:
            # Item : Create
            item = QListWidgetItem(QIcon(image.path), image.name)
            gallery.addItem(item)
