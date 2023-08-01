'''
    QListWidget with delete functionality.
'''
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QListWidget, QListWidgetItem


class DeleteableListWidget(QListWidget):
    ''' List widget with delete functionality.'''

    # Singal emitted when an item is deleted.
    itemSelectedDelete = pyqtSignal(QListWidgetItem)

    def __init__(self, parent=None):
        super(DeleteableListWidget, self).__init__(parent)

    def keyPressEvent(self, event):
        ''' Overriden keyPressEvent.'''

        # Delete selected items.
        if event.key() == Qt.Key_Delete:
            for item in self.selectedItems():
                self.itemSelectedDelete.emit(item)
                self.takeItem(self.row(item))

        # Default behaviour
        else:
            super(DeleteableListWidget, self).keyPressEvent(event)
