# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'MainWindow.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


import tango_rc
from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName('MainWindow')
        MainWindow.resize(1591, 834)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.centralwidget.sizePolicy().hasHeightForWidth())
        self.centralwidget.setSizePolicy(sizePolicy)
        self.centralwidget.setObjectName('centralwidget')
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName('gridLayout')
        self.splitter_4 = QtWidgets.QSplitter(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.splitter_4.sizePolicy().hasHeightForWidth())
        self.splitter_4.setSizePolicy(sizePolicy)
        self.splitter_4.setOrientation(QtCore.Qt.Horizontal)
        self.splitter_4.setObjectName('splitter_4')
        self.verticalLayoutWidget = QtWidgets.QWidget(self.splitter_4)
        self.verticalLayoutWidget.setObjectName('verticalLayoutWidget')
        self.verticalLayoutLeft = QtWidgets.QVBoxLayout(
            self.verticalLayoutWidget)
        self.verticalLayoutLeft.setSizeConstraint(
            QtWidgets.QLayout.SetMaximumSize)
        self.verticalLayoutLeft.setContentsMargins(0, 0, 0, 0)
        self.verticalLayoutLeft.setObjectName('verticalLayoutLeft')
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName('horizontalLayout')
        self.label_4 = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label_4.setObjectName('label_4')
        self.horizontalLayout.addWidget(self.label_4)
        self.imageScalingComboBox = QtWidgets.QComboBox(
            self.verticalLayoutWidget)
        self.imageScalingComboBox.setObjectName('imageScalingComboBox')
        self.imageScalingComboBox.addItem('')
        self.imageScalingComboBox.addItem('')
        self.imageScalingComboBox.addItem('')
        self.horizontalLayout.addWidget(self.imageScalingComboBox)
        self.verticalLayoutLeft.addLayout(self.horizontalLayout)
        self.gallery = QtWidgets.QListWidget(self.verticalLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.gallery.sizePolicy().hasHeightForWidth())
        self.gallery.setSizePolicy(sizePolicy)
        self.gallery.setObjectName('gallery')
        self.verticalLayoutLeft.addWidget(self.gallery)
        self.label_3 = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label_3.setObjectName('label_3')
        self.verticalLayoutLeft.addWidget(self.label_3)
        self.horizontalLayout_9 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_9.setSizeConstraint(
            QtWidgets.QLayout.SetMinimumSize)
        self.horizontalLayout_9.setObjectName('horizontalLayout_9')
        self.imagePreview = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.imagePreview.setMaximumSize(QtCore.QSize(16777215, 250))
        self.imagePreview.setObjectName('imagePreview')
        self.horizontalLayout_9.addWidget(self.imagePreview)
        self.imageCorelations = QtWidgets.QListWidget(
            self.verticalLayoutWidget)
        self.imageCorelations.setMaximumSize(QtCore.QSize(16777215, 250))
        self.imageCorelations.setObjectName('imageCorelations')
        self.horizontalLayout_9.addWidget(self.imageCorelations)
        self.verticalLayoutLeft.addLayout(self.horizontalLayout_9)
        self.layoutWidget = QtWidgets.QWidget(self.splitter_4)
        self.layoutWidget.setObjectName('layoutWidget')
        self.verticalLayoutRight = QtWidgets.QVBoxLayout(self.layoutWidget)
        self.verticalLayoutRight.setSizeConstraint(
            QtWidgets.QLayout.SetMaximumSize)
        self.verticalLayoutRight.setContentsMargins(0, 0, 0, 0)
        self.verticalLayoutRight.setObjectName('verticalLayoutRight')
        self.fileLabel = QtWidgets.QLabel(self.layoutWidget)
        self.fileLabel.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.fileLabel.sizePolicy().hasHeightForWidth())
        self.fileLabel.setSizePolicy(sizePolicy)
        self.fileLabel.setMinimumSize(QtCore.QSize(300, 32))
        self.fileLabel.setObjectName('fileLabel')
        self.verticalLayoutRight.addWidget(self.fileLabel)
        self.sliderLayout = QtWidgets.QHBoxLayout()
        self.sliderLayout.setObjectName('sliderLayout')
        self.fileNumberSliderLabel = QtWidgets.QLabel(self.layoutWidget)
        self.fileNumberSliderLabel.setObjectName('fileNumberSliderLabel')
        self.sliderLayout.addWidget(self.fileNumberSliderLabel)
        self.prevFileButton = QtWidgets.QToolButton(self.layoutWidget)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(
            ':/icons/16x16/media-skip-backward.png'), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.prevFileButton.setIcon(icon)
        self.prevFileButton.setObjectName('prevFileButton')
        self.sliderLayout.addWidget(self.prevFileButton)
        self.nextFileButton = QtWidgets.QToolButton(self.layoutWidget)
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(
            ':/icons/16x16/media-skip-forward.png'), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.nextFileButton.setIcon(icon1)
        self.nextFileButton.setObjectName('nextFileButton')
        self.sliderLayout.addWidget(self.nextFileButton)
        spacerItem = QtWidgets.QSpacerItem(
            40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.sliderLayout.addItem(spacerItem)
        self.verticalLayoutRight.addLayout(self.sliderLayout)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName('horizontalLayout_2')
        self.label_2 = QtWidgets.QLabel(self.layoutWidget)
        self.label_2.setObjectName('label_2')
        self.horizontalLayout_2.addWidget(self.label_2)
        self.SaveFileAnnotationsButton = QtWidgets.QPushButton(
            self.layoutWidget)
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap(
            ':/icons/16x16/document-save-as.png'), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.SaveFileAnnotationsButton.setIcon(icon2)
        self.SaveFileAnnotationsButton.setObjectName(
            'SaveFileAnnotationsButton')
        self.horizontalLayout_2.addWidget(self.SaveFileAnnotationsButton)
        self.DeleteImageAnnotationsButton = QtWidgets.QPushButton(
            self.layoutWidget)
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap(
            ':/icons/16x16/process-stop.png'), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.DeleteImageAnnotationsButton.setIcon(icon3)
        self.DeleteImageAnnotationsButton.setObjectName(
            'DeleteImageAnnotationsButton')
        self.horizontalLayout_2.addWidget(self.DeleteImageAnnotationsButton)
        spacerItem1 = QtWidgets.QSpacerItem(
            40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem1)
        self.verticalLayoutRight.addLayout(self.horizontalLayout_2)
        self.fileSummaryLabel = QtWidgets.QLabel(self.layoutWidget)
        self.fileSummaryLabel.setTextFormat(QtCore.Qt.MarkdownText)
        self.fileSummaryLabel.setObjectName('fileSummaryLabel')
        self.verticalLayoutRight.addWidget(self.fileSummaryLabel)
        self.fileSelectorTableWidget = QtWidgets.QTableWidget(
            self.layoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.fileSelectorTableWidget.sizePolicy().hasHeightForWidth())
        self.fileSelectorTableWidget.setSizePolicy(sizePolicy)
        self.fileSelectorTableWidget.setMinimumSize(QtCore.QSize(0, 300))
        self.fileSelectorTableWidget.setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarAlwaysOn)
        self.fileSelectorTableWidget.setSizeAdjustPolicy(
            QtWidgets.QAbstractScrollArea.AdjustToContentsOnFirstShow)
        self.fileSelectorTableWidget.setSelectionMode(
            QtWidgets.QAbstractItemView.SingleSelection)
        self.fileSelectorTableWidget.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectRows)
        self.fileSelectorTableWidget.setObjectName('fileSelectorTableWidget')
        self.fileSelectorTableWidget.setColumnCount(0)
        self.fileSelectorTableWidget.setRowCount(0)
        self.verticalLayoutRight.addWidget(self.fileSelectorTableWidget)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName('horizontalLayout_3')
        self.label = QtWidgets.QLabel(self.layoutWidget)
        self.label.setMinimumSize(QtCore.QSize(90, 0))
        self.label.setObjectName('label')
        self.horizontalLayout_3.addWidget(self.label)
        self.detectAnnotationsButton = QtWidgets.QPushButton(self.layoutWidget)
        icon4 = QtGui.QIcon()
        icon4.addPixmap(QtGui.QPixmap(
            ':/icons/16x16/camera-photo.png'), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.detectAnnotationsButton.setIcon(icon4)
        self.detectAnnotationsButton.setObjectName('detectAnnotationsButton')
        self.horizontalLayout_3.addWidget(self.detectAnnotationsButton)
        spacerItem2 = QtWidgets.QSpacerItem(
            40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem2)
        self.verticalLayoutRight.addLayout(self.horizontalLayout_3)
        self.gridLayout.addWidget(self.splitter_4, 1, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1591, 22))
        self.menubar.setObjectName('menubar')
        self.menuMenu = QtWidgets.QMenu(self.menubar)
        self.menuMenu.setObjectName('menuMenu')
        self.menuPomoc = QtWidgets.QMenu(self.menubar)
        self.menuPomoc.setObjectName('menuPomoc')
        self.menuEdit = QtWidgets.QMenu(self.menubar)
        self.menuEdit.setObjectName('menuEdit')
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName('statusbar')
        MainWindow.setStatusBar(self.statusbar)
        self.actionOtw_rz = QtWidgets.QAction(MainWindow)
        self.actionOtw_rz.setObjectName('actionOtw_rz')
        self.actionZamknij = QtWidgets.QAction(MainWindow)
        self.actionZamknij.setObjectName('actionZamknij')
        self.actionO_programie = QtWidgets.QAction(MainWindow)
        self.actionO_programie.setObjectName('actionO_programie')
        self.actionOtworzLokacje = QtWidgets.QAction(MainWindow)
        self.actionOtworzLokacje.setObjectName('actionOtworzLokacje')
        self.actionZamknijProgram = QtWidgets.QAction(MainWindow)
        self.actionZamknijProgram.setObjectName('actionZamknijProgram')
        self.actionZapisz = QtWidgets.QAction(MainWindow)
        self.actionZapisz.setObjectName('actionZapisz')
        self.actionNextLocation = QtWidgets.QAction(MainWindow)
        self.actionNextLocation.setObjectName('actionNextLocation')
        self.actionPrevLocation = QtWidgets.QAction(MainWindow)
        self.actionPrevLocation.setObjectName('actionPrevLocation')
        self.actionMountRO = QtWidgets.QAction(MainWindow)
        self.actionMountRO.setObjectName('actionMountRO')
        self.actionNextConfiguration = QtWidgets.QAction(MainWindow)
        self.actionNextConfiguration.setObjectName('actionNextConfiguration')
        self.actionPrevConfiguration = QtWidgets.QAction(MainWindow)
        self.actionPrevConfiguration.setObjectName('actionPrevConfiguration')
        self.actionSave_screenshoot = QtWidgets.QAction(MainWindow)
        self.actionSave_screenshoot.setObjectName('actionSave_screenshoot')
        self.actionSave_copy = QtWidgets.QAction(MainWindow)
        self.actionSave_copy.setObjectName('actionSave_copy')
        self.menuMenu.addAction(self.actionOtworzLokacje)
        self.menuMenu.addAction(self.actionZapisz)
        self.menuMenu.addAction(self.actionZamknijProgram)
        self.menuEdit.addAction(self.actionSave_screenshoot)
        self.menuEdit.addAction(self.actionSave_copy)
        self.menubar.addAction(self.menuMenu.menuAction())
        self.menubar.addAction(self.menuEdit.menuAction())
        self.menubar.addAction(self.menuPomoc.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate(
            'MainWindow', 'YAYA - YOLO annoter'))
        self.label_4.setText(_translate('MainWindow', 'Image scaling:'))
        self.imageScalingComboBox.setItemText(
            0, _translate('MainWindow', 'Resize'))
        self.imageScalingComboBox.setItemText(
            1, _translate('MainWindow', 'ResizeAspectRatio'))
        self.imageScalingComboBox.setItemText(
            2, _translate('MainWindow', 'OriginalSize'))
        self.label_3.setText(_translate(
            'MainWindow', 'Images corelation inside identity :'))
        self.imagePreview.setText(_translate('MainWindow', 'TextLabel'))
        self.fileLabel.setText(_translate(
            'MainWindow', 'Filename (number/all)'))
        self.fileNumberSliderLabel.setText(
            _translate('MainWindow', 'Slider label'))
        self.prevFileButton.setText(_translate('MainWindow', '...'))
        self.prevFileButton.setShortcut(_translate('MainWindow', ','))
        self.nextFileButton.setText(_translate('MainWindow', '...'))
        self.nextFileButton.setShortcut(_translate('MainWindow', '.'))
        self.label_2.setText(_translate('MainWindow', 'Image'))
        self.SaveFileAnnotationsButton.setText(
            _translate('MainWindow', '(S)ave'))
        self.SaveFileAnnotationsButton.setShortcut(
            _translate('MainWindow', 'S'))
        self.DeleteImageAnnotationsButton.setText(
            _translate('MainWindow', '(X)Delete'))
        self.DeleteImageAnnotationsButton.setShortcut(
            _translate('MainWindow', 'X'))
        self.fileSummaryLabel.setText(_translate('MainWindow', 'TextLabel'))
        self.fileSelectorTableWidget.setSortingEnabled(True)
        self.label.setText(_translate('MainWindow', 'Annotations'))
        self.detectAnnotationsButton.setText(
            _translate('MainWindow', '(D)etect '))
        self.detectAnnotationsButton.setShortcut(_translate('MainWindow', 'D'))
        self.menuMenu.setTitle(_translate('MainWindow', 'File'))
        self.menuPomoc.setTitle(_translate('MainWindow', 'Help'))
        self.menuEdit.setTitle(_translate('MainWindow', 'Edit'))
        self.actionOtw_rz.setText(_translate('MainWindow', 'Otwórz'))
        self.actionZamknij.setText(_translate('MainWindow', 'Zamknij'))
        self.actionO_programie.setText(_translate('MainWindow', 'O programie'))
        self.actionOtworzLokacje.setText(
            _translate('MainWindow', 'Open directory'))
        self.actionOtworzLokacje.setShortcut(
            _translate('MainWindow', 'Ctrl+O'))
        self.actionZamknijProgram.setText(_translate('MainWindow', 'Exit'))
        self.actionZamknijProgram.setShortcut(
            _translate('MainWindow', 'Ctrl+X'))
        self.actionZapisz.setText(_translate('MainWindow', 'Save'))
        self.actionZapisz.setShortcut(_translate('MainWindow', 'Ctrl+S', 'S'))
        self.actionNextLocation.setText(
            _translate('MainWindow', 'Następna lokacja'))
        self.actionNextLocation.setShortcut(_translate('MainWindow', 'Ctrl+N'))
        self.actionPrevLocation.setText(
            _translate('MainWindow', 'Poprzednia lokacja'))
        self.actionPrevLocation.setShortcut(_translate('MainWindow', 'Ctrl+B'))
        self.actionMountRO.setText(_translate(
            'MainWindow', 'Przemontuj lokacje'))
        self.actionNextConfiguration.setText(
            _translate('MainWindow', 'Następna konfiguracja'))
        self.actionNextConfiguration.setShortcut(
            _translate('MainWindow', 'Ctrl+.'))
        self.actionPrevConfiguration.setText(
            _translate('MainWindow', 'Poprzednia konfiguracja'))
        self.actionPrevConfiguration.setShortcut(
            _translate('MainWindow', 'Ctrl+,'))
        self.actionSave_screenshoot.setText(
            _translate('MainWindow', 'Save screenshoot'))
        self.actionSave_screenshoot.setShortcut(
            _translate('MainWindow', 'Shift+S'))
        self.actionSave_copy.setText(_translate('MainWindow', 'Save copy'))
        self.actionSave_copy.setShortcut(_translate('MainWindow', 'Shift+C'))
