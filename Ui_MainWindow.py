# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'MainWindow.ui'
#
# Created by: PyQt5 UI code generator 5.13.0
#
# WARNING! All changes made in this file will be lost!


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
        self.horizontalLayout_9 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_9.setSizeConstraint(
            QtWidgets.QLayout.SetMaximumSize)
        self.horizontalLayout_9.setObjectName('horizontalLayout_9')
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName('verticalLayout')
        self.label_6 = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label_6.setObjectName('label_6')
        self.verticalLayout.addWidget(self.label_6)
        self.imagePreview = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.imagePreview.setMaximumSize(QtCore.QSize(16777215, 250))
        self.imagePreview.setObjectName('imagePreview')
        self.verticalLayout.addWidget(self.imagePreview)
        self.imageDesc = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.imageDesc.setObjectName('imageDesc')
        self.verticalLayout.addWidget(self.imageDesc)
        spacerItem = QtWidgets.QSpacerItem(
            20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem)
        self.horizontalLayout_9.addLayout(self.verticalLayout)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setSizeConstraint(
            QtWidgets.QLayout.SetMaximumSize)
        self.verticalLayout_2.setObjectName('verticalLayout_2')
        self.label_3 = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label_3.setObjectName('label_3')
        self.verticalLayout_2.addWidget(self.label_3)
        self.identityGallery = QtWidgets.QListWidget(self.verticalLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.identityGallery.sizePolicy().hasHeightForWidth())
        self.identityGallery.setSizePolicy(sizePolicy)
        self.identityGallery.setMinimumSize(QtCore.QSize(0, 400))
        self.identityGallery.setMaximumSize(QtCore.QSize(16777215, 600))
        self.identityGallery.setObjectName('identityGallery')
        self.verticalLayout_2.addWidget(self.identityGallery)
        spacerItem1 = QtWidgets.QSpacerItem(
            20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_2.addItem(spacerItem1)
        self.horizontalLayout_9.addLayout(self.verticalLayout_2)
        self.verticalLayoutLeft.addLayout(self.horizontalLayout_9)
        self.identityCompareLabel = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.identityCompareLabel.setObjectName('identityCompareLabel')
        self.verticalLayoutLeft.addWidget(self.identityCompareLabel)
        self.identityCompareGallery = QtWidgets.QListWidget(
            self.verticalLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.identityCompareGallery.sizePolicy().hasHeightForWidth())
        self.identityCompareGallery.setSizePolicy(sizePolicy)
        self.identityCompareGallery.setObjectName('identityCompareGallery')
        self.verticalLayoutLeft.addWidget(self.identityCompareGallery)
        self.layoutWidget = QtWidgets.QWidget(self.splitter_4)
        self.layoutWidget.setObjectName('layoutWidget')
        self.verticalLayoutRight = QtWidgets.QVBoxLayout(self.layoutWidget)
        self.verticalLayoutRight.setSizeConstraint(
            QtWidgets.QLayout.SetMaximumSize)
        self.verticalLayoutRight.setContentsMargins(0, 0, 0, 0)
        self.verticalLayoutRight.setObjectName('verticalLayoutRight')
        self.modelDetails = QtWidgets.QLabel(self.layoutWidget)
        self.modelDetails.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.modelDetails.sizePolicy().hasHeightForWidth())
        self.modelDetails.setSizePolicy(sizePolicy)
        self.modelDetails.setMinimumSize(QtCore.QSize(300, 32))
        self.modelDetails.setTextFormat(QtCore.Qt.MarkdownText)
        self.modelDetails.setObjectName('modelDetails')
        self.verticalLayoutRight.addWidget(self.modelDetails)
        self.modelResults = QtWidgets.QLabel(self.layoutWidget)
        self.modelResults.setTextFormat(QtCore.Qt.MarkdownText)
        self.modelResults.setObjectName('modelResults')
        self.verticalLayoutRight.addWidget(self.modelResults)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setSizeConstraint(
            QtWidgets.QLayout.SetMinimumSize)
        self.horizontalLayout_2.setObjectName('horizontalLayout_2')
        self.identitiesSelectedLabel = QtWidgets.QLabel(self.layoutWidget)
        self.identitiesSelectedLabel.setObjectName('identitiesSelectedLabel')
        self.horizontalLayout_2.addWidget(self.identitiesSelectedLabel)
        self.mergeIdentitiesButton = QtWidgets.QPushButton(self.layoutWidget)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(
            ':/icons/32x32/emblem-favorite.png'), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.mergeIdentitiesButton.setIcon(icon)
        self.mergeIdentitiesButton.setObjectName('mergeIdentitiesButton')
        self.horizontalLayout_2.addWidget(self.mergeIdentitiesButton)
        spacerItem2 = QtWidgets.QSpacerItem(
            40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem2)
        self.verticalLayoutRight.addLayout(self.horizontalLayout_2)
        self.Selectionmode = QtWidgets.QHBoxLayout()
        self.Selectionmode.setSizeConstraint(QtWidgets.QLayout.SetMinimumSize)
        self.Selectionmode.setObjectName('Selectionmode')
        self.selectionIdentity = QtWidgets.QRadioButton(self.layoutWidget)
        self.selectionIdentity.setObjectName('selectionIdentity')
        self.Selectionmode.addWidget(self.selectionIdentity)
        self.selectionIdentityCompared = QtWidgets.QRadioButton(
            self.layoutWidget)
        self.selectionIdentityCompared.setObjectName(
            'selectionIdentityCompared')
        self.Selectionmode.addWidget(self.selectionIdentityCompared)
        self.similarityMethod = QtWidgets.QComboBox(self.layoutWidget)
        self.similarityMethod.setObjectName('similarityMethod')
        self.Selectionmode.addWidget(self.similarityMethod)
        spacerItem3 = QtWidgets.QSpacerItem(
            40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.Selectionmode.addItem(spacerItem3)
        self.verticalLayoutRight.addLayout(self.Selectionmode)
        self.fileSelectorTableWidget = QtWidgets.QTableWidget(
            self.layoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.fileSelectorTableWidget.sizePolicy().hasHeightForWidth())
        self.fileSelectorTableWidget.setSizePolicy(sizePolicy)
        self.fileSelectorTableWidget.setMinimumSize(QtCore.QSize(0, 400))
        self.fileSelectorTableWidget.setMaximumSize(
            QtCore.QSize(16777215, 16777215))
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
        self.menuModels = QtWidgets.QMenu(self.menubar)
        self.menuModels.setObjectName('menuModels')
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
        self.menubar.addAction(self.menuModels.menuAction())
        self.menubar.addAction(self.menuPomoc.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate(
            'MainWindow', 'YAYA - YOLO annoter'))
        self.label_6.setText(_translate('MainWindow', 'Identity anchored'))
        self.imagePreview.setText(_translate('MainWindow', 'Image'))
        self.imageDesc.setText(_translate('MainWindow', 'Desc'))
        self.label_3.setText(_translate(
            'MainWindow', 'Identity images (selected image, identity images).'))
        self.identityCompareLabel.setText(_translate(
            'MainWindow', 'Comparison identity gallery :'))
        self.modelDetails.setText(_translate(
            'MainWindow', 'Filename (number/all)'))
        self.modelResults.setText(_translate('MainWindow', 'TextLabel'))
        self.identitiesSelectedLabel.setText(
            _translate('MainWindow', 'Identities'))
        self.mergeIdentitiesButton.setText(_translate('MainWindow', 'Merge'))
        self.selectionIdentity.setText(
            _translate('MainWindow', 'Selection identity'))
        self.selectionIdentityCompared.setText(_translate(
            'MainWindow', 'Selection comparison gallery'))
        self.fileSelectorTableWidget.setSortingEnabled(True)
        self.menuMenu.setTitle(_translate('MainWindow', 'File'))
        self.menuPomoc.setTitle(_translate('MainWindow', 'Help'))
        self.menuEdit.setTitle(_translate('MainWindow', 'Edit'))
        self.menuModels.setTitle(_translate('MainWindow', 'Models'))
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
