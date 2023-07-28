'''
Created on 29 gru 2020

@author: spasz
'''
import shutil
import sys
import os
import logging
from ObjectDetectors.common.Detector import NmsMethod
from ReID.FeaturesClassifier import FeaturesClassifier
from Ui_MainWindow import Ui_MainWindow
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QTableWidgetItem,\
    QListWidgetItem, QButtonGroup, QMessageBox
from PyQt5 import QtCore, QtGui
from ViewerEditorImage import ViewerEditorImage
from engine.AnnoterReid import AnnoterReid
from helpers.files import ChangeExtension, FixPath
from PyQt5.QtCore import Qt
from datetime import datetime
from views.ViewIdentityCorelations import ViewIdentityCorelations
from views.ViewIdentityGallery import ViewIdentityGallery
from views.ViewImagesSummary import ViewImagesSummary
from views.ViewImagesTable import ViewImagesTable
from views.ViewImagesTableRow import ViewImagesTableRow


class MainWindowGui(Ui_MainWindow):
    '''
    classdocs
    '''

    def __init__(self,
                 args,
                 reid_classifier: FeaturesClassifier,
                 annoter: AnnoterReid,
                 ):
        '''
        Constructor
        '''
        # Config
        self.info = {'Callbacks': True}
        # Store initial arguments
        self.args = args
        # Store annoter handle
        self.annoter = annoter
        # Identity number
        self.identityNumber = None
        # Idenitity selected image number
        self.identitySelectedImageNumber = None

        # UI - creation
        self.App = QApplication(sys.argv)
        self.ui = Ui_MainWindow()
        self.window = QMainWindow()
        self.ui.setupUi(self.window)

        # Setup all
        self.SetupDefault()
        self.Setup()

    def ImageIDToRowNumber(self, imageID):
        ''' Image number to row index.'''
        # Found index
        foundIndex = None

        # Find rowIndex of imageNumber
        for rowIndex in range(self.ui.fileSelectorTableWidget.rowCount()):
            item = self.ui.fileSelectorTableWidget.item(rowIndex, 0)
            if (int(item.toolTip()) == imageID):
                foundIndex = rowIndex
                break

        return foundIndex

    def SetupDefault(self):
        ''' Sets default for UI.'''
        # Images table : Setup
        ViewImagesTable.View(self.ui.fileSelectorTableWidget,
                             self.annoter.identities)
        self.ui.fileSelectorTableWidget.itemClicked.connect(
            self.CallbackFileSelectorItemClicked)

        # # Images summary : Setup
        # ViewImagesSummary.View(self.ui.fileSummaryLabel,
        #                        self.annoter.identities)

        # Menu handling
        self.ui.actionZamknijProgram.triggered.connect(self.CallbackClose)
        self.ui.actionZapisz.triggered.connect(
            self.CallbackSaveFileAnnotationsButton)
        self.ui.actionOtworzLokacje.triggered.connect(
            self.CallbackOpenLocation)
        self.ui.actionSave_copy.triggered.connect(
            self.CallbackSaveCopy)

        # # Buttons group - for mode buttons
        # self.modeButtonGroup = QButtonGroup(self.window)
        # self.modeButtonGroup.addButton(self.ui.addAnnotationsButton)
        # self.modeButtonGroup.addButton(self.ui.removeAnnotationsButton)
        # self.modeButtonGroup.addButton(self.ui.paintCircleButton)
        # self.modeButtonGroup.addButton(self.ui.renameAnnotationsButton)

        # Buttons player
        self.ui.nextFileButton.clicked.connect(self.CallbackNextFile)
        self.ui.prevFileButton.clicked.connect(self.CallbackPrevFile)
        # Buttons Image
        self.ui.SaveFileAnnotationsButton.clicked.connect(
            self.CallbackSaveFileAnnotationsButton)
        self.ui.DeleteImageAnnotationsButton.clicked.connect(
            self.CallbackDeleteImageAnnotationsButton)
        # # Buttons - Annotations
        # self.ui.addAnnotationsButton.clicked.connect(
        #     self.CallbackAddAnnotationsButton)
        # self.ui.renameAnnotationsButton.clicked.connect(
        #     self.CallbackRenameAnnotationsButton)
        # self.ui.removeAnnotationsButton.clicked.connect(
        #     self.CallbackRemoveAnnotationsButton)
        # self.ui.detectAnnotationsButton.clicked.connect(
        #     self.CallbackDetectAnnotations)
        # self.ui.hideLabelsButton.clicked.connect(
        #     self.CallbackHideLabelsButton)
        # self.ui.hideAnnotationsButton.clicked.connect(
        #     self.CallbackHideAnnotationsButton)
        # self.ui.ClearAnnotationsButton.clicked.connect(
        #     self.CallbackClearAnnotationsButton)
        # # Buttons - Painting
        # self.ui.paintCircleButton.clicked.connect(
        #     self.CallbackPaintCircleButton)

        # Gallery Callbacks :
        self.ui.gallery.itemClicked.connect(self.CallbackGalleryItemClicked)

    def Setup(self):
        ''' Setup again UI.'''
        # Identity number : Default first
        if (self.identityNumber is None):
            self.identityNumber = self.annoter.indentities_ids[0]

        # Identitiy selcted image number : Default first
        if (self.identitySelectedImageNumber is None):
            self.identitySelectedImageNumber = 0

        # Identity : Get current
        identity = self.annoter.identities[self.identityNumber]

        # Identity Preview : Show
        ViewIdentityGallery.View(self.ui.gallery,
                                 identity)

        # Identity Preview : Correlations
        ViewIdentityCorelations.View(self.ui.imagePreview,
                                     self.ui.imageDesc,
                                     self.ui.imageCorelations,
                                     identity,
                                     self.identitySelectedImageNumber
                                     )

    def Run(self):
        '''  Run gui window thread and return exit code.'''
        self.window.show()
        return self.App.exec_()

    def CallbackGalleryItemClicked(self, item: QListWidgetItem):
        ''' Callback when gallery item was clicked.'''
        # Identity number : Get
        self.identitySelectedImageNumber = int(item.toolTip())

        # Setup UI again
        self.Setup()

    def CallbackImageScalingTextChanged(self, text):
        ''' Callback when image scaling text changed.'''

    def CallbackLabelsRowChanged(self, index):
        ''' Current labels row changed. '''
        self.ui.viewerEditor.SetClassNumber(index)

    def CallbackFileSelectorItemClicked(self, item):
        ''' When file selector item was clicked.'''
        # Identity number : Get
        self.identityNumber = int(item.toolTip())
        # Idenitity image number : Reset to None
        self.identitySelectedImageNumber = None

        # Setup UI again
        self.Setup()

    def CallbackPaintSizeSlider(self):
        ''' Paint size slider changed.'''
        self.ui.viewerEditor.SetEditorModeArgument(
            self.ui.paintSizeSlider.value())
        self.Setup()

    def CallbackPaintCircleButton(self):
        ''' Paint circle button.'''
        self.ui.toolSettingsStackedWidget.setCurrentWidget(self.ui.pageCircle)
        self.ui.viewerEditor.SetEditorMode(ViewerEditorImage.ModePaintCircle,
                                           argument=self.ui.paintSizeSlider.value())

    def CallbackDetectAnnotations(self):
        ''' Detect annotations.'''
        self.ui.toolSettingsStackedWidget.setCurrentWidget(
            self.ui.pageDetector)
        self.Setup()

    def CallbackDetectorUpdate(self):
        ''' Detector update.'''
        self.ui.detectorConfidenceLabel.setText(
            f'Confidence: {self.ui.detectorConfidenceSlider.value()/100:02}%')
        self.ui.detectorNmsLabel.setText(
            f'NMS: {self.ui.detectorNmsSlider.value()/100:02}%')

    def CallbackAddAnnotationsButton(self):
        ''' Remove annotations.'''
        self.ui.toolSettingsStackedWidget.setCurrentWidget(
            self.ui.pageAnnotations)
        self.ui.viewerEditor.SetEditorMode(
            ViewerEditorImage.ModeAddAnnotation)

    def CallbackRenameAnnotationsButton(self):
        ''' Remove annotations.'''
        self.ui.toolSettingsStackedWidget.setCurrentWidget(
            self.ui.pageAnnotations)
        self.ui.viewerEditor.SetEditorMode(
            ViewerEditorImage.ModeRenameAnnotation)

    def CallbackRemoveAnnotationsButton(self):
        ''' Remove annotations.'''
        self.ui.toolSettingsStackedWidget.setCurrentWidget(
            self.ui.pageAnnotations)
        self.ui.viewerEditor.SetEditorMode(
            ViewerEditorImage.ModeRemoveAnnotation)

    def CallbackHideLabelsButton(self):
        '''Callback'''
        self.ui.viewerEditor.SetOption('isLabelsHidden',
                                       not self.ui.viewerEditor.GetOption('isLabelsHidden'))

    def CallbackHideAnnotationsButton(self):
        '''Callback'''
        self.ui.viewerEditor.SetOption('isAnnotationsHidden',
                                       not self.ui.viewerEditor.GetOption('isAnnotationsHidden'))

    def CallbackSaveFileAnnotationsButton(self):
        '''Callback'''
        self.Setup()

    def CallbackClearAnnotationsButton(self):
        '''Callback'''
        self.Setup()

    def CallbackDeleteImageAnnotationsButton(self):
        '''Callback'''
        # # Remove QtableWidget row
        # rowIndex = self.ImageIDToRowNumber(self.annoter.GetFileID())
        # if (rowIndex is not None):
        #     self.ui.fileSelectorTableWidget.removeRow(rowIndex)
        #     # Remove annoter data
        #     self.annoter.Delete()
        #     self.annoter.Process()
        #     self.Setup()

    def CallbackNextFile(self):
        '''Callback'''
        self.Setup()

    def CallbackPrevFile(self):
        '''Callback'''
        self.Setup()

    def CallbackOpenLocation(self):
        ''' Open location callback.'''
        filepath = str(QFileDialog.getExistingDirectory(
            None, 'Select Directory'))
        self.annoter.OpenLocation(FixPath(filepath))

        self.SetupDefault()
        self.Setup()

    def CallbackClose(self):
        ''' Close GUI callback.'''
        logging.debug('Closing application!')
        self.window.close()

    def CallbackSaveCopy(self):
        ''' Save configuration.'''
        return
        file = self.annoter.GetFile()

        if (file is not None):
            # Set default screenshots location
            if (os.name == 'posix'):
                screenshotsPath = '/home/%s/Obrazy/' % (
                    os.environ.get('USER', 'Error'))
            else:
                screenshotsPath = 'C:\\'

            # Copy original image file to screenshots path
            shutil.copy(file['Path'],
                        screenshotsPath+file['Name'])

            # Copy annotations .txt file to screenshots path
            annotationsPath = ChangeExtension(file['Path'], '.txt')
            if (os.path.exists(annotationsPath)):
                shutil.copy(annotationsPath,
                            screenshotsPath+ChangeExtension(file['Name'], '.txt'))
