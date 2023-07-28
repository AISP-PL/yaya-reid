'''
Created on 29 gru 2020

@author: spasz
'''
import shutil
import sys
import os
import logging
from ReID.FeaturesClassifier import FeaturesClassifier
from Ui_MainWindow import Ui_MainWindow
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QTableWidgetItem,\
    QListWidgetItem, QButtonGroup, QMessageBox
from ViewerEditorImage import ViewerEditorImage
from engine.AnnoterReid import AnnoterReid
from helpers.algebra import SimilarityMethod
from helpers.files import ChangeExtension, FixPath
from views.ViewIdentity import ViewIdentity
from views.ViewIdentityCorelations import ViewIdentityCorelations
from views.ViewImagesTable import ViewImagesTable


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

        # Identity1 : number
        self.identityNumber = None
        # Idenitity1 : Image number
        self.identityImageNumber = None

        # Identity2 (comapared) : number
        self.identityComparedNumber = None

        # UI - creation
        self.App = QApplication(sys.argv)
        self.ui = Ui_MainWindow()
        self.window = QMainWindow()
        self.ui.setupUi(self.window)

        # Setup all
        self.SetupDefault()
        self.Setup()

    @property
    def similarityMethod(self) -> SimilarityMethod:
        ''' Read from qcombobox and return SimilarityMethod.'''
        return SimilarityMethod[self.ui.similarityMethod.currentText()]

    def SetupDefault(self):
        ''' Sets default for UI.'''
        # Images table : Setup
        ViewImagesTable.View(self.ui.fileSelectorTableWidget,
                             self.annoter.identities)
        self.ui.fileSelectorTableWidget.itemClicked.connect(
            self.CallbackIdentitySelectorItemClicked)

        # ComboBox with method from Enum SimilarityMethod.
        for method in SimilarityMethod:
            self.ui.similarityMethod.addItem(method.name)

        # Menu handling
        self.ui.actionZamknijProgram.triggered.connect(self.CallbackClose)
        self.ui.actionZapisz.triggered.connect(
            self.CallbackSaveFileAnnotationsButton)
        self.ui.actionOtworzLokacje.triggered.connect(
            self.CallbackOpenLocation)
        self.ui.actionSave_copy.triggered.connect(
            self.CallbackSaveCopy)

        # # Buttons group - for mode buttons
        self.modeButtonGroup = QButtonGroup(self.window)
        self.modeButtonGroup.addButton(self.ui.selectionIdentity)
        self.modeButtonGroup.addButton(self.ui.selectionIdentityCompared)
        # Default Identity selection radio button enabled
        self.ui.selectionIdentity.setChecked(True)

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
        self.ui.identityGallery.itemClicked.connect(
            self.CallbackGalleryItemClicked)

    def Setup(self):
        ''' Setup again UI.'''
        # Identity number : Default first
        if (self.identityNumber is None):
            self.identityNumber = self.annoter.indentities_ids[0]

        # Identitiy selcted image number : Default first
        if (self.identityImageNumber is None):
            self.identityImageNumber = 0

        # Identity : Get current
        identity = self.annoter.identities[self.identityNumber]

        # View : Identity
        ViewIdentity.View(self.ui.imagePreview,
                          self.ui.imageDesc,
                          identity,
                          self.identityImageNumber)

        # Identity Preview : Correlations
        ViewIdentityCorelations.View(self.ui.identityGallery,
                                     identity1=identity,
                                     imageIndex1=self.identityImageNumber,
                                     identity2=identity,
                                     method=self.similarityMethod)

        # Identity compared preview : Correlations
        if (self.identityComparedNumber is not None):
            ViewIdentityCorelations.View(self.ui.identityCompareGallery,
                                         identity1=identity,
                                         imageIndex1=self.identityImageNumber,
                                         identity2=self.annoter.identities[self.identityComparedNumber],
                                         method=self.similarityMethod)

    def Run(self):
        '''  Run gui window thread and return exit code.'''
        self.window.show()
        return self.App.exec_()

    def CallbackGalleryItemClicked(self, item: QListWidgetItem):
        ''' Callback when gallery item was clicked.'''
        # Identity number : Get
        self.identityImageNumber = int(item.toolTip())

        # Setup UI again
        self.Setup()

    def CallbackImageScalingTextChanged(self, text):
        ''' Callback when image scaling text changed.'''

    def CallbackLabelsRowChanged(self, index):
        ''' Current labels row changed. '''
        self.ui.viewerEditor.SetClassNumber(index)

    def CallbackIdentitySelectorItemClicked(self, item):
        ''' When file selector item was clicked.'''
        # Selection : Base identity selection
        if (self.ui.selectionIdentity.isChecked()):
            # Identity number : Get
            self.identityNumber = int(item.toolTip())
            # Idenitity image number : Reset to None
            self.identityImageNumber = None

        # Selection : Identity compared selection
        elif (self.ui.selectionIdentityCompared.isChecked()):
            self.identityComparedNumber = int(item.toolTip())

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
