'''
Created on 29 gru 2020

@author: spasz
'''
import shutil
import sys
import os
import logging
from ReID.FeaturesClassifier import FeaturesClassifier
from ReID.ReidClassifier import CreateReidClassifier, GetReidClassifier, ModelsList
from Ui_MainWindow import Ui_MainWindow
from PyQt5.QtWidgets import QAction, QApplication, QMainWindow, QFileDialog, QTableWidgetItem,\
    QListWidgetItem, QButtonGroup, QMessageBox
from ViewerEditorImage import ViewerEditorImage
from engine.AnnoterReid import AnnoterReid
from helpers.algebra import CosineSimilarity, SimilarityMethod
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

        # Menu : Add classifier model to menu Models
        models = ModelsList()
        for index, model in enumerate(models):
            modelName, modelPath = model
            action = QAction(f'{modelName}:{modelPath}', self.window)
            action.setToolTip(str(index))
            self.ui.menuModels.addAction(action)
        # Menu : Add callback to menu Models
        self.ui.menuModels.triggered.connect(self.CallbackModelClicked)

        # # Buttons group - for mode buttons
        self.modeButtonGroup = QButtonGroup(self.window)
        self.modeButtonGroup.addButton(self.ui.selectionIdentity)
        self.modeButtonGroup.addButton(self.ui.selectionIdentityCompared)
        # Default Identity selection radio button enabled
        self.ui.selectionIdentity.setChecked(True)

        # Buttons Image
        self.ui.mergeIdentitiesButton.clicked.connect(
            self.CallbackMergeIdentitiesButton)

        # Gallery Callbacks :
        self.ui.identityGallery.itemClicked.connect(
            self.CallbackGalleryItemClicked)
        self.ui.identityGallery.itemSelectedDelete.connect(
            self.CallbackGalleryItemDeleted)

    def Setup(self):
        ''' Setup again UI.'''
        # Identity number : Default first
        if (self.identityNumber is None):
            self.identityNumber = self.annoter.indentities_ids[0]

        # Identitiy selcted image number : Default first
        if (self.identityImageNumber is None):
            self.identityImageNumber = 0

        # REID classifier : Get
        reidClassifier = GetReidClassifier()
        self.ui.modelDetails.setText(
            f'{reidClassifier.model_name}:{reidClassifier.model_path}')

        # Identity : Get current
        identity = self.annoter.identities[self.identityNumber]
        # Identity : Update image features with current reid clasifier
        identity.FeaturesUpdate(reidClassifier)

        # REID classifier : View all model details
        self.ui.modelResults.setText(f'**ALL : consistency: {self.annoter.consistency_avg*100:2.2f}%,\n' +
                                     f'separation: {self.annoter.separation_avg*100:2.2f}%.**\n\n'
                                     f'**Identity** consistency: {identity.consistency*100:2.2f}%,\n' +
                                     f'separation: {self.annoter.SeparationAvg(identity)*100:2.2f}%.'
                                     )

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
            # Identity2 (compared) : Get
            identityCompared = self.annoter.identities[self.identityComparedNumber]
            # Identitiy2 : Features update
            identityCompared.FeaturesUpdate(reidClassifier)

            # Identitiy compared : Label
            self.ui.identityCompareLabel.setText(
                f'Identity compared : {self.identityComparedNumber}.')

            # Identitiy compared : View
            ViewIdentityCorelations.View(self.ui.identityCompareGallery,
                                         identity1=identity,
                                         imageIndex1=self.identityImageNumber,
                                         identity2=identityCompared,
                                         method=self.similarityMethod)

        # Identities : Label
        text = f'Selected {self.identityNumber} '
        if (self.identityComparedNumber is not None):
            text += f' compared {self.identityComparedNumber}.'
        self.ui.identitiesSelectedLabel.setText(text)

    def Run(self):
        '''  Run gui window thread and return exit code.'''
        self.window.show()
        return self.App.exec_()

    def CallbackModelClicked(self, action: QAction):
        ''' Callback when model was clicked.'''
        models = ModelsList()
        # Model get  by index from tooltip.
        model = models[int(action.toolTip())]

        # Model unpack to name and path
        modelName, modelPath = model

        CreateReidClassifier(modelName, modelPath)
        self.Setup()

    def CallbackGalleryItemClicked(self, item: QListWidgetItem):
        ''' Callback when gallery item was clicked.'''
        # Identity number : Get
        self.identityImageNumber = int(item.toolTip())

        # Setup UI again
        self.Setup()

    def CallbackGalleryItemDeleted(self, item: QListWidgetItem):
        ''' Callback when gallery item was clicked.'''
        # Identity number : Get
        deletedImageNumber = int(item.toolTip())

        # Identity : Get
        identity = self.annoter.identities[self.identityNumber]
        # Identity : Delete image
        identity.DeleteImage(deletedImageNumber)

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

            # Identity : Get
            identity = self.annoter.identities[self.identityNumber]
            # Identity similarites to all identities : Get
            similarities = self.annoter.Similarities(identity)
            # Table : Update again
            ViewImagesTable.Update(self.ui.fileSelectorTableWidget,
                                   self.annoter.identities,
                                   similarities)

        # Selection : Identity compared selection
        elif (self.ui.selectionIdentityCompared.isChecked()):
            self.identityComparedNumber = int(item.toolTip())

        # Setup UI again
        self.Setup()

    def CallbackMergeIdentitiesButton(self):
        ''' Merge two identities.'''
        # Check : Ideentiti selected number is not None
        if (self.identityNumber is None):
            return

        # Check : Identity compared number is not None
        if (self.identityComparedNumber is None):
            return

        # Check : Identity number is not equal to identity compared number
        if (self.identityNumber == self.identityComparedNumber):
            return

        # Identity1 : Get
        identity1 = self.annoter.identities[self.identityNumber]
        # Identity2 : Get
        identity2 = self.annoter.identities[self.identityComparedNumber]

        # Identity1 : Merge identiy1 <- identity2
        identity1.Merge(identity2)

        # Identity2 : Remove from annoter
        self.annoter.Remove(identity2)
        # Identity2 : Remove from table (find row with identity2 first)
        for rowNumber in range(self.ui.fileSelectorTableWidget.rowCount()):
            identityNumber = int(
                self.ui.fileSelectorTableWidget.item(rowNumber, 0).toolTip())
            if (identityNumber == self.identityComparedNumber):
                self.ui.fileSelectorTableWidget.removeRow(rowNumber)
                break

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
