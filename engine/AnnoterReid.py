'''
Created on 17 lis 2020

@author: spasz
'''
from __future__ import annotations
from dataclasses import dataclass, field
from math import sqrt
import os
import re
import time
from ReID.FeaturesClassifier import FeaturesClassifier
import cv2
import logging

import numpy as np
from helpers.algebra import Normalize, NormalizedVectorToInt, Pooling1dToSize
from helpers.files import IsImageFile, DeleteFile, GetNotExistingSha1Filepath, FixPath, GetFilename,\
    GetExtension
from helpers.textAnnotations import ReadAnnotations, SaveAnnotations, IsExistsAnnotations,\
    DeleteAnnotations, SaveDetections, ReadDetections
from helpers.visuals import Visuals


@dataclass
class ReidFileInfo:
    ''' Informations stored in name of reid image file.'''
    # Identity number
    identity: int = field(init=True, default=None)
    # Camera number
    camera: int = field(init=True, default=None)
    # Frame number
    frame: int = field(init=True, default=None)

    @staticmethod
    def PatternAispReid(text: str) -> ReidFileInfo:
        ''' Parse AISP reid filename.'''
        # Filename pattern
        pattern = re.compile(r'ID([-\d]+)_CAM(\d)_FRAME(\d)')
        # Regular expression : Get results
        regexResults = pattern.search(text)
        if regexResults is None:
            return None

        # Get pid, camid, frame
        pid, camid, frame = map(int, regexResults.groups())

        return ReidFileInfo(pid, camid, frame)

    @staticmethod
    def FromFilename(filename: str) -> ReidFileInfo:
        ''' Create fileinfo from filename.'''
        # Patter : AISP
        result = ReidFileInfo.PatternAispReid(filename)
        if result is not None:
            return result

        # Pattern : Market1501
        # @TODO

        return None


@dataclass
class ImageData:
    ''' Dataclass representing image and features and visuals.'''
    # Image path
    path: str = field(init=True, default=None)
    # Camera number
    camera: int = field(init=True, default=1)
    # Image visuals
    visuals: Visuals = field(init=True, default=None)
    # Image features
    features: np.array = field(init=True, default=None)

    @property
    def name(self) -> str:
        ''' Return name of image.'''
        return os.path.basename(self.path)


@dataclass
class Identity:
    ''' Class representing identity with all images.'''
    # Identity number :
    number: int = field(init=True, default=None)
    # Identity ImageData list
    images: list = field(init=True, default=None)

    @property
    def image(self) -> ImageData:
        ''' Return first image.'''
        # Check : Images list is not empty
        if (len(self.images) == 0):
            return None

        return self.images[0]

    @property
    def images_count(self) -> int:
        ''' Count of images.'''
        return len(self.images)

    @property
    def hue(self) -> float:
        ''' Return average hue of all images.'''
        # Check : Images list is not empty
        if (len(self.images) == 0):
            return None

        # Get hue
        hue = [image.visuals.hue for image in self.images]
        return np.mean(hue)

    @property
    def brightness(self) -> float:
        ''' Return average brightness of all images.'''
        # Check : Images list is not empty
        if (len(self.images) == 0):
            return None

        # Get brightness
        brightness = [image.visuals.brightness for image in self.images]
        return np.mean(brightness)

    @property
    def saturation(self) -> float:
        ''' Return average saturation of all images.'''
        # Check : Images list is not empty
        if (len(self.images) == 0):
            return None

        # Get saturation
        saturation = [image.visuals.saturation for image in self.images]
        return np.mean(saturation)

    @property
    def imhash(self) -> float:
        ''' Return average imhash of all images.'''
        # Check : Images list is not empty
        if (len(self.images) == 0):
            return None

        # Get imhash
        imhash = [image.visuals.dhash for image in self.images]
        return np.mean(imhash)

    @property
    def features(self) -> np.array:
        ''' Return average features of all np.arrays.'''
        # Check : Images list is not empty
        if (len(self.images) == 0):
            return None

        # Get features
        features = [image.features for image in self.images]
        # Get average
        average = np.mean(features, axis=0)

        return average

    @property
    def features_binrepr(self) -> int:
        ''' Return int(binary) representation of features vector.'''
        vector = Pooling1dToSize(self.features, size=64)
        vector_norm = Normalize(vector)
        return NormalizedVectorToInt(vector_norm)


@dataclass
class AnnoterReid:
    ''' Class reading all images and annotations.'''
    # Path to directory with images
    dirpath: str = field(init=True, default=None)
    # Arguments : Namespace from argparse
    args: object = field(init=True, default=None)
    # ReID features classifier
    features_classifier: FeaturesClassifier = field(init=True, default=None)
    # Found identities list
    identities: list = field(init=False, default_factory=list)

    def __post_init__(self):
        ''' Post init method.'''
        # Check : Features classifier is not None
        if (self.features_classifier is None):
            raise ValueError('Features classifier is None!')

        # Location : Open and parse data
        self.OpenLocation(self.dirpath)

    @property
    def indentities_ids(self) -> list:
        ''' Return list of identities ids.'''
        return list(self.identities.keys())

    @property
    def identities_count(self) -> int:
        ''' Count of identities.'''
        return len(self.identities)

    @property
    def images_count(self) -> int:
        ''' Count of images.'''
        return sum([len(identity.images) for identity in self.identities])

    @staticmethod
    def ImagenameToReidInfo(imagename: str) -> int:
        ''' Convert imagename to identity number.'''
        # Filename : Get filename
        filename = GetFilename(imagename)
        # Identity : Get identity number
        identity = int(filename.split('_')[0])
        return identity

    def OpenLocation(self, path: str):
        ''' Open images/annotations location.'''
        # Check : Check if path exists
        if (not os.path.exists(path)):
            logging.error('(Annoter) Path `%s` not exists!', path)
            return

        # Dirpath : Store
        self.dirpath = path

        # Excludes : List of excludes
        excludes = ['.', '..', './', '.directory']
        # Images : List all directory images.
        images = [filename for filename in os.listdir(path)
                  if (filename not in excludes) and (IsImageFile(filename))]

        # Identities : Create identities
        self.identities = {}
        # Processing all files
        startTime = time.time()
        for index, imagename in enumerate(images):
            # Filepath : Create filepath
            imagepath = f'{path}{imagename}'

            # ReidInfo : Get reid info
            reidInfo = ReidFileInfo.FromFilename(imagename)

            # Calculate visuals
            visuals = Visuals.LoadCreate(imagepath=imagepath)

            # Features : LoadCreate features
            features = self.features_classifier.LoadCreate(imagepath=imagepath)

            # Identity : Create identity if not exists
            if (reidInfo.identity not in self.identities):
                self.identities[reidInfo.identity] = Identity(number=reidInfo.identity,
                                                              images=[])

            # Identity : Append image
            self.identities[reidInfo.identity].images.append(ImageData(path=imagepath,
                                                                       camera=reidInfo.camera,
                                                                       visuals=visuals,
                                                                       features=features))

    # def GetFileAnnotations(self, filepath):
    #     ''' Read file annotations if possible.'''
    #     txtAnnotes = []
    #     # If exists annotations file
    #     if (IsExistsAnnotations(filepath)):
    #         txtAnnotes = ReadAnnotations(filepath)
    #         txtAnnotes = [annote.fromTxtAnnote(el) for el in txtAnnotes]
    #         logging.debug(
    #             '(Annoter) Loaded annotations from %s!', filepath)

    #         # Post-check of errors
    #         self.errors = self.__checkOfErrors()

    #     return txtAnnotes

    # def GetFileImage(self, filepath: str):
    #     ''' Read file annotations if possible.'''
    #     if (filepath is None) or (len(filepath) == 0):
    #         return None

    #     if (not os.path.exists(filepath)):
    #         return None

    #     try:
    #         image = cv2.imread(filepath)
    #         return image
    #     except:
    #         logging.fatal(
    #             '(Annoter) CV2 error when readings image `%s`!', filepath)
    #         return None

    # def ReadFileDetections(self, filepath: str):
    #     ''' Read file annotations if possible.'''
    #     detAnnotes = []
    #     # If detector annotations not exists then call detector
    #     if (IsExistsAnnotations(filepath, extension='.detector')):
    #         detAnnotes = ReadDetections(filepath, extension='.detector')
    #         detAnnotes = [annote.fromDetection(el) for el in detAnnotes]

    #     return detAnnotes

    # def ProcessFileDetections(self, im, filepath) -> list:
    #     ''' Read file annotations if possible.'''
    #     if (self.detector is None) or (im is None):
    #         return []

    #     # Call detector manually!
    #     detAnnotes = self.detector.Detect(im,
    #                                       confidence=self.confidence,
    #                                       nms_thresh=self.nms,
    #                                       boxRelative=True,
    #                                       nmsMethod=self.nmsMethod)

    #     # Save/Update detector annotations file
    #     SaveDetections(filepath, detAnnotes, extension='.detector')

    #     # Create annotes
    #     detAnnotes = [annote.fromDetection(el) for el in detAnnotes]
    #     return detAnnotes

    # def CalculateYoloMetrics(self, txtAnnotes: list, detAnnotes: list) -> Metrics:
    #     ''' Calculate mAP between two annotations sets.'''
    #     metrics = Metrics()
    #     if (len(txtAnnotes)):
    #         metrics = EvaluateMetrics(txtAnnotes, detAnnotes)

    #     return metrics

    # def GetFile(self):
    #     ''' Returns current filepath.'''
    #     if (len(self.files)):
    #         return self.files[self.offset]

    #     return None

    # def GetFilename(self):
    #     ''' Returns current filepath.'''
    #     if (len(self.files)):
    #         return self.files[self.offset]['Name']

    #     return 'Not exists!'

    # def GetFilepath(self):
    #     ''' Returns current filepath.'''
    #     return self.files[self.offset]['Path']

    # def GetErrors(self):
    #     ''' Returns current errors list.'''
    #     return self.errors

    # def GetImage(self):
    #     ''' Returns current image.'''
    #     return self.image

    # def GetImageSize(self):
    #     ''' Returns current image.'''
    #     if (self.image is not None):
    #         h, w, bytes = self.image.shape
    #         return w, h, bytes

    #     return 0, 0, 0

    # def GetAnnotationsList(self):
    #     ''' Returns annotations list'''
    #     return [GetFilename(f['Name'])+'.txt' for f in self.files]

    # def GetFiles(self):
    #     ''' Returns images list'''
    #     return self.files

    # def GetFileID(self):
    #     ''' Returns current image ID.'''
    #     if (self.offset < len(self.files)):
    #         return self.files[self.offset]['ID']

    #     return 0

    # def GetFileIndex(self):
    #     ''' Returns current image number.'''
    #     return self.offset

    # def GetFilesCount(self) -> int:
    #     ''' Returns count of processed images number.'''
    #     if (self.files is None):
    #         return 0

    #     return len(self.files)

    # def GetFilesAnnotatedCount(self):
    #     ''' Returns count of processed images number.'''
    #     annotated = sum([int(fileEntry['IsAnnotation'])
    #                      for fileEntry in self.files])
    #     return annotated

    # def SetImageID(self, fileID):
    #     ''' Sets current image number.'''
    #     # index of ID file
    #     foundIndex = None
    #     # Find image with these id
    #     for index, fileEntry in enumerate(self.files):
    #         if (fileEntry['ID'] == fileID):
    #             foundIndex = index
    #             break

    #     if (foundIndex is not None) and (foundIndex) and (foundIndex < len(self.files)):
    #         self.offset = foundIndex
    #         self.Process()
    #         return True

    #     return False

    # def SetImageNumber(self, number):
    #     ''' Sets current image number.'''
    #     if (number >= 0) and (number < len(self.files)):
    #         self.offset = number
    #         self.Process()
    #         return True

    #     return False

    # def TransformShape(self, mode='Flip'):
    #     ''' Transform image and annotations shape.'''
    #     if (mode == 'Flip'):
    #         self.image = transformations.Flip(self.image)
    #         for a in self.annotations:
    #             a.box = boxes.FlipHorizontally(1, a.box)

    #     # Store image modification info
    #     for a in self.annotations:
    #         a.authorType = annote.AnnoteAuthorType.byHuman
    #     self.errors.add('ImageModified!')

    # def PaintCircles(self, points, radius, color):
    #     ''' Paint list of circles Circle on image.'''
    #     for x, y in points:
    #         self.image = cv2.circle(self.image,
    #                                 (round(x), round(y)),
    #                                 radius,
    #                                 color,
    #                                 -1)
    #     self.errors.add('ImageModified!')

    # def GetAnnotations(self):
    #     ''' Returns current annotations.'''
    #     return self.annotations

    # @property
    # def annotations_count(self) -> int:
    #     ''' Returns current annotations.'''
    #     return len(self.annotations)

    # @staticmethod
    # def AnnotationsSelectClasses(annotations, classes):
    #     ''' Returns current annotations.'''
    #     if (annotations is not None):
    #         annotations = [a for a in annotations if (
    #             a.classNumber in classes)]

    #     return annotations

    # def GetAnnotationsSelectClasses(self, classes):
    #     ''' Returns current annotations.'''
    #     return self.AnnotationsSelectClasses(self.annotations, classes)

    # def AddAnnotation(self, box, classNumber):
    #     ''' Adds new annotation by human.'''
    #     self.annotations.append(annote.Annote(
    #         box, classNumber=classNumber, authorType=annote.AnnoteAuthorType.byHand))
    #     logging.debug('(Annoter) Added annotation class %u!', classNumber)
    #     self.errors = self.__checkOfErrors()

    # def ClearAnnotations(self):
    #     ''' Clear all annotations.'''
    #     self.annotations = []
    #     self.errors = self.__checkOfErrors()
    #     logging.debug('(Annoter) Cleared annotations!')

    # def RemoveAnnotation(self, element):
    #     '''Remove annotation .'''
    #     if (len(self.annotations) != 0):
    #         self.annotations.remove(element)
    #         self.errors = self.__checkOfErrors()

    # def IsSynchronized(self):
    #     ''' Is all annotations synchronized with file.'''
    #     isSynchronized = True
    #     for element in self.annotations:
    #         if (element.GetAuthorType() != annote.AnnoteAuthorType.byHuman):
    #             isSynchronized = False
    #             break

    #     return isSynchronized

    # def __isClearImageSynchronized(self):
    #     ''' True if image was modified.'''
    #     result = not ('ImageModified!' in self.errors)
    #     if (result is False):
    #         self.errors.remove('ImageModified!')
    #     return result

    # def Delete(self, fileEntry=None):
    #     ''' Deletes image and annotations.'''
    #     # If not specified fileEntry then use
    #     # current fileEntry.
    #     if (fileEntry is None):
    #         # Use current file
    #         if (self.offset < self.GetFilesCount()):
    #             fileEntry = self.files[self.offset]

    #     # If entry exists then delete it
    #     if (fileEntry is not None):
    #         self.ClearAnnotations()
    #         DeleteAnnotations(fileEntry['Path'])
    #         DeleteFile(fileEntry['Path'])
    #         self.files.remove(fileEntry)

    #         # Step back offset
    #         self.offset = min(self.offset,
    #                           self.GetFilesCount()-1)

    # def Create(self):
    #     ''' Creates new filepath for new image file.'''
    #     filename = self.GetFilename()
    #     filename, filepath = GetNotExistingSha1Filepath(
    #         filename, self.dirpath)
    #     cv2.imwrite('%s' % (filepath), self.image)
    #     self.filenames.insert(self.offset, filename)
    #     logging.info('(Annoter) New image %s created!', filename)

    # def Save(self):
    #     ''' Save current annotations.'''
    #     filename = self.GetFilename()

    #     # If image was modified, then save it also
    #     if (self.__isClearImageSynchronized() == False):
    #         # Create temporary and original paths
    #         imgpath = FixPath(self.dirpath) + filename
    #         tmppath = FixPath(self.dirpath) + 'tmp' + GetExtension(filename)
    #         # Save temporary image
    #         result = cv2.imwrite(tmppath, self.image)
    #         if (result is False):
    #             logging.error('(Annoter) Writing image "%s" failed!', imgpath)
    #             return

    #         # If saved then atomic move image to original image
    #         os.system('mv -fv "%s" "%s" ' % (tmppath, imgpath))

    #     # Check other errors
    #     self.errors = self.__checkOfErrors()
    #     if (len(self.errors) != 0):
    #         logging.error('(Annoter) Errors exists in annotations!')
    #         return

    #     # Save annotations
    #     annotations = [annote.toTxtAnnote(el) for el in self.annotations]
    #     annotations = SaveAnnotations(
    #         self.dirpath+filename, annotations)
    #     logging.debug('(Annoter) Saved annotations for %s!', filename)

    #     # Process file again after save
    #     self.Process()

    #     # Update file entry
    #     self.files[self.offset]['IsAnnotation'] = True
    #     self.files[self.offset]['Annotations'] = self.annotations
    #     self.files[self.offset]['AnnotationsClasses'] = ','.join(
    #         {f'{item.classNumber}' for item in self.annotations})

    # def IsEnd(self):
    #     '''True if files ended.'''
    #     return (self.offset == self.GetFilesCount())

    # def __checkOfErrors(self):
    #     '''Check current image/annotations for errors.'''
    #     errors = set()
    #     if (len(self.annotations) != len(prefilters.FilterIOUbyConfidence(self.annotations, self.annotations))):
    #         logging.error('(Annoter) Annotations overrides each other!')
    #         errors.add('Override error!')

    #     return errors

    # def Process(self,
    #             processImage=True,
    #             forceDetector=False):
    #     ''' process file.'''
    #     if (self.offset >= 0) and (self.offset < self.GetFilesCount()):
    #         fileEntry = self.GetFile()

    #         # Read image
    #         if (processImage is True):
    #             self.image = im = self.GetFileImage(fileEntry['Path'])

    #         # All txt annotations
    #         txtAnnotations = self.GetFileAnnotations(fileEntry['Path'])
    #         # Detector annotations list
    #         detAnnotes = []
    #         # if annotations file not exists or empty then detect.
    #         if (self.noDetector is False) and ((processImage is True) or (len(txtAnnotations) == 0)):
    #             # Process detector
    #             detAnnotes = self.ProcessFileDetections(im, fileEntry['Path'])

    #             # Calculate metrics
    #             metrics = self.CalculateYoloMetrics(
    #                 txtAnnotations, detAnnotes)
    #             # For view : Filter by IOU internal with same annotes and also with txt annotes.
    #             if (len(txtAnnotations)):
    #                 detAnnotes = prefilters.FilterIOUbyConfidence(detAnnotes,
    #                                                               detAnnotes + txtAnnotations,
    #                                                               maxIOU=sqrt(self.nms))

    #             # Store metrics
    #             fileEntry['Metrics'] = metrics

    #         # All annotations
    #         self.annotations = txtAnnotations + detAnnotes

    #         # Post-check of errors
    #         self.errors = self.__checkOfErrors()

    #         return True

    #     return False

    # def ProcessNext(self):
    #     ''' Process next image.'''
    #     if (self.offset < (self.GetFilesCount()-1)):
    #         self.offset += 1
    #         self.Process()
    #         return True

    #     return False

    # def ProcessPrev(self):
    #     ''' Process next image.'''
    #     if (self.offset > 0):
    #         self.offset -= 1
    #         self.Process()
    #         return True

    #     return False
