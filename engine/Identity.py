'''
    Single ReID identity with all images.

'''
from __future__ import annotations
from dataclasses import dataclass, field
from functools import cached_property
import os
import shutil
from ReID.FeaturesClassifier import FeaturesClassifier
from engine.ReidFileInfo import ReidDataset, ReidFileInfo
from engine.ImageData import ImageData
import numpy as np
from helpers.algebra import CosineSimilarity, EucledeanDistance, MaxPooling1dToSize, Normalize, NormalizedVectorToInt, Pooling1dToSize, SimilarityMethod


@dataclass
class Identity:
    ''' Class representing identity with all images.'''
    # Identity number :
    number: int = field(init=True, default=None)
    # Identity ImageData list
    images: list = field(init=True, default=None)
    # Identity dataset type
    dataset: ReidDataset = field(init=True, default=ReidDataset.AispReid)

    def __post_init__(self):
        ''' Post init.'''
        # Check : Invalid images list
        if (self.images is None):
            self.images = []

        # Similarity matrix : Default is None

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
    def last_frame(self) -> int:
        ''' Return last frame number.'''
        # Check : Images list is not empty
        if (len(self.images) == 0):
            return 0

        # Frame numbers
        frames = sorted([image.frame for image in self.images])
        return frames[-1]

    @cached_property
    def hue(self) -> float:
        ''' Return average hue of all images.'''
        # Check : Images list is not empty
        if (len(self.images) == 0):
            return None

        # Get hue
        hue = [image.visuals.hue for image in self.images]
        return np.mean(hue)

    @cached_property
    def brightness(self) -> float:
        ''' Return average brightness of all images.'''
        # Check : Images list is not empty
        if (len(self.images) == 0):
            return None

        # Get brightness
        brightness = [image.visuals.brightness for image in self.images]
        return np.mean(brightness)

    @cached_property
    def saturation(self) -> float:
        ''' Return average saturation of all images.'''
        # Check : Images list is not empty
        if (len(self.images) == 0):
            return None

        # Get saturation
        saturation = [image.visuals.saturation for image in self.images]
        return np.mean(saturation)

    @cached_property
    def imhash(self) -> float:
        ''' Return average imhash of all images.'''
        # Check : Images list is not empty
        if (len(self.images) == 0):
            return None

        # Get imhash
        imhash = [image.visuals.dhash for image in self.images]
        return np.mean(imhash)

    @cached_property
    def features(self) -> np.array:
        ''' Return average features of all np.arrays.'''
        # Check : Images list is not empty
        if (len(self.images) == 0):
            return None

        # Get features
        features = [image.features for image in self.images]
        # Get average
        average = np.median(features, axis=0)

        return average

    @cached_property
    def features_binrepr(self) -> int:
        ''' Return int(binary) representation of features vector.'''
        features = self.images[0].features
        vector = MaxPooling1dToSize(features, size=64)
        vector_norm = Normalize(vector)
        return NormalizedVectorToInt(vector_norm)

    @cached_property
    def consistency(self) -> float:
        ''' Return min images features cosine distance.
            Named an identity consistency.'''
        # Check : Images list is not empty
        if (len(self.images) == 0):
            return 0

        # Get Image0 to other images similarities
        similarities = self.ImageSimilarities(image=self.images[0],
                                              method=SimilarityMethod.CosineSimilarity)
        return min(similarities)

    def AddImage(self, image: ImageData):
        ''' Add image to identity.'''
        # Check : Image is not None
        if (image is None):
            return None

        # Add image
        self.images.append(image)
        sorted(self.images, key=lambda image: image.frame)

    def DeleteImage(self, imageNumber: int):
        ''' Add image to identity.'''
        # Check : Image number < images count
        if (imageNumber >= len(self.images)):
            return None

        # Image : Get
        image = self.images[imageNumber]

        # Images : Remove
        os.remove(image.path)
        del self.images[imageNumber]

        # Images : Sort
        sorted(self.images, key=lambda image: image.frame)

    def FeaturesUpdate(self, features_classifier: FeaturesClassifier):
        ''' Update features of all images.'''
        # Check : Images list is not empty
        if (len(self.images) == 0):
            return None

        # Update features

        for image in self.images:
            features = features_classifier.LoadCreate(
                imagepath=image.path, force=True)
            image.features = features

    def ImageSimilarities(self, image: ImageData, method: SimilarityMethod) -> list:
        ''' Calculate given image similarities to all identity images.'''
        # Cosine similarity : For all images
        if (method == SimilarityMethod.CosineSimilarity):
            return [CosineSimilarity(image.features, image2.features)
                    for image2 in self.images]
        elif (method == SimilarityMethod.EuclideanDistance):
            return [EucledeanDistance(image.features, image2.features)
                    for image2 in self.images]

        return None

    def Merge(self,
              identity2: Identity):
        ''' Merge identity2 to self.'''
        # last_frame : Get
        last_frame = self.last_frame

        # Images : Move on disk and update paths
        for index, image in enumerate(identity2.images):
            # Image : Update frame number
            image.frame = last_frame + index + 1

            # Expected path : Create
            expectedName = ReidFileInfo.toPath(identity_number=self.number,
                                               camera_number=image.camera,
                                               frame_number=image.frame,
                                               dataset=self.dataset
                                               )
            expectedPath = f'{image.location}/{expectedName}'
            # Move image to expected path.
            shutil.move(image.path, expectedPath)

            # Image : Update path
            image.path = expectedPath

            # Images : Add moved image
            self.images.append(image)

        # Images : Sort by frame
        self.images = sorted(self.images, key=lambda image: image.frame)

        # Reset cached properties
        self.__dict__.pop('hue', None)
        self.__dict__.pop('brightness', None)
        self.__dict__.pop('saturation', None)
        self.__dict__.pop('imhash', None)
        self.__dict__.pop('features', None)
        self.__dict__.pop('features_binrepr', None)
        self.__dict__.pop('consistency', None)
