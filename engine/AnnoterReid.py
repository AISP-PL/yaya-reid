'''
Created on 17 lis 2020

@author: spasz
'''
from __future__ import annotations
from dataclasses import dataclass, field
import os
import re
import time

import numpy as np
from ReID.FeaturesClassifier import FeaturesClassifier
import logging
from tqdm import tqdm
from helpers.algebra import CosineSimilarity

from helpers.files import IsImageFile, DeleteFile, GetNotExistingSha1Filepath, FixPath, GetFilename,\
    GetExtension
from helpers.visuals import Visuals
from engine.Identity import Identity
from engine.ImageData import ImageData


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
class AnnoterReid:
    ''' Class reading all images and annotations.'''
    # Path to directory with images
    dirpath: str = field(init=True, default=None)
    # Arguments : Namespace from argparse
    args: object = field(init=True, default=None)
    # ReID features classifier
    features_classifier: FeaturesClassifier = field(init=True, default=None)
    # Found identities list
    identities: dict = field(init=False, default_factory=list)

    # Matrix of Identity.features x Identity.features similarities
    similarity_matrix: np.ndarray = field(init=False, default=None)

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

    @property
    def consistency_avg(self) -> float:
        ''' Return average consistency.'''
        # Consistency : Get all consistency
        consistency = [
            self.identities[identityID].consistency for identityID in self.identities]
        # Return average
        return sum(consistency) / len(consistency)

    @property
    def similarity_avg(self) -> float:
        ''' Return average similarity.'''
        return np.mean(self.similarity_matrix)

    @property
    def similarity_min(self) -> float:
        ''' Return minimum similarity.'''
        return np.min(self.similarity_matrix)

    @property
    def similarity_max(self) -> float:
        ''' Return maximum similarity.'''
        return np.max(self.similarity_matrix)

    @property
    def separation_avg(self) -> float:
        ''' Return average separation.'''
        return 1 - self.similarity_avg

    @property
    def separation_min(self) -> float:
        ''' Return minimum separation.'''
        return 1 - self.similarity_max

    @property
    def separation_max(self) -> float:
        ''' Return maximum separation.'''
        return 1 - self.similarity_min

    @staticmethod
    def ImagenameToReidInfo(imagename: str) -> int:
        ''' Convert imagename to identity number.'''
        # Filename : Get filename
        filename = GetFilename(imagename)
        # Identity : Get identity number
        identity = int(filename.split('_')[0])
        return identity

    def Similarities(self, identity: Identity) -> dict:
        ''' Return identity (to other identities) similarities as dict.'''
        # Row index of identity in matrix
        index = self.indentities_ids.index(identity.number)

        # Similarities dict : Create from matrix
        similarities = {}
        for index, value in enumerate(self.similarity_matrix[index, :]):
            similarities[self.indentities_ids[index]] = value

        return similarities

    def SeparationAvg(self, identity: Identity) -> float:
        ''' Return separation of identity.'''
        # Index position in matrix
        index = self.indentities_ids.index(identity.number)
        # Similarity : Get similarity
        similarity = np.mean(self.similarity_matrix[index, :])
        return 1 - similarity

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

        # ProgressBar : Create
        progress = tqdm(total=len(images),
                        desc=f'Loading reid images', unit='images')

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
            features = self.features_classifier.LoadCreate(
                imagepath=imagepath, force=self.args.force)

            # Identity : Create identity if not exists
            if (reidInfo.identity not in self.identities):
                self.identities[reidInfo.identity] = Identity(number=reidInfo.identity,
                                                              images=[])

            # Identity : Append image
            self.identities[reidInfo.identity].images.append(ImageData(path=imagepath,
                                                                       camera=reidInfo.camera,
                                                                       frame=reidInfo.frame,
                                                                       visuals=visuals,
                                                                       features=features))

            # Progress : Update
            progress.update(1)

        # Progress : Close
        progress.close()

        # Similarity Matrix : Create
        self.SimilarityMatrixCreate()

    def SimilarityMatrixCreate(self):
        ''' Create similarity matrix.'''
        # Create matrix
        self.similarity_matrix = np.zeros(
            (self.identities_count, self.identities_count))

        # Progress : Create
        progress = tqdm(total=self.identities_count,
                        desc=f'Creating similarity matrix',
                        unit='identities')

        # Processing all identities
        for index1, identityID1 in enumerate(self.identities):
            # Processing all identities
            for index2, identityID2 in enumerate(self.identities):
                # Similarity : Calculate similarity
                similarity = CosineSimilarity(self.identities[identityID1].features,
                                              self.identities[identityID2].features)

                # Matrix : Store
                self.similarity_matrix[index1, index2] = similarity

            # Progress : Update
            progress.update(1)
