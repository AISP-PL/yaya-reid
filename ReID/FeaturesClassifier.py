'''
    "Visual Features" classifier is an dataclass with network model
    used for features extraction from image.
'''
from __future__ import annotations
from dataclasses import dataclass, field
import os
import numpy as np
import logging
from torchreid.reid.utils import FeatureExtractor
from helpers.files import ChangeExtension
from helpers.json import jsonRead
import pickle


@dataclass
class FeaturesClassifier:
    ''' Features classifier dataclass '''
    # Model name
    model_name: str = field(init=True, repr=False, default='resnet18')
    # Model path
    model_path: str = field(init=True, repr=False,
                            default='ReID/Resnet18/model.pth.tar')
    # Features extractor handle
    features_extractor: FeatureExtractor = field(
        init=False, repr=False, default=None)

    def __post_init__(self):
        ''' Post initialization. '''
        # Check : Model path existys
        if (self.model_path is None) or (len(self.model_path) == 0):
            raise ValueError('(FeaturesClassifier) Model path is not set.')

        # Check : Model exists
        if (not os.path.exists(self.model_path)):
            raise ValueError(
                '(FeaturesClassifier) Model path does not exists.')

        # Create features extractor
        self.features_extractor = FeatureExtractor(
            model_name=self.model_name,
            model_path=self.model_path,
            device='cuda'
        )

    def Close(self):
        ''' Close features classifier. '''

    def Extract(self, images: list) -> list:
        '''
            Extract features from np.array images list.
            Parameters:
                images: list - list of np.array images

            Returns:
                list - list of np.array features
        '''
        # Extract features
        features = self.features_extractor(images)
        # Return features
        return list(features.cpu().numpy())

    def LoadCreate(self, imagepath: str) -> np.array:
        ''' Load if exists, otherwise create, save and return features. '''
        # Features  : Load
        features = self.Load(imagepath)
        if (features is not None):
            return features

        # Features  : Create
        features = self.Extract([imagepath])[0]
        # Features  : Save
        self.Save(imagepath, features)

        return features

    def Load(self, imagepath: str) -> np.array:
        ''' Load if exists. '''
        # Features  : Load
        picklepath = ChangeExtension(imagepath, '.features.pickle')
        if (not os.path.exists(picklepath)):
            return None

        # Load features pickle
        with open(picklepath, 'rb') as f:
            features = pickle.load(f)
            return features

    def Save(self, imagepath: str, features: np.array):
        ''' Save . '''
        # Features  : Load
        picklepath = ChangeExtension(imagepath, '.features.pickle')
        # SAve features pickle
        with open(picklepath, 'wb') as f:
            features = pickle.dump(features, f)

    def __eq__(self, classifier: FeaturesClassifier) -> bool:
        ''' Classifiers are equal if they have same model name and model path. '''
        return (self.model_name == classifier.model_name) and (self.model_path == classifier.model_path)
