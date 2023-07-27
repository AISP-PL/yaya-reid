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

        # Log
        logging.info('(FeaturesClassifier) Created model: %s, path: %s',
                     self.model_name, self.model_path)

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

    def __eq__(self, classifier: FeaturesClassifier) -> bool:
        ''' Classifiers are equal if they have same model name and model path. '''
        return (self.model_name == classifier.model_name) and (self.model_path == classifier.model_path)
