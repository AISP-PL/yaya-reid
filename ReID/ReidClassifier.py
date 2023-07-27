'''
    ReID classifier is an single instance of currently
    used "FeaturesClassifier" as a singleton.
'''
from __future__ import annotations
from dataclasses import dataclass, field
import os
from ReID.FeaturesClassifier import FeaturesClassifier

# Model : Currently used model
__model = None


def CreateReidClassifier(model_name: str = 'resnet50',
                         model_path: str = 'ReID/Resnet50/model.pth.tar') -> FeaturesClassifier:
    ''' Create FeaturesClassifier. '''
    global __model

    # Check : Current model has same name and path
    if (__model is not None) and (__model.model_name == model_name) and (__model.model_path == model_path):
        return __model

    # Check : Model exists, remove it
    if (__model is not None):
        __model.Close()
        __model = None

    # Model : Create new model
    __model = FeaturesClassifier(
        model_name=model_name,
        model_path=model_path
    )

    # Return model
    return __model


def GetReidClassifier() -> FeaturesClassifier:
    ''' Get FeaturesClassifier. '''
    global __model

    # Check : Model is not created
    if (__model is None):
        raise ValueError('(ReidClassifier) Classifier not created!')

    # Return model
    return __model


def ModelsList(modelsPath: str = 'ReID') -> list:
    '''
        Get list of available models from ReID/ directory.
        Each subdirectory is a model name (e.g. Resnet50, Resnet18).
        Each *.tar file inside subdirectory is a model file (e.g. model.pth.tar).
    '''
    # Models : list of [(model_name, [model_path)]
    models = []

    # Subdirectories : Get all subdirectories in ReID/
    subdirs = [f.path for f in os.scandir(modelsPath) if f.is_dir()]
    # Models : Get all files in subdirectories
    for subdir in subdirs:
        subdirmodels = [(os.path.basename(subdir), f.path) for f in os.scandir(subdir)
                        if f.is_file() and f.name.endswith('.pth.tar')]
        models.extend(subdirmodels)

    # Return models
    return models


def ModelsPrint(models: list):
    ''' Print models dict. '''
    for index, model in enumerate(models):
        model_name, model_path = model
        print(f'{index} : {model_name} / {model_path}.')


def ModelCreate(models: list, modelNumber: int):
    ''' Create ReID classifier based on models dict and number. '''
    # Check : Model number is valid
    if (modelNumber < 0) or (modelNumber >= len(models)):
        raise ValueError(
            f'(ReidClassifier) Invalid model number {modelNumber}!')

    # Create model
    model_name, model_path = models[modelNumber]
    CreateReidClassifier(model_name=model_name, model_path=model_path)
