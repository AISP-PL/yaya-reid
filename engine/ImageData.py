'''
   Single Image data with features and visuals.
'''
from dataclasses import dataclass, field
import numpy as np
import os

from helpers.visuals import Visuals


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
