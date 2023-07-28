'''
    Single ReID identity with all images.

'''
from dataclasses import dataclass, field
from engine.ImageData import ImageData
import numpy as np
from helpers.algebra import CosineSimilarity, Normalize, NormalizedVectorToInt, Pooling1dToSize


@dataclass
class Identity:
    ''' Class representing identity with all images.'''
    # Identity number :
    number: int = field(init=True, default=None)
    # Identity ImageData list
    images: list = field(init=True, default=None)

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

    def ImageSimilarities(self, image: ImageData) -> np.array:
        ''' Calculate given image similarities to all identity images.'''
        # Cosine similarity : For all images
        results = [CosineSimilarity(image.features, image2.features)
                   for image2 in self.images]
        return results