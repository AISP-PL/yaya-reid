'''
    Single ReID identity with all images.

'''
from dataclasses import dataclass, field
from ReID.FeaturesClassifier import FeaturesClassifier
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
        features = self.images[0].features
        vector = MaxPooling1dToSize(features, size=64)
        vector_norm = Normalize(vector)
        return NormalizedVectorToInt(vector_norm)

    @property
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
