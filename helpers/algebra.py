'''
Created on 1 paź 2020

@author: spasz
'''

import math
import numpy as np
from numpy import dot, cross
from numpy.linalg import norm


def GetDistance(p1, p2):
    # Deprecated
    ''' Calculates euclidean distance between points.'''
    return math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)


def GetTranslation(p1, p2):
    ''' Calculates vector translation.'''
    x1, y1 = p1
    x2, y2 = p2
    return (x2-x1, y2-y1)


def GetMiddlePoint(p1, p2):
    ''' Calculates middle point.'''
    x1, y1 = p1
    x2, y2 = p2
    return ((x2+x1)/2, (y2+y1)/2)


def EuclideanDistance(p1, p2):
    ''' Calculates metric.'''
    return math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)


def RadiansToDegree(radian):
    ''' Returns degrees.'''
    return radian*180/math.pi


def ToRadian(degree):
    ''' Returns degrees.'''
    return degree*math.pi/180


def GetHypotenuse(a, b):
    ''' Returns hypotenuse of triangle a,b,c.'''
    return math.sqrt(a**2 + b**2)


def Pooling1d(vector: np.array) -> np.array:
    ''' Pools mean/arithmetic average of vector 2 elements pairs.'''
    reshaped = vector.reshape(-1, 2)
    return np.mean(reshaped, axis=1)


def Pooling1dToSize(vector: np.array, size: int = 64) -> np.array:
    ''' Pools mean/arithmetic average of vector 2 elements pairs.'''
    # Check : Invalid vector
    if vector is None:
        return None

    # Loop : Pooling until size is reached
    while (vector.size > size):
        vector = Pooling1d(vector)

    return vector


def Normalize(vector: np.array) -> np.array:
    ''' Normalizes vector to 0..1 range.'''
    # Check : Invalid vector
    if vector is None:
        return None

    # Check : Zero vector
    if np.all(vector == 0):
        return vector

    # Normalization : 0..1
    return (vector - np.min(vector)) / (np.max(vector) - np.min(vector))


def NormalizedVectorToInt(vec: np.array) -> int:
    ''' Converts normalized vector to integer.'''
    # Round 0..1 values to 0 or 1
    bin_vec = np.round(vec).astype(int)

    # Convert [0, 1, 0 ..] to 010 string and cast as int
    binary_number = int(''.join(map(str, bin_vec)), 2)
    return binary_number


def CosineSimilarity(a: np.array, b: np.array) -> np.array:
    ''' Return cosine similarity for two vectors a.b.'''
    # Normalization : to 0..1
    normalization = norm(a) * norm(b)
    # Check : Norm is 0, then similarity is 0. Probably sth went wrong.
    if (normalization == 0):
        return 0

    return dot(a, b)/normalization
