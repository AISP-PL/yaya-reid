'''
    Test of Algebra module
'''
import pytest
import numpy as np
from helpers.algebra import Pooling1d, Pooling1dToSize, NormalizedVectorToInt


def test_vector_pooling():
    ''' Test of vector pooling.'''

    # Pooling : Simple test
    vector = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    pooled = Pooling1d(vector)
    assert pooled.size == int(vector.size / 2)

    # Pooling : Simple test
    vector = np.array([1, 2, 3, 4,
                       5, 6, 7, 8])
    pooled = Pooling1dToSize(vector, size=2)
    assert pooled.size == 2


def test_vector_toint():
    ''' Test of vector pooling.'''

    vector_normalized = np.array([0.1, 0.2, 0.3, 0.4,
                                  0.5, 0.6, 0.7, 0.8])
    vector_int = NormalizedVectorToInt(vector_normalized)
    assert vector_int == 7


if __name__ == '__main__':
    test_vector_toint()
    test_vector_pooling()
