"""
Analysis and Design Iteration 1
"""

# Imports
from __future__ import annotation
import abc
import collections
import datetime
from math import isclose, hypot
form typing import (
    cast,
    Any,
    Optional,
    Union,
    Iterator,
    Iterable,
    Counter,
    Callable,
    Protocol
)
import weakref



# Sample Classes
class Sample:
    """
    Abstract superclass for all samples.
    """
    
    def __init__() -> None:
        pass

    def __repr__() -> str:
        pass

class KnownSample(Sample):
    """
    Abstract superclass for testing and training data, the species is set
    externally.
    """

    def __init__() -> None:
        pass

    def __repr__() -> str:
        pass

class TrainingKnownSample(KnownSample):
    """
    Training data.
    """

    pass

class TestingKnownSample(KnownSample):
    """
    Testing data. A classifier can assign a species, which may or may not be correct.
    """

    def __init__() -> None:
        pass

    def matches(self) -> bool:
        pass

    def __repr__() -> str:
        pass

class UnknownSample(Sample):
    """
    A sample provided by a User, not yet classified.
    """

    pass

class ClassifiedSample(Sample):
    """
    Created from a sample provided by a User, and the results of classification.
    """

    def __init__() -> None:
        pass

    def __repr__() -> str:
        pass

# Distance Classes
class Distance:
    """
    Abstract superclass for distance computations.
    """

    def distance() -> float:
        pass

class EuclideanDistance(Distance):
    
    def distance() -> float:
        pass

class MinkowskiDistance(Distance):
    
    def distance() -> float:
        pass

class ChebyshevDistance(Distance):
    
    def distance() -> float:
        pass

class SorensenDistance(Distance):
    
    def distance() -> float:
        pass

# Hyperparameter Class
class Hyperparameter:
    """
    A hyperparameter value and the overall quality of the classification.
    """

    def __init__() -> None:
        pass

    def test() -> None:
        pass

    def classify() -> str:
        pass

# TrainingData Class
class TrainingData:
    """
    A Set of training and testing data with methods to load and test the samples.
    """

    def __init__() -> None:
        pass

    def load() -> None:
        pass

    def test() -> None:
        pass

    def classify() -> ClassifiedSample:
        pass

# Tests
test_Sample = """
"""

test_TrainingKnownSample = """
"""

test_UnknownSample = """
"""

test_ClassifiedSample = """
"""

test_Chebyshev = """
"""

test_Euclidean = """
"""
test_Minkowski = """
"""

test_Sorensen = """
"""

test_Hyperparameter = """
"""

test_TrainingData = """
"""

__test__ = {name: case for name, case in globals().items() if name.startswith("test_")}