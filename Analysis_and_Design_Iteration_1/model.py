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
    
    def __init__(
        self,
        sepal_length: float,
        sepal_width: float,
        petal_length: float,
        petal_width: float,
    ) -> None:
        self.sepal_length = sepal_length
        self.sepal_width = sepal_width
        self.petal_length = petal_length
        self.petal_width = petal_width

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"sepal_length = {self.sepal_length}, "
            f"sepal_width = {self.sepal_width}, "
            f"petal_length = {self.petal_length}, "
            f"petal_width = {self.petal_width}, "
            f")"
        )

class KnownSample(Sample):
    """
    Abstract superclass for testing and training data, the species is set
    externally.
    """

    def __init__(
        self,
        species: str,
        sepal_length: float,
        sepal_width: float,
        petal_length: float,
        petal_width: float
    ) -> None:
        super().__init__(
            sepal_length = sepal_length,
            sepal_width = sepal_width,
            petal_length = petal_length,
            petal_width = petal_width
        )
        self.species = species
        
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"sepal_length = {self.sepal_length}, "
            f"sepal_width = {self.sepal_width}, "
            f"petal_length = {self.petal_length}, "
            f"petal_width = {self.petal_width}, "
            f"species = {self.species!r}, "
            f")"
        )

class TrainingKnownSample(KnownSample):
    """
    Training data.
    """

    pass

class TestingKnownSample(KnownSample):
    """
    Testing data. A classifier can assign a species, which may or may not be correct.
    """

    def __init__(
        self,
        /,
        species: str,
        sepal_length: float,
        sepal_width: float,
        petal_length: float,
        petal_width: float,
        classification: Optional[str] = None,
    ) -> None:
        super().__init__(
            species = species,
            sepal_length = sepal_length,
            sepal_width = sepal_width,
            petal_length = petal_length,
            petal_width = petal_width,
        )
        self.classification = classification

    def matches(self) -> bool:
        return self.species = self.classification

    def __repr__() -> str:
        return (
            f"{self.__class__.__name__}("
            f"sepal_length = {self.sepal_length}, "
            f"sepal_width = {self.sepal_width}, "
            f"petal_length = {self.petal_length}, "
            f"petal_width = {self.petal_width}, "
            f"species = {self.self.species!r}, "
            f"classificaiton = {self.classification!r}, "
            f")"
        )

class UnknownSample(Sample):
    """
    A sample provided by a User, not yet classified.
    """

    pass

class ClassifiedSample(Sample):
    """
    Created from a sample provided by a User, and the results of classification.
    """

    def __init__(
        self,
        classification: str,
        sample: UnknownSample
    ) -> None:
        super().__init__(
            sepal_length = sample.sepal_length,
            sepal_width = sample.sepal_width,
            petal_length = sample.petal_length,
            petal_width = sample.petal_width,
        )
        self.classification = classification

    def __repr__() -> str:
        return (
            f"{self.__class__.__name__}("
            f"sepal_width = {self.sepal_width}, "
            f"sepal_length = {self.sepal_length}, "
            f"petal_width = {self.petal_width}, "
            f"petal_length = {self.petal_length}, "
            f"classification = {self.classification!r}, "
            f")"
        )

# Distance Classes
class Distance:
    """
    Abstract superclass for distance computations.
    """

    def distance(
        self,
        s1: Sample,
        s2: Sample
    ) -> float:
        pass

class EuclideanDistance(Distance):
    
    def distance(self, s1: Sample, s2: Sample) -> float:
        return hypot(
            s1.sepal_length - s2.sepal_length,
            s1.sepal_width - s2.sepal_width,
            s1.petal_length - s2.petal_length,
            s1.petal_width - s2.petal_width
        )

class MinkowskiDistance(Distance):
    
    def distance(self, s1: Sample, s2: Sample) -> float:
        return sum(
            [
                abs(s1.sepal_length - s2.sepal_length),
                abs(s1.sepal_width - s2.sepal_width),
                abs(s1.petal_length - s2.petal_length),
                abs(s1.petal_width - s2.petal_width),
            ]
        )

class ChebyshevDistance(Distance):
    
    def distance(self, s1: Sample, s2: Sample) -> float:
        return max(
            [
                abs(s1.sepal_length - s2.sepal_length),
                abs(s1.sepal_width - s2.sepal_width),
                abs(s1.petal_length - s2.petal_length),
                abs(s1.petal_width - s2.petal_width),
            ]
        )

class SorensenDistance(Distance):
    
    def distance(self, s1: Sample, s2: Sample) -> float:
        return sum(
            [
                abs(s1.sepal_length - s2.sepal_length),
                abs(s1.sepal_width - s2.sepal_width),
                abs(s1.petal_length - s2.petal_length),
                abs(s1.petal_width - s2.petal_width),
            ]
        ) / sum(
            [
                abs(s1.sepal_length + s2.sepal_length),
                abs(s1.sepal_width + s2.sepal_width),
                abs(s1.petal_length + s2.petal_length),
                abs(s1.petal_width + s2.petal_width),
            ]
        )

# Hyperparameter Class
class Hyperparameter:
    """
    A hyperparameter value and the overall quality of the classification.
    """

    def __init__(self, k: int, algorithm: "Distance", training: "TrainingData") -> None:
        self.k = k
        self.algorithm = algorithm
        self.data: weakref.ReferenceType["TrainingData"] = weakref.ref(training)
        self.quality: float

    def test(self) -> None:
        # Run the entire test suite.
        training_data: Optional["TrainingData"] = self.data()

        if not training_data:
            raise RuntimeError("Broken Weak Reference")

        pass_count, fail_count = 0, 0

        for sample in training_data.testing:
            sample.classification = self.classify(sample)

            if sample.matches():
                pass_count += 1
            else:
                fail_count += 1

        self.quality = pass_count / (pass_count + fail_count)

    def classify(self, sample: Union[UnknownSample, TestingKnownSample]) -> str:
        # The k-NN algorithm
        training_data = self.data()

        if not training_data:
            raise RuntimeError("No TrainingData object")
        
        distances: list[tuple[float, TrainingKnownSample]] = sorted(
            (self.algorithm.distance(sample, known), known) for known in training_data.training
        )

        k_nearest = (known.species for d, known in distances[: self.k])
        frequency: Counter[str] = collections.Counter(k_nearest)
        best_fit, *others = frequency.most_common()
        species, votes = best_fit
        return species

# TrainingData Class
class TrainingData:
    """
    A Set of training and testing data with methods to load and test the samples.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.uploaded: datetime.datetime
        self.tested: datetime.datetime
        self.training: list[TrainingKnownSample] = []
        self.testing: list[TestingKnownSample] = []
        self.tuning: list[Hyperparameter] = []

    def load(self, raw_data_iterable: Iterable[dict[str, str]]) -> None:
        # Extract TestingKnownSample and TrainingKnownSample from raw data
        for n, row in enumerate(raw_data_iterable):
            if n % 5 == 0:
                test = TestingKnownSample(
                    species = row["species"],
                    sepal_length = float(row["sepal_length"]),
                    sepal_width = float(row["sepal_width"]),
                    petal_length = float(row["petal_length"]),
                    petal_width = float(row["petal_width"]),
                )
                self.testing.append(test)
            else:
                train = TrainingKnownSample(
                    species = float(row["species"]),
                    sepal_length = float(row["sepal_length"]),
                    sepal_width = float(row["sepal_width"]),
                    petal_length = float(row["petal_length"]),
                    petal_width = float(row["petal_width"]),
                )
                self.training.append(train)
        self.uploaded = datetime.datetime.now(tz = datetime.timezone.utc)

    def test(self, parameter: Hyperparameter) -> None:
        # Test this hyperparameter value
        parameter.test()
        self.tuning.append(parameter)
        
        self.tested = datetime.datetime.now(tz = datetime.timezone.utc)

    def classify(self, parameter: Hyperparameter, sample: UnknownSample) -> ClassifiedSample:
        return ClassifiedSample(
            classification = parameter.classify(sample),
            sample = sample
        )

# Tests
test_Sample = """
>>> x = Sample(1, 2, 3, 4)
>>> x
Sample(sepal_length = 1, sepal_width = 2, petal_length = 3, petal_width = 4, )
"""

test_TrainingKnownSample = """
>>> s1 = TrainingKnownSample(
        sepal_length = 5.1, sepal_width = 3.5, petal_length = 1.4, petal_width = 0.2, species = "Iris-setosa"
    )
>>> s1
TrainingKnownSample(sepal_length = 5.1, sepal_width = 3.5, petal_length = 1.4, petal_width = 0.2, species = "Iris-setosa")
"""

test_TestingKnownSample = """
>>> s2 = TestingKnownSample(
        sepal_length = 5.1, sepal_width = 3.5, petal_length = 1.4, petal_width = 0.2, species = "Iris-setosa"
    )
>>> s2
TestingKnownSample(sepal_length = 5.1, sepal_width = 3.5, petal_length = 1.4, petal_width = 0.2, species = "Iris-setosa", classification = None)
>>> s2.classification = "wrong"
TestingKnownSample(sepal_length = 5.1, sepal_width = 3.5, petal_length = 1.4, petal_width = 0.2, species = "Iris-setosa", classification = "wrong", )
"""

test_UnknownSample = """
>>> u = UnknownSample(sepal_length = 5.1, sepal_width = 3.5, petal_length = 1.4, petal_width = 0.2, )
>>> u
UnknownSample(sepal_length = 5.1, sepal_width = 3.5, petal_length = 1.4, petal_width = 0.2, )
"""

test_ClassifiedSample = """
>>> u = UnknownSample(sepal_length = 5.1, sepal_width = 3.5, petal_length = 1.4, petal_width = 0.2, )
>>> c = ClassifiedSample(classification = "Iris-setosa", sample = u)
>>> c
ClassifiedSample(sepal_length = 5.1, sepal_width = 3.5, petal_length = 1.4, petal_width = 0.2, classification = "Iris-setosa", )
"""

test_Chebyshev = """
>>> s1 = TrainingKnownSample(
        sepal_length = 5.1, sepal_width = 3.5, petal_length = 1.4, petal_width = 0.2, species = "Iris-setosa"
    )
>>> u = UnknownSample(***("sepal_length": 7.9, "sepal_width": 3.2, "petal_length": 4.7, "petal_width": 1.4 ))

>>> algorithm = ChebyshevDistance()
>>> isclose(3.3, algorithm.distance(s1, u))
True
"""

test_Euclidean = """
>>> s1 = TrainingKnownSample(
        sepal_length = 5.1, sepal_width = 3.5, petal_length = 1.4, petal_width = 0.2, species = "Iris-setosa"
    )
>>> u = UnknownSample(***("sepal_length": 7.9, "sepal_width": 3.2, "petal_length": 4.7, "petal_width": 1.4 ))

>>> algorithm = EuclideanDistance()
>>> isclose(4.50111097, algorithm.distance(s1, u))
True
"""

test_Minkowski = """
>>> s1 = TrainingKnownSample(
        sepal_length = 5.1, sepal_width = 3.5, petal_length = 1.4, petal_width = 0.2, species = "Iris-setosa"
    )
>>> u = UnknownSample(***("sepal_length": 7.9, "sepal_width": 3.2, "petal_length": 4.7, "petal_width": 1.4 ))

>>> algorithm = MinkowskiDistance()
>>> isclose(7.6, algorithm.distance(s1, u))
True
"""

test_Sorensen = """
>>> s1 = TrainingKnownSample(
        sepal_length = 5.1, sepal_width = 3.5, petal_length = 1.4, petal_width = 0.2, species = "Iris-setosa"
    )
>>> u = UnknownSample(***("sepal_length": 7.9, "sepal_width": 3.2, "petal_length": 4.7, "petal_width": 1.4 ))

>>> algorithm = SorensenDistance()
>>> isclose(3.3, algorithm.distance(s1, u))
"""

test_Hyperparameter = """
>>> td = TrainingData("test")
>>> s2 = TestingKnownSample(
        sepal_length = 5.1, sepal_width = 3.5, petal_length = 1.4, petal_width = 0.2, species = "Iris-setosa"
    )
>>> td.testing = [s2]
>>> t1 = TrainingKnownSample(
        **{"sepal_length": 5,1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2, "species": "Iris-setosa"}
    )
>>> t2 = TrainingKnownSample(
        ***{"sepal_length": 7.9, "sepal_width": 3.2, "petal_length": 4.7, "petal_width": 1.4, "Iris-versicolor"}
    )
>>> td.training = [t1, t2]
>>> h = Hyperparameter(k = 3, algorithm = Chebyshev(), training = td)
>>> u = UnknownSample(sepal_length = 5.1, sepal_width = 3.5, petal_legnth = 1.4, petal_width = 0.2)
>>> h.classify(u)
"Iris-setosa"
>>> h.test()
>>> print(f"data = {td.name!r}, k = {h.k}, quality = {h.quality}")
data = "test", k = 3, quality = 1.0
"""

test_TrainingData = """
>>> td = TrainingData("test")
>>> raw_data = [
        {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2, "species": "Iris-setosa"},
        {"sepal_length": 7.9, "sepal_width": 3.2, "petal_length": 4.7, "petal_width": 1.4, "species": "Iris-versicolor"},
    ]
>>> td.load(raw_data)
>>> h = Hyperparameter(k = 3, algorithm = Chebyshev(), training = td)
>>> len(td.training)
1
>>> len(td.testing)
1
>>> td.test(h)
>>> print(f"data = {td.name!r}, k = {h.k}, quality = {h.quality}")
data = "test", k = 3, quality = 0.0
"""

__test__ = {name: case for name, case in globals().items() if name.startswith("test_")}