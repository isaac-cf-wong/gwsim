from abc import ABC, abstractmethod
from typing import Union
import numpy as np


SAMPLES_DTYPE = np.ndarray
GIVEN_DTYPE = Union[np.ndarray, None]
PROB_DTYPE = np.ndarray

class Distribution(ABC):
    def __init__(self, name: Union[str, None]=None):
        """Distribution.

        Args:
            name (Union[str, None], optional): Name. Defaults to None.
        """
        self.name = name

    def __repr__(self) -> str:
        """Printable representation of the object.

        Returns:
            str: Printable representation of the object.
        """
        return f'Distribution({self.name})'

    @property
    def name(self) -> Union[str, None]:
        """Name of the distribution.

        Returns:
            str: Name of the distribution.
        """
        return self._name

    @name.setter
    def name(self, value: Union[str, None]):
        """Set the

        Args:
            value (str): _description_
        """
        self._name = value

    @abstractmethod
    def logprob(self, samples: SAMPLES_DTYPE, given: GIVEN_DTYPE) -> PROB_DTYPE:
        """Log probability density for continuous distributions,
        of log probability for discrete distributions.

        Args:
            samples (np.ndarray): Samples.
            given (Union[np.ndarray, None]): Conditioned samples.

        Raises:
            NotImplementedError: This method should be implemented in a subclass.

        Returns:
            np.ndarray: Log probability density for continuous distributions,
            or log probability for discrete distributions.
        """
        raise NotImplementedError('This method should be implemented in a subclass.')

    def prob(self, samples: SAMPLES_DTYPE, given: GIVEN_DTYPE) -> PROB_DTYPE:
        return np.exp(self.logprob(samples=samples, given=given))

    @abstractmethod
    def sample(self, number: int=1, given: GIVEN_DTYPE=None) -> SAMPLES_DTYPE:
        """Draw samples from the distribution $p(x | y)$.

        Args:
            number (int, optional): Number of samples. Defaults to 1.
            given (Union[np.ndarray,None], optional): Values y to be conditioned on. Defaults to None.

        Raises:
            NotImplementedError: This method should be implemented in a subclass.

        Returns:
            np.ndarray: An array of samples.
        """
        raise NotImplementedError('This method should be implemented in a subclass.')
