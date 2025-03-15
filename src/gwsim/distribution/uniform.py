import numpy as np
from .base import (Distribution,
                   SAMPLES_DTYPE,
                   GIVEN_DTYPE,
                   PROB_DTYPE)
from ..utils.random import get_rng


class Uniform(Distribution):
    def __init__(self,
                 low: float=0.,
                 high: float=1.,
                 name: str='uniform'):
        """Uniform distribution.

        Args:
            low (float, optional): Lower bound. Defaults to 0..
            high (float, optional): Upper bound. Defaults to 1..
            name (str, optional): Name of the distribution. Defaults to 'uniform'.
        """
        super().__init__(name=name)
        self._low = low
        self._high = high
        self._lognorm = None

    @property
    def low(self) -> float:
        """Lower bound of the distribution.

        Returns:
            float: Lower bound of the distribution.
        """
        return self._low

    @low.setter
    def low(self, value: float):
        """Set the lower bound.

        Args:
            value (float): Lower bound.

        Raises:
            ValueError: value must be smaller than high.
        """
        if value >= self.high:
            raise ValueError(f'value = {value} must be smaller than high = {self.high}.')
        self._low = value
        self._lognorm = None

    @property
    def high(self) -> float:
        """Upper bound.

        Returns:
            float: Upper bound.
        """
        return self._high

    @high.setter
    def high(self, value: float):
        """Set the upper bound.

        Args:
            value (float): Upper bound.
        """
        if value <= self.low:
            raise ValueError(f'value = {value} must be larger than low = {self.low}.')
        self._high = value
        self._lognorm = None

    @property
    def lognorm(self):
        if self._lognorm is None:
            self._lognorm = -np.log(self.high - self.low)
        return self._lognorm

    def logprob(self, samples: SAMPLES_DTYPE, given: GIVEN_DTYPE=None) -> PROB_DTYPE:
        return np.log(samples) + self.lognorm

    def sample(self, number: int=1, given=None) -> np.ndarray:
        """Drae samples.

        Args:
            number (int, optional): Number of samples. Defaults to 1.
            given (None, optional): A placeholder. Defaults to None.

        Returns:
            np.ndarray: Samples.
        """
        return get_rng().uniform(low=self.low,
                                 high=self.high,
                                 size=(number, 1))
