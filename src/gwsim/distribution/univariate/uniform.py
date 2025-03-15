import numpy as np
from .univariate import (UnivariateDistribution,
                         SAMPLES_DTYPE,
                         GIVEN_DTYPE,
                         PROB_DTYPE)
from ...utils.random import get_rng


class Uniform(UnivariateDistribution):
    def __init__(self,
                 x_min: float = 0.,
                 x_max: float = 1.,
                 name: str = 'uniform'):
        """Uniform distribution.

        Args:
            x_min (float, optional): Minimum. Defaults to 0..
            x_max (float, optional): Maximum. Defaults to 1..
            name (str, optional): Name of the distribution. Defaults to 'uniform'.
        """
        super().__init__(x_min=x_min,
                         x_max=x_max,
                         name=name)

    def __repr__(self) -> str:
        """Printable representation of the object.

        Returns:
            str: Printable representation of the object.
        """
        return f'Uniform({self.name})'

    def calculate_log_norm(self) -> float:
        """Calculate the log normalization constant.

        Returns:
            float: Log normalization constant.
        """
        return -np.log(self._x_max - self._x_min)

    def log_prob(self, samples: SAMPLES_DTYPE, given: GIVEN_DTYPE = None) -> PROB_DTYPE:
        """Log probability density.

        Args:
            samples (SAMPLES_DTYPE): Samples
            given (GIVEN_DTYPE, optional): Conditioned samples. Defaults to None.

        Returns:
            PROB_DTYPE: Log probablity density.
        """
        return np.log(samples) + self.log_norm

    def sample(self, number: int = 1, given=None) -> np.ndarray:
        """Drae samples.

        Args:
            number (int, optional): Number of samples. Defaults to 1.
            given (None, optional): A placeholder. Defaults to None.

        Returns:
            np.ndarray: Samples.
        """
        return get_rng().uniform(low=self.x_min,
                                 high=self.x_max,
                                 size=(number, 1))
