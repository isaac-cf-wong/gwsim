from typing import Union
import numpy as np
from .base import (Distribution,
                   SAMPLES_DTYPE,
                   GIVEN_DTYPE,
                   PROB_DTYPE)
from .uniform import Uniform
from ..utils.random import get_rng


class PowerLaw(Distribution):
    """
    A class that represents a power law distribution.

    The probability density funcion for a power law distribution is given by:

    .. math::
        p(x | \alpha, x_{\text{min}}, x_{\text{max}}) =
        \frac{\alpha - 1}{x_{\text{min}}} \left( \frac{x}{x_{\text{min}}} \right)^{-\alpha}

    for :math:`\alpha > 0`, and

    .. math::
        p(x | \alpha = 0, x_{\text{min}}, x_{\text{max}}) =
        \frac{1}{x_{\text{min}}} \quad \text{(uniform distribution for } \alpha = 0 \text{)}

    where:
    - :math:`\alpha` is the exponent parameter (typically greater than 1).
    - :math:`x_{\text{min}}` is the minimum value of `x` (the cutoff below which the distribution is zero).
    - :math:`x_{\text{max}}` is the maximum value of `x`.
    - :math:`x` is the variable of interest.
    """
    def __new__(cls,
                alpha: float,
                xmin: float,
                xmax: Union[float, None]=None,
                name: Union[str, None]=None):
        """Return a PowerLaw instance, or a Uniform instance if alpha == 0.

        Args:
            alpha (float): Spectral index.
            xmin (float): Minimum.
            xmax (Union[float, None], optional): Maximum. Defaults to None.
            name (Union[str, None], optional): Name. Defaults to None.

        Raises:
            ValueError: If alpha is 0, xmax must be provided.

        Returns:
            Union[Uniform, PowerLaw]: The appropriate distribution class.
        """
        if alpha == 0:
            if xmax is None:
                raise ValueError(f'xmax = {xmax} must be provided for a uniform distribution.')
            if name is None:
                name = 'uniform'
            return Uniform(low=xmin, high=xmax, name=name)
        return super().__new__(cls)

    def __init__(self,
                 alpha: float,
                 xmin: float,
                 xmax: Union[float, None]=None,
                 name: str='power_law'):
        """Power law distribution.

        Args:
            alpha (float): Spectral index.
            xmin (float): Minimum.
            xmax (Union[float, None], optional): Maximum. If not provided, it is assumed to be infinity. Defaults to None.
            name (str, optional): Name. Defaults to 'power_law'.

        Raises:
            ValueError: xmin must be non-negative.
            ValueError: alpha is negative and xmin cannot be zero.
            ValueError: alpha >= -1 but xmax is not provided.
        """
        super().__init__(name=name)
        if xmin < 0:
            raise ValueError(f'xmin = {xmin} must be non-negative.')
        if alpha < 0.0 and xmin == 0.0:
            raise ValueError(f'alpha = {alpha} is negative and xmin = {xmin} cannot be zero.')
        if alpha >= -1.0 and xmax is None:
            raise ValueError(f'alpha = {alpha} >= -1 but xmax = {xmax} is not provided.')
        self._alpha = alpha
        self._xmin = xmin
        self._xmax = xmax
        self._lognorm = None

    @property
    def alpha(self) -> float:
        """Spectral index.

        Returns:
            float: Spectral index.
        """
        return self._alpha

    @alpha.setter
    def alpha(self, value: float):
        """Set the spectral index.

        Args:
            value (float): Spectral index.
        """
        self._alpha = value
        self._log_norm = None

    @property
    def xmin(self) -> float:
        """Minimum.

        Returns:
            float: Minimum.
        """
        return self._xmin

    @xmin.setter
    def xmin(self, value: float):
        """Set the minimum.

        Args:
            value (float): Minimum.
        """
        self._xmin = value
        self._log_norm = None

    @property
    def xmax(self) -> Union[float, None]:
        """Maximum.

        Returns:
            Union[float, None]: Maximum.
        """
        return self._xmax

    @xmax.setter
    def xmax(self, value: float):
        """Set the maximum.

        Args:
            value (float): Maximum.
        """
        self._xmax = value
        self._log_norm = None

    @property
    def lognorm(self) -> float:
        """Log normalization constant.

        Raises:
            Exception: Uncatched exceptions in __init__.

        Returns:
            float: Log normalization constant.
        """
        if self._lognorm is None:
            if self.alpha != -1 and self.xmax is not None:
                self._lognorm = np.log(1 + self.alpha) - \
                    np.log(self.xmax**(1+self.alpha) - self.xmin**(1+self.alpha))
            elif self.alpha < -1 and self.xmax is None:
                self._lognorm = np.log(-1 - self.alpha) - (1 + self.alpha)*np.log(self.xmin)
            elif self.alpha == -1:
                self._lognorm = -np.log(np.log(self.xmax) - np.log(self.xmin))
            else:
                raise Exception(f'Unexpected error. alpha = {self.alpha}, xmin = {self.xmin}, and xmax = {self.xmax} miss the conditions to compute the log normalization constant.')
        return self._lognorm

    def logprob(self, samples: SAMPLES_DTYPE, given: GIVEN_DTYPE=None) -> PROB_DTYPE:
        """Log probability density.

        Args:
            samples (np.ndarray): Samples.

        Returns:
            np.ndarray: Log probability density.
        """
        return self.alpha * np.log(samples) + self.lognorm

    def sample(self, number: int=1, given: GIVEN_DTYPE=None) -> SAMPLES_DTYPE:
        """Draw samples.

        Args:
            number (int, optional): Number of samples. Defaults to 1.

        Returns:
            np.ndarray: An array of samples.
        """
        if self.alpha != -1 and self.xmax is not None:
            a = self.xmax ** (1+self.alpha)
            b = self.xmin ** (1+self.alpha)
            return (get_rng().random((number, 1))*(a-b) + b) ** (1 / (1+self.alpha))
        elif self.alpha == -1:
            log_xmin = np.log(self.xmin)
            log_xmax = np.log(self.xmax)
            return np.exp(get_rng().random((number, 1)) * (log_xmax - log_xmin) + log_xmin)
        else:
            return (1. - get_rng().random((number, 1)))**(1/(1+self.alpha)) * self.xmin
