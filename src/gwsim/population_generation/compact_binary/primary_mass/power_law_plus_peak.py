from ....distribution import Distribution


class PowerLawPlusPeak(Distribution):
    def __init__(self, name='mass_1'):
        super().__init__(name=name)