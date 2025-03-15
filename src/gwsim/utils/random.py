
from random import Random
from numpy.random import default_rng, SeedSequence, Generator


class RandomManager:
    """Singleton manager for random number generation.
    """
    _rng = default_rng()

    @classmethod
    def get_rng(cls) -> Generator:
        """Get the random number generator.

        Returns:
            Generator: Random number generator.
        """
        return cls._rng

    @classmethod
    def seed(cls, seed_: int):
        """Set the seed of the random number generator.

        Args:
            seed_ (int): Seed.
        """
        cls._rng = default_rng(seed_)

    @classmethod
    def generate_seeds(cls, nseeds: int) -> list:
        """Generate the seeds using the numpy SeedSequence class such that
        the BitGenerators are independent and very probably non-overlapping.

        Args:
            nseeds (int): Number of seeds.

        Returns:
            list: A list of SeedSequence.
        """
        return SeedSequence(_rng.integers(0, 2**63 - 1, size=4)).spawn(nseeds)


# Alias for easy  access
get_rng = RandomManager.get_rng
seed = RandomManager.seed
generate_seeds = RandomManager.generate_seeds
