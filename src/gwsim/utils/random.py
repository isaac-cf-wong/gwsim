
from numpy.random import default_rng, SeedSequence, Generator


_rng = default_rng()


def get_rng() -> Generator:
    """Get the random number generator.

    Returns:
        Generator: Random number generator.
    """
    return _rng


def seed(seed: int):
    """Set the seed of the random number generator.

    Args:
        seed (int): Seed.
    """
    global _rng
    _rng = default_rng(seed)


def generate_seeds(nseeds: int) -> list:
    """Generate the seeds using the numpy SeedSequence class such that
    the BitGenerators are independent and very probably non-overlapping.

    Args:
        nseeds (int): Number of seeds.

    Returns:
        list: A list of SeedSequence.
    """
    return SeedSequence(_rng.integers(0, 2**63 - 1, size=4)).spawn(nseeds)
