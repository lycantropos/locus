from hypothesis import strategies

from locus.core.hilbert import SQUARE_SIZE

hilbert_coordinates = strategies.integers(0, SQUARE_SIZE - 1)
integers = strategies.integers()
non_zero_integers = strategies.integers(max_value=-1) | strategies.integers(1)
