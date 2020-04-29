from hypothesis import strategies

from locus.core.hilbert import SQUARE_SIZE

hilbert_coordinates = strategies.integers(0, SQUARE_SIZE - 1)
