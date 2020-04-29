from hypothesis import strategies

from locus.core.utils import HILBERT_SQUARE_SIZE

hilbert_coordinates = strategies.integers(0, HILBERT_SQUARE_SIZE - 1)
