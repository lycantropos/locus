from hypothesis import given

from locus.core.utils import (HILBERT_SQUARE_SIZE,
                              to_hilbert_index)
from tests.utils import (equivalence,
                         to_hilbert_index_complete)
from . import strategies


@given(strategies.hilbert_coordinates, strategies.hilbert_coordinates)
def test_basic(x: int, y: int) -> None:
    result = to_hilbert_index(x, y)

    assert isinstance(result, int)


@given(strategies.hilbert_coordinates, strategies.hilbert_coordinates)
def test_properties(x: int, y: int) -> None:
    result = to_hilbert_index(x, y)

    assert result >= 0
    assert equivalence(result == 0, x == y == 0)
    assert result == to_hilbert_index_complete(HILBERT_SQUARE_SIZE, x, y)


@given(strategies.hilbert_coordinates, strategies.hilbert_coordinates,
       strategies.hilbert_coordinates, strategies.hilbert_coordinates)
def test_bijection(x: int, y: int, other_x: int, other_y: int) -> None:
    result = to_hilbert_index(x, y)

    assert equivalence(result == to_hilbert_index(other_x, other_y),
                       x == other_x and y == other_y)
