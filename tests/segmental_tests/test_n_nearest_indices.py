from heapq import nsmallest

from ground.hints import Segment
from hypothesis import given

from locus.segmental import Tree
from tests.hints import ScalarT
from tests.utils import to_segment_squared_distance

from . import strategies


@given(strategies.tree_with_segment_and_size_strategy)
def test_basic(
    tree_with_segment_and_n: tuple[Tree[ScalarT], Segment[ScalarT], int],
) -> None:
    tree, segment, n = tree_with_segment_and_n

    result = tree.n_nearest_indices(n, segment)

    assert isinstance(result, (list, range))
    assert all(isinstance(element, int) for element in result)


@given(strategies.tree_with_segment_and_size_strategy)
def test_properties(
    tree_with_segment_and_n: tuple[Tree[ScalarT], Segment[ScalarT], int],
) -> None:
    tree, segment, n = tree_with_segment_and_n

    result = tree.n_nearest_indices(n, segment)

    def to_segment_distance(index: int, /) -> ScalarT:
        return to_segment_squared_distance(tree.segments[index], segment)

    indices = range(len(tree.segments))
    assert 0 < len(result) <= n
    assert all(index in indices for index in result)
    assert set(nsmallest(n, map(to_segment_distance, indices))) == set(
        map(to_segment_distance, result)
    )
