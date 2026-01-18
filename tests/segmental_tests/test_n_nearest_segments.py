from heapq import nsmallest

from ground.hints import Segment
from hypothesis import given

from locus.segmental import Tree
from tests.hints import ScalarT
from tests.utils import context, to_segment_squared_distance

from . import strategies


@given(strategies.trees_with_segments_and_sizes)
def test_basic(
    tree_with_segment_and_n: tuple[Tree[ScalarT], Segment[ScalarT], int],
) -> None:
    tree, segment, n = tree_with_segment_and_n

    result = tree.n_nearest_segments(n, segment)

    assert isinstance(result, (list, type(tree.segments)))
    assert all(isinstance(element, context.segment_cls) for element in result)


@given(strategies.trees_with_segments_and_sizes)
def test_properties(
    tree_with_segment_and_n: tuple[Tree[ScalarT], Segment[ScalarT], int],
) -> None:
    tree, segment, n = tree_with_segment_and_n

    result = tree.n_nearest_segments(n, segment)

    def to_segment_distance(tree_segment: Segment[ScalarT], /) -> ScalarT:
        return to_segment_squared_distance(tree_segment, segment)

    assert 0 < len(result) <= n
    assert all(segment in tree.segments for segment in result)
    assert set(nsmallest(n, map(to_segment_distance, tree.segments))) == set(
        map(to_segment_distance, result)
    )
