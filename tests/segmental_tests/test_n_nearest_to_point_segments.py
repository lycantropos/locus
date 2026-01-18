from heapq import nsmallest

from ground.hints import Point, Segment
from hypothesis import given

from locus.segmental import Tree
from tests.hints import ScalarT
from tests.utils import context, to_segment_point_squared_distance

from . import strategies


@given(strategies.trees_with_points_and_sizes)
def test_basic(
    tree_with_point_and_n: tuple[Tree[ScalarT], Point[ScalarT], int],
) -> None:
    tree, point, n = tree_with_point_and_n

    result = tree.n_nearest_to_point_segments(n, point)

    assert isinstance(result, (list, type(tree.segments)))
    assert all(isinstance(element, context.segment_cls) for element in result)


@given(strategies.trees_with_points_and_sizes)
def test_properties(
    tree_with_point_and_n: tuple[Tree[ScalarT], Point[ScalarT], int],
) -> None:
    tree, point, n = tree_with_point_and_n

    result = tree.n_nearest_to_point_segments(n, point)

    def to_point_distance(segment: Segment[ScalarT], /) -> ScalarT:
        return to_segment_point_squared_distance(segment, point)

    assert 0 < len(result) <= n
    assert all(segment in tree.segments for segment in result)
    assert set(nsmallest(n, map(to_point_distance, tree.segments))) == set(
        map(to_point_distance, result)
    )
