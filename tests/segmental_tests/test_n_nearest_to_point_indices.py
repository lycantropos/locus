from heapq import nsmallest

from ground.hints import Point
from hypothesis import given

from locus.segmental import Tree
from tests.hints import ScalarT
from tests.utils import to_segment_point_squared_distance

from . import strategies


@given(strategies.trees_with_points_and_sizes)
def test_basic(
    tree_with_point_and_n: tuple[Tree[ScalarT], Point[ScalarT], int],
) -> None:
    tree, point, n = tree_with_point_and_n

    result = tree.n_nearest_to_point_indices(n, point)

    assert isinstance(result, (list, range))
    assert all(isinstance(element, int) for element in result)


@given(strategies.trees_with_points_and_sizes)
def test_properties(
    tree_with_point_and_n: tuple[Tree[ScalarT], Point[ScalarT], int],
) -> None:
    tree, point, n = tree_with_point_and_n

    result = tree.n_nearest_to_point_indices(n, point)

    def to_point_distance(index: int, /) -> ScalarT:
        return to_segment_point_squared_distance(tree.segments[index], point)

    indices = range(len(tree.segments))
    assert 0 < len(result) <= n
    assert all(index in indices for index in result)
    assert set(nsmallest(n, map(to_point_distance, indices))) == set(
        map(to_point_distance, result)
    )
