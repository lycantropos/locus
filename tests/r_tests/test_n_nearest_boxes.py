from heapq import nsmallest

from ground.hints import Box, Point
from hypothesis import given

from locus.r import Tree
from tests.hints import ScalarT
from tests.utils import context, to_box_point_squared_distance

from . import strategies


@given(strategies.trees_with_points_and_sizes)
def test_basic(
    tree_with_point_and_n: tuple[Tree[ScalarT], Point[ScalarT], int],
) -> None:
    tree, point, n = tree_with_point_and_n

    result = tree.n_nearest_boxes(n, point)

    assert isinstance(result, (list, type(tree.boxes)))
    assert all(isinstance(element, context.box_cls) for element in result)


@given(strategies.trees_with_points_and_sizes)
def test_properties(
    tree_with_point_and_n: tuple[Tree[ScalarT], Point[ScalarT], int],
) -> None:
    tree, point, n = tree_with_point_and_n

    result = tree.n_nearest_boxes(n, point)

    def to_point_distance(box: Box[ScalarT], /) -> ScalarT:
        return to_box_point_squared_distance(box, point)

    assert 0 < len(result) <= n
    assert all(box in tree.boxes for box in result)
    assert set(nsmallest(n, map(to_point_distance, tree.boxes))) == set(
        map(to_point_distance, result)
    )
