from functools import partial
from heapq import nsmallest

from ground.hints import Point
from hypothesis import given

from locus.kd import Tree
from tests.hints import ScalarT
from tests.utils import context, to_point_squared_distance

from . import strategies


@given(strategies.trees_with_points_and_sizes)
def test_basic(
    tree_with_point_and_n: tuple[Tree[ScalarT], Point[ScalarT], int],
) -> None:
    tree, point, n = tree_with_point_and_n

    result = tree.n_nearest_points(n, point)

    assert isinstance(result, (list, type(tree.points)))
    assert all(isinstance(element, context.point_cls) for element in result)


@given(strategies.trees_with_points_and_sizes)
def test_properties(
    tree_with_point_and_n: tuple[Tree[ScalarT], Point[ScalarT], int],
) -> None:
    tree, point, n = tree_with_point_and_n

    result = tree.n_nearest_points(n, point)

    to_point_distance = partial(to_point_squared_distance, point)
    assert 0 < len(result) <= n
    assert all(point in tree.points for point in result)
    assert set(nsmallest(n, map(to_point_distance, tree.points))) == set(
        map(to_point_distance, result)
    )
