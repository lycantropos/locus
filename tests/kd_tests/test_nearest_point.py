from functools import partial

from ground.hints import Point
from hypothesis import given

from locus.kd import Tree
from tests.hints import ScalarT
from tests.utils import (
    all_unique,
    context,
    equivalence,
    to_point_squared_distance,
)

from . import strategies


@given(strategies.trees_with_points)
def test_basic(tree_with_point: tuple[Tree[ScalarT], Point[ScalarT]]) -> None:
    tree, point = tree_with_point

    result = tree.nearest_point(point)

    assert isinstance(result, context.point_cls)


@given(strategies.trees)
def test_fixed_points(tree: Tree[ScalarT]) -> None:
    assert all(tree.nearest_point(point) == point for point in tree.points)


@given(strategies.trees)
def test_uniqueness_criteria(tree: Tree[ScalarT]) -> None:
    assert equivalence(
        all(tree.nearest_point(point) is point for point in tree.points),
        all_unique(tree.points),
    )


@given(strategies.trees_with_points)
def test_properties(
    tree_with_point: tuple[Tree[ScalarT], Point[ScalarT]],
) -> None:
    tree, point = tree_with_point

    result = tree.nearest_point(point)

    to_point_distance = partial(to_point_squared_distance, point)
    assert result in tree.points
    assert min(map(to_point_distance, tree.points)) == to_point_distance(
        result
    )
