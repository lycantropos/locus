from ground.hints import Point
from hypothesis import given

from locus.core.kd import Item
from locus.kd import Tree
from tests.hints import ScalarT
from tests.utils import (
    all_unique,
    equivalence,
    is_kd_item,
    to_point_squared_distance,
)

from . import strategies


@given(strategies.tree_with_point_strategy)
def test_basic(tree_with_point: tuple[Tree[ScalarT], Point[ScalarT]]) -> None:
    tree, point = tree_with_point

    result = tree.nearest_item(point)

    assert is_kd_item(result)


@given(strategies.tree_strategy)
def test_uniqueness_criteria(tree: Tree[ScalarT]) -> None:
    assert equivalence(
        all(
            tree.nearest_item(point) == (index, point)
            for index, point in enumerate(tree.points)
        ),
        all_unique(tree.points),
    )


@given(strategies.tree_with_point_strategy)
def test_properties(
    tree_with_point: tuple[Tree[ScalarT], Point[ScalarT]],
) -> None:
    tree, point = tree_with_point

    result = tree.nearest_item(point)

    def to_point_distance(item: Item[ScalarT], /) -> ScalarT:
        return to_point_squared_distance(item[1], point)

    items = list(enumerate(tree.points))
    assert result in items
    assert min(map(to_point_distance, items)) == to_point_distance(result)
