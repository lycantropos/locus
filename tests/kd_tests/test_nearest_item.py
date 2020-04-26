from typing import Tuple

from hypothesis import given

from locus.core.utils import planar_distance
from locus.hints import (Coordinate,
                         Point)
from locus.kd import (Item,
                      Tree)
from tests.utils import (all_unique,
                         equivalence,
                         is_item)
from . import strategies


@given(strategies.trees_with_points)
def test_basic(tree_with_point: Tuple[Tree, Point]) -> None:
    tree, point = tree_with_point

    result = tree.nearest_item(point)

    assert is_item(result)


@given(strategies.trees)
def test_uniqueness_criteria(tree: Tree) -> None:
    assert equivalence(all(tree.nearest_item(point) == (index, point)
                           for index, point in enumerate(tree.points)),
                       all_unique(tree.points))


@given(strategies.trees_with_points)
def test_properties(tree_with_point: Tuple[Tree, Point]) -> None:
    tree, point = tree_with_point

    result = tree.nearest_item(point)

    def to_point_distance(item: Item) -> Coordinate:
        return planar_distance(item[1], point)

    assert result in enumerate(tree.points)
    assert (min(map(to_point_distance, enumerate(tree.points)))
            == to_point_distance(result))