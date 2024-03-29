from typing import Tuple

from ground.hints import (Box,
                          Point,
                          Scalar)
from hypothesis import given

from locus.r import Tree
from tests.utils import (is_box,
                         to_box_point_distance)
from . import strategies


@given(strategies.trees_with_points)
def test_basic(tree_with_point: Tuple[Tree, Point]) -> None:
    tree, point = tree_with_point

    result = tree.nearest_box(point)

    assert is_box(result)


@given(strategies.trees_with_points)
def test_properties(tree_with_point: Tuple[Tree, Point]) -> None:
    tree, point = tree_with_point

    result = tree.nearest_box(point)

    def to_point_distance(box: Box) -> Scalar:
        return to_box_point_distance(box, point)

    assert result in tree.boxes
    assert (min(map(to_point_distance, tree.boxes))
            == to_point_distance(result))
