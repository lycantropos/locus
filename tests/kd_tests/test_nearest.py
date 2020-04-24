from functools import partial
from typing import Tuple

from hypothesis import given

from locus.core.utils import squared_distance
from locus.hints import Point
from locus.kd import Tree
from tests.utils import is_point
from . import strategies


@given(strategies.trees_with_points)
def test_basic(tree_with_point: Tuple[Tree, Point]) -> None:
    tree, point = tree_with_point

    result = tree.nearest(point)

    assert is_point(result)


@given(strategies.trees_with_points)
def test_properties(tree_with_point: Tuple[Tree, Point]) -> None:
    tree, point = tree_with_point

    result = tree.nearest(point)

    to_point_distance = partial(squared_distance, point)
    assert result in tree
    assert min(map(to_point_distance, tree)) == to_point_distance(result)
