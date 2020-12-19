from typing import List

from ground.hints import Point
from hypothesis import given

from locus.kd import Tree
from tests.utils import (is_kd_tree_balanced,
                         is_kd_tree_valid,
                         to_balanced_tree_height,
                         to_kd_tree_height)
from . import strategies


@given(strategies.non_empty_points_lists)
def test_basic(points: List[Point]) -> None:
    result = Tree(points)

    assert isinstance(result, Tree)


@given(strategies.non_empty_points_lists)
def test_properties(points: List[Point]) -> None:
    result = Tree(points)

    assert is_kd_tree_valid(result)
    assert is_kd_tree_balanced(result)
    assert to_kd_tree_height(result) == to_balanced_tree_height(len(points), 2)
