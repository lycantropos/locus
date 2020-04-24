from typing import List

from hypothesis import given

from locus.hints import Point
from locus.kd import (Tree,
                      tree)
from tests.utils import (is_tree_balanced,
                         is_tree_valid,
                         to_balanced_tree_height,
                         to_tree_height)
from . import strategies


@given(strategies.non_empty_points_lists)
def test_basic(points: List[Point]) -> None:
    result = tree(points)

    assert isinstance(result, Tree)


@given(strategies.non_empty_points_lists)
def test_properties(points: List[Point]) -> None:
    result = tree(points)

    assert is_tree_valid(result)
    assert is_tree_balanced(result)
    assert to_tree_height(result) == to_balanced_tree_height(len(points))
