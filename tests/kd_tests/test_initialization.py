from collections.abc import Sequence

from ground.hints import Point
from hypothesis import given

from locus.kd import Tree
from tests.hints import ScalarT
from tests.utils import (
    context,
    is_kd_tree_balanced,
    is_kd_tree_valid,
    to_balanced_tree_height,
    to_kd_tree_height,
)

from .strategies import non_empty_point_sequence_strategy


@given(non_empty_point_sequence_strategy)
def test_basic(points: list[Point[ScalarT]]) -> None:
    result = Tree(points, context=context)

    assert isinstance(result, Tree)


@given(non_empty_point_sequence_strategy)
def test_properties(points: Sequence[Point[ScalarT]]) -> None:
    result = Tree(points, context=context)

    assert is_kd_tree_valid(result)
    assert is_kd_tree_balanced(result)
    assert to_kd_tree_height(result) == to_balanced_tree_height(len(points), 2)
