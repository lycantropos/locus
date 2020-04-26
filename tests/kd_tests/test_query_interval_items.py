from typing import Tuple

from hypothesis import given

from locus.core.utils import point_in_interval
from locus.hints import Interval
from locus.kd import Tree
from tests.utils import is_item
from . import strategies


@given(strategies.trees_with_intervals)
def test_basic(tree_with_interval: Tuple[Tree, Interval]) -> None:
    tree, interval = tree_with_interval

    result = tree.query_interval_items(interval)

    assert isinstance(result, list)
    assert all(is_item(element) for element in result)


@given(strategies.trees_with_intervals)
def test_properties(tree_with_interval: Tuple[Tree, Interval]) -> None:
    tree, interval = tree_with_interval

    result = tree.query_interval_items(interval)

    items = list(enumerate(tree.points))
    assert all(item in items for item in result)
    assert all(point_in_interval(point, interval)
               for _, point in result)
    assert all(item in result
               for item in items
               if point_in_interval(item[1], interval))