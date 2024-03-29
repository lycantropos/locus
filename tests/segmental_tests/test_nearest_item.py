from typing import Tuple

from ground.hints import (Scalar,
                          Segment)
from hypothesis import given

from locus.core.segmental import Item
from locus.segmental import Tree
from tests.utils import (is_segmental_item,
                         to_segments_distance)
from . import strategies


@given(strategies.trees_with_segments)
def test_basic(tree_with_segment: Tuple[Tree, Segment]) -> None:
    tree, segment = tree_with_segment

    result = tree.nearest_item(segment)

    assert is_segmental_item(result)


@given(strategies.trees_with_segments)
def test_properties(tree_with_segment: Tuple[Tree, Segment]) -> None:
    tree, segment = tree_with_segment

    result = tree.nearest_item(segment)

    def to_segment_distance(item: Item) -> Scalar:
        return to_segments_distance(item[1], segment)

    items = list(enumerate(tree.segments))
    assert result in items
    assert min(map(to_segment_distance, items)) == to_segment_distance(result)
