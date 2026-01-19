from ground.hints import Segment
from hypothesis import given

from locus._core.segmental import Item
from locus.segmental import Tree
from tests.hints import ScalarT
from tests.utils import is_segmental_item, to_segment_squared_distance

from . import strategies


@given(strategies.trees_with_segments)
def test_basic(
    tree_with_segment: tuple[Tree[ScalarT], Segment[ScalarT]],
) -> None:
    tree, segment = tree_with_segment

    result = tree.nearest_item(segment)

    assert is_segmental_item(result)


@given(strategies.trees_with_segments)
def test_properties(
    tree_with_segment: tuple[Tree[ScalarT], Segment[ScalarT]],
) -> None:
    tree, segment = tree_with_segment

    result = tree.nearest_item(segment)

    def to_segment_distance(item: Item[ScalarT], /) -> ScalarT:
        return to_segment_squared_distance(item[1], segment)

    items = list(enumerate(tree.segments))
    assert result in items
    assert min(map(to_segment_distance, items)) == to_segment_distance(result)
