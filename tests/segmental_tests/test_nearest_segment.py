from ground.hints import Segment
from hypothesis import given

from locus.segmental import Tree
from tests.hints import ScalarT
from tests.utils import context, to_segment_squared_distance

from . import strategies


@given(strategies.trees_with_segments)
def test_basic(
    tree_with_segment: tuple[Tree[ScalarT], Segment[ScalarT]],
) -> None:
    tree, segment = tree_with_segment

    result = tree.nearest_segment(segment)

    assert isinstance(result, context.segment_cls)


@given(strategies.trees_with_segments)
def test_properties(
    tree_with_segment: tuple[Tree[ScalarT], Segment[ScalarT]],
) -> None:
    tree, segment = tree_with_segment

    result = tree.nearest_segment(segment)

    def to_segment_distance(tree_segment: Segment[ScalarT], /) -> ScalarT:
        return to_segment_squared_distance(tree_segment, segment)

    assert result in tree.segments
    assert min(map(to_segment_distance, tree.segments)) == to_segment_distance(
        result
    )
