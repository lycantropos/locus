from ground.hints import Segment
from hypothesis import given

from locus.segmental import Tree
from tests.hints import ScalarT
from tests.utils import to_segment_squared_distance

from . import strategies


@given(strategies.trees_with_segments)
def test_basic(
    tree_with_segment: tuple[Tree[ScalarT], Segment[ScalarT]],
) -> None:
    tree, segment = tree_with_segment

    result = tree.nearest_index(segment)

    assert isinstance(result, int)


@given(strategies.trees_with_segments)
def test_properties(
    tree_with_segment: tuple[Tree[ScalarT], Segment[ScalarT]],
) -> None:
    tree, segment = tree_with_segment

    result = tree.nearest_index(segment)

    def to_segment_distance(index: int, /) -> ScalarT:
        return to_segment_squared_distance(tree.segments[index], segment)

    indices = range(len(tree.segments))
    assert result in indices
    assert min(map(to_segment_distance, indices)) == to_segment_distance(
        result
    )
