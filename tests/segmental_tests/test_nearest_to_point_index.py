from typing import Tuple

from ground.hints import (Scalar,
                          Segment)
from hypothesis import given

from locus.segmental import Tree
from tests.utils import to_segments_distance
from . import strategies


@given(strategies.trees_with_segments)
def test_basic(tree_with_segment: Tuple[Tree, Segment]) -> None:
    tree, segment = tree_with_segment

    result = tree.nearest_index(segment)

    assert isinstance(result, int)


@given(strategies.trees_with_segments)
def test_properties(tree_with_segment: Tuple[Tree, Segment]) -> None:
    tree, segment = tree_with_segment

    result = tree.nearest_index(segment)

    def to_segment_distance(index: int) -> Scalar:
        return to_segments_distance(tree.segments[index], segment)

    indices = range(len(tree.segments))
    assert result in indices
    assert (min(map(to_segment_distance, indices))
            == to_segment_distance(result))
