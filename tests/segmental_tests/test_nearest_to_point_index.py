from typing import Tuple

from hypothesis import given

from locus.core.segment import distance_to
from locus.hints import (Coordinate,
                         Segment)
from locus.segmental import Tree
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

    def to_segment_distance(index: int) -> Coordinate:
        return distance_to(tree.segments[index], segment)

    indices = range(len(tree.segments))
    assert result in indices
    assert (min(map(to_segment_distance, indices))
            == to_segment_distance(result))
