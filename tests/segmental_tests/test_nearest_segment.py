from typing import Tuple

from hypothesis import given

from locus.core.segment import distance_to
from locus.hints import (Coordinate,
                         Segment)
from locus.segmental import Tree
from tests.utils import is_segment
from . import strategies


@given(strategies.trees_with_segments)
def test_basic(tree_with_segment: Tuple[Tree, Segment]) -> None:
    tree, segment = tree_with_segment

    result = tree.nearest_segment(segment)

    assert is_segment(result)


@given(strategies.trees_with_segments)
def test_properties(tree_with_segment: Tuple[Tree, Segment]) -> None:
    tree, segment = tree_with_segment

    result = tree.nearest_segment(segment)

    def to_segment_distance(tree_segment: Segment) -> Coordinate:
        return distance_to(tree_segment, segment)

    assert result in tree.segments
    assert (min(map(to_segment_distance, tree.segments))
            == to_segment_distance(result))
