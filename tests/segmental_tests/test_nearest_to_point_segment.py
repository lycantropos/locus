from ground.hints import Point, Segment
from hypothesis import given

from locus.segmental import Tree
from tests.hints import ScalarT
from tests.utils import context, to_segment_point_squared_distance

from . import strategies


@given(strategies.trees_with_points)
def test_basic(tree_with_point: tuple[Tree[ScalarT], Point[ScalarT]]) -> None:
    tree, point = tree_with_point

    result = tree.nearest_to_point_segment(point)

    assert isinstance(result, context.segment_cls)


@given(strategies.trees_with_points)
def test_properties(
    tree_with_point: tuple[Tree[ScalarT], Point[ScalarT]],
) -> None:
    tree, point = tree_with_point

    result = tree.nearest_to_point_segment(point)

    def to_point_distance(segment: Segment[ScalarT], /) -> ScalarT:
        return to_segment_point_squared_distance(segment, point)

    assert result in tree.segments
    assert min(map(to_point_distance, tree.segments)) == to_point_distance(
        result
    )
