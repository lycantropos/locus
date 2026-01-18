from ground.hints import Point
from hypothesis import given

from locus.segmental import Tree
from tests.hints import ScalarT
from tests.utils import to_segment_point_squared_distance

from . import strategies


@given(strategies.tree_with_point_strategy)
def test_basic(tree_with_point: tuple[Tree[ScalarT], Point[ScalarT]]) -> None:
    tree, point = tree_with_point

    result = tree.nearest_to_point_index(point)

    assert isinstance(result, int)


@given(strategies.tree_with_point_strategy)
def test_properties(
    tree_with_point: tuple[Tree[ScalarT], Point[ScalarT]],
) -> None:
    tree, point = tree_with_point

    result = tree.nearest_to_point_index(point)

    def to_point_distance(index: int, /) -> ScalarT:
        return to_segment_point_squared_distance(tree.segments[index], point)

    indices = range(len(tree.segments))
    assert result in indices
    assert min(map(to_point_distance, indices)) == to_point_distance(result)
