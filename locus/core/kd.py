from collections.abc import Callable, Sequence
from enum import Enum, auto
from operator import attrgetter
from typing import Final, Generic, TypeAlias, final

from ground.hints import Point
from reprit.base import generate_repr
from typing_extensions import Self

from locus.core.hints import HasCustomRepr, ScalarT

Item: TypeAlias = tuple[int, Point[ScalarT]]


@final
class Nil(Enum):
    _VALUE = auto()


NIL: Final = Nil._VALUE  # noqa: SLF001


class Node(HasCustomRepr, Generic[ScalarT]):
    """Represents node of *kd*-tree."""

    projector: Callable[[Point[ScalarT]], ScalarT]

    __slots__ = (
        'index',
        'is_y_axis',
        'left',
        'metric',
        'point',
        'projector',
        'right',
    )

    def __init__(
        self,
        index: int,
        point: Point[ScalarT],
        /,
        *,
        is_y_axis: bool,
        left: Self | Nil,
        right: Self | Nil,
        metric: Callable[[Point[ScalarT], Point[ScalarT]], ScalarT],
    ) -> None:
        self.index, self.point = index, point
        self.is_y_axis, self.projector = (
            is_y_axis,
            attrgetter(_to_point_attribute_name(is_y_axis)),
        )
        self.left, self.right = left, right
        self.metric = metric

    __repr__ = generate_repr(__init__)

    @property
    def item(self, /) -> Item[ScalarT]:
        return self.index, self.point

    @property
    def projection(self, /) -> ScalarT:
        return self.projector(self.point)

    def distance_to_point(self, point: Point[ScalarT], /) -> ScalarT:
        return self.metric(self.point, point)

    def distance_to_coordinate(self, coordinate: ScalarT, /) -> ScalarT:
        difference = self.projection - coordinate
        return difference * difference


def _to_point_attribute_name(is_y_axis: bool, /) -> str:  # noqa: FBT001
    return 'y' if is_y_axis else 'x'


def create_node(
    indices: Sequence[int],
    points: Sequence[Point[ScalarT]],
    /,
    *,
    is_y_axis: bool,
    metric: Callable[[Point[ScalarT], Point[ScalarT]], ScalarT],
) -> Node[ScalarT] | Nil:
    if not indices:
        return NIL

    projector: Callable[[Point[ScalarT]], ScalarT] = attrgetter(
        _to_point_attribute_name(is_y_axis)
    )

    def index_sorting_key(index: int, /) -> ScalarT:
        return projector(points[index])

    indices = sorted(indices, key=index_sorting_key)
    middle_index = (len(indices) - 1) // 2
    pivot_index = indices[middle_index]
    next_is_y_axis = not is_y_axis
    return Node(
        pivot_index,
        points[pivot_index],
        is_y_axis=is_y_axis,
        left=create_node(
            indices[:middle_index],
            points,
            is_y_axis=next_is_y_axis,
            metric=metric,
        ),
        right=create_node(
            indices[middle_index + 1 :],
            points,
            is_y_axis=next_is_y_axis,
            metric=metric,
        ),
        metric=metric,
    )
