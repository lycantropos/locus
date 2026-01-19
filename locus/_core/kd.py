from collections.abc import Callable, Sequence
from enum import Enum, auto
from operator import attrgetter
from typing import Final, Generic, Literal, TypeAlias, final

from ground.hints import Point
from reprit.base import generate_repr
from typing_extensions import Self

from locus._core.hints import HasCustomRepr, ScalarT

Item: TypeAlias = tuple[int, Point[ScalarT]]
_Projector: TypeAlias = Callable[[Point[ScalarT]], ScalarT]
_PointsMetric: TypeAlias = Callable[[Point[ScalarT], Point[ScalarT]], ScalarT]


@final
class Nil(Enum):
    _VALUE = auto()


NIL: Final = Nil._VALUE  # noqa: SLF001


class Node(HasCustomRepr, Generic[ScalarT]):
    """Represents node of *kd*-tree."""

    @property
    def index(self, /) -> int:
        return self._index

    @property
    def is_y_axis(self, /) -> bool:
        return self._is_y_axis

    @property
    def item(self, /) -> Item[ScalarT]:
        return self._index, self._point

    @property
    def left(self, /) -> Self | Nil:
        return self._left

    @property
    def point(self, /) -> Point[ScalarT]:
        return self._point

    @property
    def projection(self, /) -> ScalarT:
        return self._projector(self._point)

    @property
    def projector(self, /) -> _Projector[ScalarT]:
        return self._projector

    @property
    def right(self, /) -> Self | Nil:
        return self._right

    def distance_to_point(self, point: Point[ScalarT], /) -> ScalarT:
        return self._metric(self._point, point)

    def distance_to_coordinate(self, coordinate: ScalarT, /) -> ScalarT:
        difference = self.projection - coordinate
        return difference * difference

    _projector: Callable[[Point[ScalarT]], ScalarT]

    __slots__ = (
        '_index',
        '_is_y_axis',
        '_left',
        '_metric',
        '_point',
        '_projector',
        '_right',
    )

    def __init__(
        self,
        index: int,
        point: Point[ScalarT],
        _metric: _PointsMetric[ScalarT],
        /,
        *,
        is_y_axis: bool,
        left: Self | Nil,
        right: Self | Nil,
    ) -> None:
        (
            self._index,
            self._is_y_axis,
            self._left,
            self._metric,
            self._point,
            self._right,
        ) = index, is_y_axis, left, _metric, point, right
        self._projector = attrgetter(_to_point_attribute_name(is_y_axis))

    __repr__ = generate_repr(__init__)


def create_node(
    indices: Sequence[int],
    points: Sequence[Point[ScalarT]],
    /,
    *,
    is_y_axis: bool,
    metric: _PointsMetric[ScalarT],
) -> Node[ScalarT] | Nil:
    if not indices:
        return NIL

    projector: _Projector[ScalarT] = attrgetter(
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
        metric,
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
    )


def _to_point_attribute_name(is_y_axis: bool, /) -> Literal['x', 'y']:  # noqa: FBT001
    return 'y' if is_y_axis else 'x'
