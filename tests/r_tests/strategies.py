from functools import partial
from typing import (List,
                    Optional,
                    Tuple)

from ground.hints import (Box,
                          Scalar)
from hypothesis import strategies

from locus.core.hilbert import MAX_COORDINATE
from locus.r import Tree
from tests.strategies import (scalars_strategies,
                              to_boxes,
                              to_points)
from tests.utils import (Point,
                         Strategy)

MIN_BOXES_SIZE = 2
max_children_counts = (strategies.sampled_from([2 ** power
                                                for power in range(1, 10)])
                       | strategies.integers(2, MAX_COORDINATE))
boxes_strategies = scalars_strategies.map(to_boxes)
boxes_lists = (boxes_strategies
               .flatmap(partial(strategies.lists,
                                min_size=MIN_BOXES_SIZE)))
trees = strategies.builds(Tree,
                          boxes_lists,
                          max_children=max_children_counts)


def scalars_to_trees_with_boxes(scalars: Strategy[Scalar],
                                *,
                                min_size: int = MIN_BOXES_SIZE,
                                max_size: Optional[int] = None
                                ) -> Strategy[Tuple[Tree, Box]]:
    boxes = to_boxes(scalars)
    return strategies.tuples(
            strategies.builds(Tree,
                              strategies.lists(boxes,
                                               min_size=min_size,
                                               max_size=max_size),
                              max_children=max_children_counts),
            boxes)


trees_with_boxes = scalars_strategies.flatmap(scalars_to_trees_with_boxes)


def scalars_to_trees_with_points(scalars: Strategy[Scalar],
                                 *,
                                 min_size: int = MIN_BOXES_SIZE,
                                 max_size: Optional[int] = None
                                 ) -> Strategy[Tuple[Tree, Point]]:
    return (strategies.tuples(
            strategies.builds(Tree,
                              strategies.lists(to_boxes(scalars),
                                               min_size=min_size,
                                               max_size=max_size),
                              max_children=max_children_counts),
            to_points(scalars)))


trees_with_points = scalars_strategies.flatmap(scalars_to_trees_with_points)


def scalars_to_trees_with_points_and_sizes(
        scalars: Strategy[Scalar],
        *,
        min_size: int = MIN_BOXES_SIZE,
        max_size: Optional[int] = None) -> Strategy[Tuple[Tree, Point, int]]:
    def boxes_with_point_to_trees_with_points_and_sizes(
            boxes_with_point: Tuple[List[Box], Point]
    ) -> Strategy[Tuple[Tree, Point, int]]:
        boxes, point = boxes_with_point
        return strategies.tuples(
                strategies.builds(Tree,
                                  strategies.just(boxes),
                                  max_children=max_children_counts),
                strategies.just(point),
                strategies.integers(1, len(boxes)))

    return (strategies.tuples(
            strategies.lists(to_boxes(scalars),
                             min_size=min_size,
                             max_size=max_size),
            to_points(scalars))
            .flatmap(boxes_with_point_to_trees_with_points_and_sizes))


trees_with_points_and_sizes = (
    scalars_strategies.flatmap(scalars_to_trees_with_points_and_sizes))
