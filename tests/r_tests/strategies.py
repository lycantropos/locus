from functools import partial
from typing import (List,
                    Optional,
                    Tuple)

from ground.hints import (Box,
                          Coordinate)
from hypothesis import strategies

from locus.core.hilbert import MAX_COORDINATE
from locus.r import Tree
from tests.strategies import (coordinates_strategies,
                              coordinates_to_boxes,
                              coordinates_to_points)
from tests.utils import (Point,
                         Strategy)

MIN_BOXES_SIZE = 2
max_children_counts = (strategies.sampled_from([2 ** power
                                                for power in range(1, 10)])
                       | strategies.integers(2, MAX_COORDINATE))
boxes_strategies = coordinates_strategies.map(coordinates_to_boxes)
boxes_lists = (boxes_strategies
               .flatmap(partial(strategies.lists,
                                min_size=MIN_BOXES_SIZE)))
trees = strategies.builds(Tree,
                          boxes_lists,
                          max_children=max_children_counts)


def coordinates_to_trees_with_boxes(coordinates: Strategy[Coordinate],
                                    *,
                                    min_size: int = MIN_BOXES_SIZE,
                                    max_size: Optional[int] = None
                                    ) -> Strategy[Tuple[Tree, Box]]:
    boxes = coordinates_to_boxes(coordinates)
    return strategies.tuples(
            strategies.builds(Tree,
                              strategies.lists(boxes,
                                               min_size=min_size,
                                               max_size=max_size),
                              max_children=max_children_counts),
            boxes)


trees_with_boxes = (coordinates_strategies
                    .flatmap(coordinates_to_trees_with_boxes))


def coordinates_to_trees_with_points(coordinates: Strategy[Coordinate],
                                     *,
                                     min_size: int = MIN_BOXES_SIZE,
                                     max_size: Optional[int] = None
                                     ) -> Strategy[Tuple[Tree, Point]]:
    boxes = coordinates_to_boxes(coordinates)
    return (strategies.tuples(
            strategies.builds(Tree,
                              strategies.lists(boxes,
                                               min_size=min_size,
                                               max_size=max_size),
                              max_children=max_children_counts),
            coordinates_to_points(coordinates)))


trees_with_points = (coordinates_strategies
                     .flatmap(coordinates_to_trees_with_points))


def coordinates_to_trees_with_points_and_sizes(
        coordinates: Strategy[Coordinate],
        *,
        min_size: int = MIN_BOXES_SIZE,
        max_size: Optional[int] = None) -> Strategy[Tuple[Tree, Point, int]]:
    def to_trees_with_points_and_sizes(boxes_with_point
                                       : Tuple[List[Box], Point]
                                       ) -> Strategy[Tuple[Tree, Point, int]]:
        boxes, point = boxes_with_point
        return strategies.tuples(
                strategies.builds(Tree,
                                  strategies.just(boxes),
                                  max_children=max_children_counts),
                strategies.just(point),
                strategies.integers(1, len(boxes)))

    return (strategies.tuples(
            strategies.lists(coordinates_to_boxes(coordinates),
                             min_size=min_size,
                             max_size=max_size),
            coordinates_to_points(coordinates))
            .flatmap(to_trees_with_points_and_sizes))


trees_with_points_and_sizes = (
    coordinates_strategies.flatmap(coordinates_to_trees_with_points_and_sizes))
