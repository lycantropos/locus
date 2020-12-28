from functools import partial
from typing import (List,
                    Optional,
                    Tuple)

from ground.hints import (Box,
                          Coordinate)
from hypothesis import strategies

from locus.kd import Tree
from tests.strategies import (coordinates_strategies,
                              coordinates_to_boxes,
                              coordinates_to_points,
                              points_strategies)
from tests.utils import (Point,
                         Strategy,
                         identity)

non_empty_points_lists = points_strategies.flatmap(partial(strategies.lists,
                                                           min_size=1))
trees = non_empty_points_lists.map(Tree)


def points_to_trees(points: Strategy[Point],
                    *,
                    min_size: int = 1,
                    max_size: Optional[int] = None) -> Strategy[Tree]:
    return (strategies.lists(points,
                             min_size=min_size,
                             max_size=max_size)
            .map(Tree))


def points_to_trees_with_points(points: Strategy[Point]
                                ) -> Strategy[Tuple[Tree, Point]]:
    return strategies.tuples(points_to_trees(points), points)


trees_with_points = (points_strategies
                     .flatmap(points_to_trees_with_points))


def points_to_trees_with_points_and_sizes(points: Strategy[Point]
                                          ) -> Strategy[Tuple[Tree, Point,
                                                              int]]:
    def to_trees_with_points_and_sizes(points_list: List[Point]
                                       ) -> Strategy[Tuple[Tree, Point, int]]:
        return strategies.tuples(strategies.just(Tree(points_list)), points,
                                 strategies.integers(1, len(points_list)))

    return (strategies.lists(points,
                             min_size=1)
            .flatmap(to_trees_with_points_and_sizes))


trees_with_points_and_sizes = (points_strategies
                               .flatmap(points_to_trees_with_points_and_sizes))


def coordinates_to_trees_with_balls(coordinates: Strategy[Coordinate],
                                    *,
                                    min_tree_size: int = 1,
                                    max_tree_size: Optional[int] = None
                                    ) -> Strategy[Tuple[Tree, Point,
                                                        Coordinate]]:
    points = coordinates_to_points(coordinates)
    return strategies.tuples(points_to_trees(points,
                                             min_size=min_tree_size,
                                             max_size=max_tree_size),
                             points, coordinates.map(abs))


trees_with_balls = (strategies.builds(coordinates_to_trees_with_balls,
                                      coordinates_strategies)
                    .flatmap(identity))


def coordinates_to_trees_with_boxes(coordinates: Strategy[Coordinate],
                                    *,
                                    min_tree_size: int = 1,
                                    max_tree_size: Optional[int] = None
                                    ) -> Strategy[Tuple[Tree, Box]]:
    return strategies.tuples(coordinates_to_trees(coordinates,
                                                  min_tree_size=min_tree_size,
                                                  max_tree_size=max_tree_size),
                             coordinates_to_boxes(coordinates))


def coordinates_to_trees(coordinates: Strategy[Coordinate],
                         *,
                         min_tree_size: int = 1,
                         max_tree_size: Optional[int] = None):
    return points_to_trees(coordinates_to_points(coordinates),
                           min_size=min_tree_size,
                           max_size=max_tree_size)


trees_with_boxes = (strategies.builds(coordinates_to_trees_with_boxes,
                                      coordinates_strategies)
                    .flatmap(identity))
