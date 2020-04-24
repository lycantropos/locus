from functools import partial
from typing import (List,
                    Optional,
                    Tuple)

from hypothesis import strategies

from locus.hints import (Coordinate,
                         Interval,
                         Point)
from locus.kd import Tree
from tests.strategies import (axes,
                              coordinates_strategies,
                              coordinates_to_points,
                              points_strategies)
from tests.utils import (Strategy,
                         identity,
                         to_homogeneous_tuples)

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
                                    dimension: int,
                                    min_tree_size: int = 1,
                                    max_tree_size: Optional[int] = None
                                    ) -> Strategy[Tuple[Tree, Point,
                                                        Coordinate]]:
    points = coordinates_to_points(coordinates,
                                   dimension=dimension)
    return strategies.tuples(points_to_trees(points,
                                             min_size=min_tree_size,
                                             max_size=max_tree_size),
                             points, coordinates.map(abs))


trees_with_balls = (strategies.builds(coordinates_to_trees_with_balls,
                                      coordinates_strategies,
                                      dimension=axes)
                    .flatmap(identity))


def coordinates_to_trees_with_intervals(coordinates: Strategy[Coordinate],
                                        *,
                                        dimension: int,
                                        min_tree_size: int = 1,
                                        max_tree_size: Optional[int] = None
                                        ) -> Strategy[Tuple[Tree, Interval]]:
    return strategies.tuples(coordinates_to_trees(coordinates,
                                                  dimension=dimension,
                                                  min_tree_size=min_tree_size,
                                                  max_tree_size=max_tree_size),
                             coordinates_to_intervals(coordinates,
                                                      dimension=dimension))


def coordinates_to_trees(coordinates: Strategy[Coordinate],
                         *,
                         dimension: int,
                         min_tree_size: int = 1,
                         max_tree_size: Optional[int] = None):
    return points_to_trees(coordinates_to_points(coordinates,
                                                 dimension=dimension),
                           min_size=min_tree_size,
                           max_size=max_tree_size)


def coordinates_to_intervals(coordinates: Strategy[Coordinate],
                             *,
                             dimension: int
                             ) -> Strategy[Interval]:
    return to_homogeneous_tuples(strategies.lists(coordinates,
                                                  min_size=2,
                                                  max_size=2,
                                                  unique=True)
                                 .map(sorted)
                                 .map(tuple),
                                 size=dimension)


trees_with_intervals = (strategies.builds(coordinates_to_trees_with_intervals,
                                          coordinates_strategies,
                                          dimension=axes)
                        .flatmap(identity))
