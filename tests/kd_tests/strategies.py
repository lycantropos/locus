from functools import partial
from typing import (List,
                    Optional,
                    Tuple)

from ground.hints import (Box,
                          Scalar)
from hypothesis import strategies

from locus.kd import Tree
from tests.strategies import (points_strategies,
                              scalars_strategies,
                              to_boxes,
                              to_points)
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


def scalars_to_trees_with_balls(scalars: Strategy[Scalar],
                                *,
                                min_tree_size: int = 1,
                                max_tree_size: Optional[int] = None
                                ) -> Strategy[Tuple[Tree, Point, Scalar]]:
    points = to_points(scalars)
    return strategies.tuples(points_to_trees(points,
                                             min_size=min_tree_size,
                                             max_size=max_tree_size),
                             points, scalars.map(abs))


trees_with_balls = (strategies.builds(scalars_to_trees_with_balls,
                                      scalars_strategies)
                    .flatmap(identity))


def scalars_to_trees_with_boxes(scalars: Strategy[Scalar],
                                *,
                                min_tree_size: int = 1,
                                max_tree_size: Optional[int] = None
                                ) -> Strategy[Tuple[Tree, Box]]:
    return strategies.tuples(scalars_to_trees(scalars,
                                              min_tree_size=min_tree_size,
                                              max_tree_size=max_tree_size),
                             to_boxes(scalars))


def scalars_to_trees(scalars: Strategy[Scalar],
                     *,
                     min_tree_size: int = 1,
                     max_tree_size: Optional[int] = None):
    return points_to_trees(to_points(scalars),
                           min_size=min_tree_size,
                           max_size=max_tree_size)


trees_with_boxes = (strategies.builds(scalars_to_trees_with_boxes,
                                      scalars_strategies)
                    .flatmap(identity))
