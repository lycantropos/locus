from typing import (List,
                    Tuple)

from hypothesis import strategies

from locus.hints import Point
from locus.kd import (Tree,
                      tree)
from tests.strategies import points_strategies
from tests.utils import Strategy


def points_to_trees_with_points(points: Strategy[Point]
                                ) -> Strategy[Tuple[Tree, Point]]:
    return strategies.tuples(strategies.lists(points,
                                              min_size=1)
                             .map(tree),
                             points)


trees_with_points = (points_strategies
                     .flatmap(points_to_trees_with_points))


def points_to_trees_with_points_and_sizes(points: Strategy[Point]
                                          ) -> Strategy[Tuple[Tree, Point,
                                                              int]]:
    def to_trees_with_points_and_sizes(points_list: List[Point]
                                       ) -> Strategy[Tuple[Tree, Point, int]]:
        return strategies.tuples(strategies.just(tree(points_list)), points,
                                 strategies.integers(1, len(points_list)))

    return (strategies.lists(points,
                             min_size=1)
            .flatmap(to_trees_with_points_and_sizes))


trees_with_points_and_sizes = (points_strategies
                               .flatmap(points_to_trees_with_points_and_sizes))
