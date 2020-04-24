from typing import (List,
                    Optional,
                    Tuple)

from hypothesis import strategies

from locus.hints import (Coordinate,
                         Point)
from locus.kd import (Tree,
                      tree)
from tests.strategies import (axes,
                              coordinates_strategies,
                              coordinates_to_points,
                              points_strategies)
from tests.utils import (Strategy,
                         identity)


def points_to_trees(points: Strategy[Point],
                    *,
                    min_size: int = 1,
                    max_size: Optional[int] = None) -> Strategy[Tree]:
    return (strategies.lists(points,
                             min_size=min_size,
                             max_size=max_size)
            .map(tree))


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
        return strategies.tuples(strategies.just(tree(points_list)), points,
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
