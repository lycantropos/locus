from functools import partial
from typing import (List,
                    Optional,
                    Tuple)

from hypothesis import strategies

from locus.core.hilbert import MAX_COORDINATE
from locus.hints import (Coordinate,
                         Interval,
                         Point)
from locus.r import Tree
from tests.strategies import (coordinates_strategies,
                              coordinates_to_intervals,
                              coordinates_to_points)
from tests.utils import Strategy

MIN_INTERVALS_SIZE = 2
max_children_counts = (strategies.sampled_from([2 ** power
                                                for power in range(1, 10)])
                       | strategies.integers(2, MAX_COORDINATE))
intervals_strategies = (coordinates_strategies
                        .map(partial(coordinates_to_intervals,
                                     dimension=2)))
intervals_lists = (intervals_strategies
                   .flatmap(partial(strategies.lists,
                                    min_size=MIN_INTERVALS_SIZE)))
trees = strategies.builds(Tree,
                          intervals_lists,
                          max_children=max_children_counts)


def coordinates_to_trees_with_intervals(coordinates: Strategy[Coordinate],
                                        *,
                                        min_size: int = MIN_INTERVALS_SIZE,
                                        max_size: Optional[int] = None
                                        ) -> Strategy[Tuple[Tree, Interval]]:
    intervals = coordinates_to_intervals(coordinates,
                                         dimension=2)
    return strategies.tuples(
            strategies.builds(Tree,
                              strategies.lists(intervals,
                                               min_size=min_size,
                                               max_size=max_size),
                              max_children=max_children_counts),
            intervals)


trees_with_intervals = (coordinates_strategies
                        .flatmap(coordinates_to_trees_with_intervals))


def coordinates_to_trees_with_points(coordinates: Strategy[Coordinate],
                                     *,
                                     min_size: int = MIN_INTERVALS_SIZE,
                                     max_size: Optional[int] = None
                                     ) -> Strategy[Tuple[Tree, Point]]:
    intervals = coordinates_to_intervals(coordinates,
                                         dimension=2)
    return (strategies.tuples(
            strategies.builds(Tree,
                              strategies.lists(intervals,
                                               min_size=min_size,
                                               max_size=max_size),
                              max_children=max_children_counts),
            coordinates_to_points(coordinates,
                                  dimension=2)))


trees_with_points = (coordinates_strategies
                     .flatmap(coordinates_to_trees_with_points))


def coordinates_to_trees_with_points_and_sizes(
        coordinates: Strategy[Coordinate],
        *,
        min_size: int = MIN_INTERVALS_SIZE,
        max_size: Optional[int] = None) -> Strategy[Tuple[Tree, Point, int]]:
    def to_trees_with_points_and_sizes(intervals_with_point
                                       : Tuple[List[Interval], Point]
                                       ) -> Strategy[Tuple[Tree, Point, int]]:
        intervals, point = intervals_with_point
        return strategies.tuples(
                strategies.builds(Tree,
                                  strategies.just(intervals),
                                  max_children=max_children_counts),
                strategies.just(point),
                strategies.integers(1, len(intervals)))

    return (strategies.tuples(
            strategies.lists(coordinates_to_intervals(coordinates,
                                                      dimension=2),
                             min_size=min_size,
                             max_size=max_size),
            coordinates_to_points(coordinates,
                                  dimension=2))
            .flatmap(to_trees_with_points_and_sizes))


trees_with_points_and_sizes = (
    coordinates_strategies.flatmap(coordinates_to_trees_with_points_and_sizes))
