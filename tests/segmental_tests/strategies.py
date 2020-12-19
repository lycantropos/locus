from functools import partial
from typing import (List,
                    Optional,
                    Tuple)

from ground.hints import (Coordinate,
                          Point,
                          Segment)
from hypothesis import strategies

from locus.core.hilbert import MAX_COORDINATE
from locus.segmental import Tree
from tests.strategies import (coordinates_strategies,
                              coordinates_to_points,
                              coordinates_to_segments)
from tests.utils import Strategy

MIN_SEGMENTS_SIZE = 2
max_children_counts = (strategies.sampled_from([2 ** power
                                                for power in range(1, 10)])
                       | strategies.integers(2, MAX_COORDINATE))
segments_strategies = coordinates_strategies.map(coordinates_to_segments)
segments_lists = (segments_strategies
                  .flatmap(partial(strategies.lists,
                                   min_size=MIN_SEGMENTS_SIZE)))
trees = strategies.builds(Tree,
                          segments_lists,
                          max_children=max_children_counts)


def coordinates_to_trees_with_segments(coordinates: Strategy[Coordinate],
                                       *,
                                       min_size: int = MIN_SEGMENTS_SIZE,
                                       max_size: Optional[int] = None
                                       ) -> Strategy[Tuple[Tree, Segment]]:
    segments = coordinates_to_segments(coordinates)
    return strategies.tuples(
            strategies.builds(Tree,
                              strategies.lists(segments,
                                               min_size=min_size,
                                               max_size=max_size),
                              max_children=max_children_counts),
            segments)


trees_with_segments = (coordinates_strategies
                       .flatmap(coordinates_to_trees_with_segments))


def coordinates_to_trees_with_points(coordinates: Strategy[Coordinate],
                                     *,
                                     min_size: int = MIN_SEGMENTS_SIZE,
                                     max_size: Optional[int] = None
                                     ) -> Strategy[Tuple[Tree, Point]]:
    return (strategies.tuples(
            strategies.builds(Tree,
                              strategies.lists(
                                      coordinates_to_segments(coordinates),
                                      min_size=min_size,
                                      max_size=max_size),
                              max_children=max_children_counts),
            coordinates_to_points(coordinates)))


trees_with_points = (coordinates_strategies
                     .flatmap(coordinates_to_trees_with_points))


def coordinates_to_trees_with_points_and_sizes(
        coordinates: Strategy[Coordinate],
        *,
        min_size: int = MIN_SEGMENTS_SIZE,
        max_size: Optional[int] = None) -> Strategy[Tuple[Tree, Point, int]]:
    def to_trees_with_points_and_sizes(segments_list_with_point
                                       : Tuple[List[Segment], Point]
                                       ) -> Strategy[Tuple[Tree, Point, int]]:
        segments_list, point = segments_list_with_point
        return strategies.tuples(
                strategies.builds(Tree,
                                  strategies.just(segments_list),
                                  max_children=max_children_counts),
                strategies.just(point),
                strategies.integers(1, len(segments_list)))

    return (strategies.tuples(
            strategies.lists(coordinates_to_segments(coordinates),
                             min_size=min_size,
                             max_size=max_size),
            coordinates_to_points(coordinates))
            .flatmap(to_trees_with_points_and_sizes))


trees_with_points_and_sizes = (
    coordinates_strategies.flatmap(coordinates_to_trees_with_points_and_sizes))


def coordinates_to_trees_with_segments_and_sizes(
        coordinates: Strategy[Coordinate],
        *,
        min_size: int = MIN_SEGMENTS_SIZE,
        max_size: Optional[int] = None) -> Strategy[Tuple[Tree, Segment, int]]:
    def to_trees_with_segments_and_sizes(
            segments_list_with_segment: Tuple[List[Segment], Segment]
    ) -> Strategy[Tuple[Tree, Segment, int]]:
        segments_list, segment = segments_list_with_segment
        return strategies.tuples(
                strategies.builds(Tree,
                                  strategies.just(segments_list),
                                  max_children=max_children_counts),
                strategies.just(segment),
                strategies.integers(1, len(segments_list)))

    segments = coordinates_to_segments(coordinates)
    return (strategies.tuples(strategies.lists(segments,
                                               min_size=min_size,
                                               max_size=max_size),
                              segments)
            .flatmap(to_trees_with_segments_and_sizes))


trees_with_segments_and_sizes = coordinates_strategies.flatmap(
        coordinates_to_trees_with_segments_and_sizes)
