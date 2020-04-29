from functools import partial
from typing import (Optional,
                    Tuple)

from hypothesis import strategies

from locus.core.hilbert import MAX_COORDINATE
from locus.hints import (Coordinate,
                         Interval)
from locus.r import Tree
from tests.strategies import (coordinates_strategies,
                              coordinates_to_intervals)
from tests.utils import Strategy

max_children_counts = strategies.integers(2, MAX_COORDINATE)
intervals_strategies = (coordinates_strategies
                        .map(partial(coordinates_to_intervals,
                                     dimension=2)))
non_empty_intervals_lists = (intervals_strategies
                             .flatmap(partial(strategies.lists,
                                              min_size=1)))


def coordinates_to_trees_with_intervals(coordinates: Strategy[Coordinate],
                                        *,
                                        min_size: int = 1,
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
