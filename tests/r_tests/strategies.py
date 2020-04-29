from functools import partial

from hypothesis import strategies

from locus.core.hilbert import MAX_COORDINATE
from tests.strategies import (coordinates_strategies,
                              coordinates_to_intervals)

max_children_counts = strategies.integers(2, MAX_COORDINATE)
intervals_strategies = (coordinates_strategies
                        .map(partial(coordinates_to_intervals,
                                     dimension=2)))
non_empty_intervals_lists = (intervals_strategies
                             .flatmap(partial(strategies.lists,
                                              min_size=1)))
