from functools import partial

from hypothesis import strategies

from tests.strategies import points_strategies

non_empty_points_lists = points_strategies.flatmap(partial(strategies.lists,
                                                           min_size=1))
