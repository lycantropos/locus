from functools import partial

from hypothesis import strategies

from tests.strategies import coordinates_strategies
from tests.utils import to_homogeneous_tuples

MAX_AXES = 10
axes = strategies.integers(1, MAX_AXES)
points_strategies = strategies.builds(to_homogeneous_tuples,
                                      coordinates_strategies,
                                      size=axes)
non_empty_points_lists = points_strategies.flatmap(partial(strategies.lists,
                                                           min_size=1))
