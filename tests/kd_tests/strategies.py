from functools import partial

from ground.hints import Box, Point
from hypothesis import strategies as st

from locus.kd import Tree
from tests.hints import ScalarT
from tests.strategies import (
    points_strategy_strategy,
    scalar_strategy_strategy,
    to_box_strategy,
    to_point_strategy,
)
from tests.utils import identity

non_empty_point_sequence_strategy = points_strategy_strategy.flatmap(
    partial(st.lists, min_size=1)
)
trees = non_empty_point_sequence_strategy.map(Tree)


def points_to_trees(
    point_strategy: st.SearchStrategy[Point[ScalarT]],
    /,
    *,
    min_size: int = 1,
    max_size: int | None = None,
) -> st.SearchStrategy[Tree[ScalarT]]:
    return st.lists(point_strategy, min_size=min_size, max_size=max_size).map(
        Tree
    )


def points_to_trees_with_points(
    point_strategy: st.SearchStrategy[Point[ScalarT]], /
) -> st.SearchStrategy[tuple[Tree[ScalarT], Point[ScalarT]]]:
    return st.tuples(points_to_trees(point_strategy), point_strategy)


trees_with_points = points_strategy_strategy.flatmap(
    points_to_trees_with_points
)


def points_to_trees_with_points_and_sizes(
    points: st.SearchStrategy[Point[ScalarT]], /
) -> st.SearchStrategy[tuple[Tree[ScalarT], Point[ScalarT], int]]:
    def to_trees_with_points_and_sizes(
        points_list: list[Point[ScalarT]],
    ) -> st.SearchStrategy[tuple[Tree[ScalarT], Point[ScalarT], int]]:
        return st.tuples(
            st.just(Tree(points_list)),
            points,
            st.integers(1, len(points_list)),
        )

    return st.lists(points, min_size=1).flatmap(to_trees_with_points_and_sizes)


trees_with_points_and_sizes = points_strategy_strategy.flatmap(
    points_to_trees_with_points_and_sizes
)


def scalars_to_trees_with_balls(
    scalar_strategy: st.SearchStrategy[ScalarT],
    /,
    *,
    min_tree_size: int = 1,
    max_tree_size: int | None = None,
) -> st.SearchStrategy[tuple[Tree[ScalarT], Point[ScalarT], ScalarT]]:
    points = to_point_strategy(scalar_strategy)
    return st.tuples(
        points_to_trees(
            points, min_size=min_tree_size, max_size=max_tree_size
        ),
        points,
        scalar_strategy.map(abs),
    )


trees_with_balls = st.builds(
    scalars_to_trees_with_balls, scalar_strategy_strategy
).flatmap(identity)


def scalars_to_trees_with_boxes(
    scalars: st.SearchStrategy[ScalarT],
    *,
    min_tree_size: int = 1,
    max_tree_size: int | None = None,
) -> st.SearchStrategy[tuple[Tree[ScalarT], Box[ScalarT]]]:
    return st.tuples(
        scalars_to_trees(
            scalars, min_tree_size=min_tree_size, max_tree_size=max_tree_size
        ),
        to_box_strategy(scalars),
    )


def scalars_to_trees(
    scalars: st.SearchStrategy[ScalarT],
    *,
    min_tree_size: int = 1,
    max_tree_size: int | None = None,
) -> st.SearchStrategy[Tree[ScalarT]]:
    return points_to_trees(
        to_point_strategy(scalars),
        min_size=min_tree_size,
        max_size=max_tree_size,
    )


trees_with_boxes = st.builds(
    scalars_to_trees_with_boxes, scalar_strategy_strategy
).flatmap(identity)
