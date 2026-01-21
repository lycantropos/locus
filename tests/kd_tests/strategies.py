from functools import partial

from ground.hints import Box, Point
from hypothesis import strategies as st

from locus.kd import Tree
from tests.hints import ScalarT
from tests.strategies import (
    point_strategy_strategy,
    scalar_strategy_strategy,
    to_box_strategy,
    to_point_strategy,
)
from tests.utils import context, identity

non_empty_point_sequence_strategy = point_strategy_strategy.flatmap(
    partial(st.lists, min_size=1)
)
tree_strategy = st.builds(
    Tree, non_empty_point_sequence_strategy, context=st.just(context)
)


def point_strategy_to_tree_strategy(
    point_strategy: st.SearchStrategy[Point[ScalarT]],
    /,
    *,
    min_size: int = 1,
    max_size: int | None = None,
) -> st.SearchStrategy[Tree[ScalarT]]:
    return st.builds(
        Tree,
        st.lists(point_strategy, min_size=min_size, max_size=max_size),
        context=st.just(context),
    )


def point_strategy_to_tree_with_point_strategy(
    point_strategy: st.SearchStrategy[Point[ScalarT]], /
) -> st.SearchStrategy[tuple[Tree[ScalarT], Point[ScalarT]]]:
    return st.tuples(
        point_strategy_to_tree_strategy(point_strategy), point_strategy
    )


tree_with_point_strategy = point_strategy_strategy.flatmap(
    point_strategy_to_tree_with_point_strategy
)


def points_to_trees_with_points_and_sizes(
    points: st.SearchStrategy[Point[ScalarT]], /
) -> st.SearchStrategy[tuple[Tree[ScalarT], Point[ScalarT], int]]:
    def to_trees_with_points_and_sizes(
        points_list: list[Point[ScalarT]], /
    ) -> st.SearchStrategy[tuple[Tree[ScalarT], Point[ScalarT], int]]:
        return st.tuples(
            st.just(Tree(points_list, context=context)),
            points,
            st.integers(1, len(points_list)),
        )

    return st.lists(points, min_size=1).flatmap(to_trees_with_points_and_sizes)


tree_with_point_and_size_strategy = point_strategy_strategy.flatmap(
    points_to_trees_with_points_and_sizes
)


def scalar_strategy_to_tree_with_ball_strategy(
    scalar_strategy: st.SearchStrategy[ScalarT],
    /,
    *,
    min_tree_size: int = 1,
    max_tree_size: int | None = None,
) -> st.SearchStrategy[tuple[Tree[ScalarT], Point[ScalarT], ScalarT]]:
    points = to_point_strategy(scalar_strategy)
    return st.tuples(
        point_strategy_to_tree_strategy(
            points, min_size=min_tree_size, max_size=max_tree_size
        ),
        points,
        scalar_strategy.map(abs),
    )


tree_with_ball_strategy = st.builds(
    scalar_strategy_to_tree_with_ball_strategy, scalar_strategy_strategy
).flatmap(identity)


def scalar_strategy_to_tree_with_box_strategy(
    scalars: st.SearchStrategy[ScalarT],
    /,
    *,
    min_tree_size: int = 1,
    max_tree_size: int | None = None,
) -> st.SearchStrategy[tuple[Tree[ScalarT], Box[ScalarT]]]:
    return st.tuples(
        scalar_strategy_to_tree_strategy(
            scalars, min_tree_size=min_tree_size, max_tree_size=max_tree_size
        ),
        to_box_strategy(scalars),
    )


def scalar_strategy_to_tree_strategy(
    scalar_strategy: st.SearchStrategy[ScalarT],
    /,
    *,
    min_tree_size: int = 1,
    max_tree_size: int | None = None,
) -> st.SearchStrategy[Tree[ScalarT]]:
    return point_strategy_to_tree_strategy(
        to_point_strategy(scalar_strategy),
        min_size=min_tree_size,
        max_size=max_tree_size,
    )


tree_with_box_strategy = st.builds(
    scalar_strategy_to_tree_with_box_strategy, scalar_strategy_strategy
).flatmap(identity)
