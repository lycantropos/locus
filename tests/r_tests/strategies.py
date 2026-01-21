from functools import partial

from ground.hints import Box, Point
from hypothesis import strategies as st

from locus._core.hilbert import MAX_COORDINATE
from locus.r import Tree
from tests.hints import ScalarT
from tests.strategies import (
    scalar_strategy_strategy,
    to_box_strategy,
    to_point_strategy,
)
from tests.utils import context

MIN_BOXES_SIZE = 2
max_child_count_strategy = st.sampled_from(
    [2**power for power in range(1, 10)]
) | st.integers(2, MAX_COORDINATE)
box_strategy_strategy = scalar_strategy_strategy.map(to_box_strategy)
box_list_strategy = box_strategy_strategy.flatmap(
    partial(st.lists, min_size=MIN_BOXES_SIZE)
)
tree_strategy = st.builds(
    Tree,
    box_list_strategy,
    context=st.just(context),
    max_children=max_child_count_strategy,
)


def scalar_strategy_to_tree_with_box_strategy(
    scalar_strategy: st.SearchStrategy[ScalarT],
    /,
    *,
    min_size: int = MIN_BOXES_SIZE,
    max_size: int | None = None,
) -> st.SearchStrategy[tuple[Tree[ScalarT], Box[ScalarT]]]:
    box_strategy = to_box_strategy(scalar_strategy)
    return st.tuples(
        st.builds(
            Tree,
            st.lists(box_strategy, min_size=min_size, max_size=max_size),
            context=st.just(context),
            max_children=max_child_count_strategy,
        ),
        box_strategy,
    )


tree_with_box_strategy = scalar_strategy_strategy.flatmap(
    scalar_strategy_to_tree_with_box_strategy
)


def scalars_to_trees_with_points(
    scalar_strategy: st.SearchStrategy[ScalarT],
    /,
    *,
    min_size: int = MIN_BOXES_SIZE,
    max_size: int | None = None,
) -> st.SearchStrategy[tuple[Tree[ScalarT], Point[ScalarT]]]:
    return st.tuples(
        st.builds(
            Tree,
            st.lists(
                to_box_strategy(scalar_strategy),
                min_size=min_size,
                max_size=max_size,
            ),
            context=st.just(context),
            max_children=max_child_count_strategy,
        ),
        to_point_strategy(scalar_strategy),
    )


tree_with_point_strategy = scalar_strategy_strategy.flatmap(
    scalars_to_trees_with_points
)


def scalar_to_tree_with_point_and_size_strategy(
    scalar_strategy: st.SearchStrategy[ScalarT],
    /,
    *,
    min_size: int = MIN_BOXES_SIZE,
    max_size: int | None = None,
) -> st.SearchStrategy[tuple[Tree[ScalarT], Point[ScalarT], int]]:
    def boxes_with_point_to_trees_with_points_and_sizes(
        boxes_with_point: tuple[list[Box[ScalarT]], Point[ScalarT]], /
    ) -> st.SearchStrategy[tuple[Tree[ScalarT], Point[ScalarT], int]]:
        boxes, point = boxes_with_point
        return st.tuples(
            st.builds(
                Tree,
                st.just(boxes),
                context=st.just(context),
                max_children=max_child_count_strategy,
            ),
            st.just(point),
            st.integers(1, len(boxes)),
        )

    return st.tuples(
        st.lists(
            to_box_strategy(scalar_strategy),
            min_size=min_size,
            max_size=max_size,
        ),
        to_point_strategy(scalar_strategy),
    ).flatmap(boxes_with_point_to_trees_with_points_and_sizes)


trees_with_points_and_sizes = scalar_strategy_strategy.flatmap(
    scalar_to_tree_with_point_and_size_strategy
)
