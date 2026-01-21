from functools import partial

from ground.hints import Point, Segment
from hypothesis import strategies as st

from locus._core.hilbert import MAX_COORDINATE
from locus.segmental import Tree
from tests.hints import ScalarT
from tests.strategies import (
    scalar_strategy_strategy,
    to_point_strategy,
    to_segment_strategy,
)
from tests.utils import context

MIN_SEGMENTS_SIZE = 2
max_children_counts = st.sampled_from(
    [2**power for power in range(1, 10)]
) | st.integers(2, MAX_COORDINATE)
segments_strategies = scalar_strategy_strategy.map(to_segment_strategy)
segments_lists = segments_strategies.flatmap(
    partial(st.lists, min_size=MIN_SEGMENTS_SIZE)
)
trees = st.builds(
    Tree,
    segments_lists,
    context=st.just(context),
    max_children=max_children_counts,
)


def scalar_strategy_to_tree_with_segment_strategy(
    scalar_strategy: st.SearchStrategy[ScalarT],
    /,
    *,
    min_size: int = MIN_SEGMENTS_SIZE,
    max_size: int | None = None,
) -> st.SearchStrategy[tuple[Tree[ScalarT], Segment[ScalarT]]]:
    segments = to_segment_strategy(scalar_strategy)
    return st.tuples(
        st.builds(
            Tree,
            st.lists(segments, min_size=min_size, max_size=max_size),
            context=st.just(context),
            max_children=max_children_counts,
        ),
        segments,
    )


trees_with_segments = scalar_strategy_strategy.flatmap(
    scalar_strategy_to_tree_with_segment_strategy
)


def scalar_strategy_to_tree_with_point_strategy(
    scalar_strategy: st.SearchStrategy[ScalarT],
    /,
    *,
    min_size: int = MIN_SEGMENTS_SIZE,
    max_size: int | None = None,
) -> st.SearchStrategy[tuple[Tree[ScalarT], Point[ScalarT]]]:
    return st.tuples(
        st.builds(
            Tree,
            st.lists(
                to_segment_strategy(scalar_strategy),
                min_size=min_size,
                max_size=max_size,
            ),
            context=st.just(context),
            max_children=max_children_counts,
        ),
        to_point_strategy(scalar_strategy),
    )


tree_with_point_strategy = scalar_strategy_strategy.flatmap(
    scalar_strategy_to_tree_with_point_strategy
)


def scalar_strategy_to_tree_with_point_and_size_strategy(
    scalar_strategy: st.SearchStrategy[ScalarT],
    /,
    *,
    min_size: int = MIN_SEGMENTS_SIZE,
    max_size: int | None = None,
) -> st.SearchStrategy[tuple[Tree[ScalarT], Point[ScalarT], int]]:
    def to_trees_with_points_and_sizes(
        segments_list_with_point: tuple[
            list[Segment[ScalarT]], Point[ScalarT]
        ],
        /,
    ) -> st.SearchStrategy[tuple[Tree[ScalarT], Point[ScalarT], int]]:
        segments_list, point = segments_list_with_point
        return st.tuples(
            st.builds(
                Tree,
                st.just(segments_list),
                context=st.just(context),
                max_children=max_children_counts,
            ),
            st.just(point),
            st.integers(1, len(segments_list)),
        )

    return st.tuples(
        st.lists(
            to_segment_strategy(scalar_strategy),
            min_size=min_size,
            max_size=max_size,
        ),
        to_point_strategy(scalar_strategy),
    ).flatmap(to_trees_with_points_and_sizes)


tree_with_point_and_size_strategy = scalar_strategy_strategy.flatmap(
    scalar_strategy_to_tree_with_point_and_size_strategy
)


def scalar_strategy_to_tree_with_segment_and_size_strategy(
    scalar_strategy: st.SearchStrategy[ScalarT],
    /,
    *,
    min_size: int = MIN_SEGMENTS_SIZE,
    max_size: int | None = None,
) -> st.SearchStrategy[tuple[Tree[ScalarT], Segment[ScalarT], int]]:
    def to_trees_with_segments_and_sizes(
        segments_list_with_segment: tuple[
            list[Segment[ScalarT]], Segment[ScalarT]
        ],
        /,
    ) -> st.SearchStrategy[tuple[Tree[ScalarT], Segment[ScalarT], int]]:
        segments_list, segment = segments_list_with_segment
        return st.tuples(
            st.builds(
                Tree,
                st.just(segments_list),
                context=st.just(context),
                max_children=max_children_counts,
            ),
            st.just(segment),
            st.integers(1, len(segments_list)),
        )

    segments = to_segment_strategy(scalar_strategy)
    return st.tuples(
        st.lists(segments, min_size=min_size, max_size=max_size), segments
    ).flatmap(to_trees_with_segments_and_sizes)


tree_with_segment_and_size_strategy = scalar_strategy_strategy.flatmap(
    scalar_strategy_to_tree_with_segment_and_size_strategy
)
