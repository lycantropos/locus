from functools import partial

from ground.hints import Point, Segment
from hypothesis import strategies as st

from locus.core.hilbert import MAX_COORDINATE
from locus.segmental import Tree
from tests.hints import ScalarT
from tests.strategies import (
    scalar_strategy_strategy,
    to_point_strategy,
    to_segment_strategy,
)

MIN_SEGMENTS_SIZE = 2
max_children_counts = st.sampled_from(
    [2**power for power in range(1, 10)]
) | st.integers(2, MAX_COORDINATE)
segments_strategies = scalar_strategy_strategy.map(to_segment_strategy)
segments_lists = segments_strategies.flatmap(
    partial(st.lists, min_size=MIN_SEGMENTS_SIZE)
)
trees = st.builds(Tree, segments_lists, max_children=max_children_counts)


def scalars_to_trees_with_segments(
    scalars: st.SearchStrategy[ScalarT],
    *,
    min_size: int = MIN_SEGMENTS_SIZE,
    max_size: int | None = None,
) -> st.SearchStrategy[tuple[Tree[ScalarT], Segment[ScalarT]]]:
    segments = to_segment_strategy(scalars)
    return st.tuples(
        st.builds(
            Tree,
            st.lists(segments, min_size=min_size, max_size=max_size),
            max_children=max_children_counts,
        ),
        segments,
    )


trees_with_segments = scalar_strategy_strategy.flatmap(
    scalars_to_trees_with_segments
)


def scalars_to_trees_with_points(
    scalars: st.SearchStrategy[ScalarT],
    *,
    min_size: int = MIN_SEGMENTS_SIZE,
    max_size: int | None = None,
) -> st.SearchStrategy[tuple[Tree[ScalarT], Point[ScalarT]]]:
    return st.tuples(
        st.builds(
            Tree,
            st.lists(
                to_segment_strategy(scalars),
                min_size=min_size,
                max_size=max_size,
            ),
            max_children=max_children_counts,
        ),
        to_point_strategy(scalars),
    )


trees_with_points = scalar_strategy_strategy.flatmap(
    scalars_to_trees_with_points
)


def scalars_to_trees_with_points_and_sizes(
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
                Tree, st.just(segments_list), max_children=max_children_counts
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


trees_with_points_and_sizes = scalar_strategy_strategy.flatmap(
    scalars_to_trees_with_points_and_sizes
)


def scalars_to_trees_with_segments_and_sizes(
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
                Tree, st.just(segments_list), max_children=max_children_counts
            ),
            st.just(segment),
            st.integers(1, len(segments_list)),
        )

    segments = to_segment_strategy(scalar_strategy)
    return st.tuples(
        st.lists(segments, min_size=min_size, max_size=max_size), segments
    ).flatmap(to_trees_with_segments_and_sizes)


trees_with_segments_and_sizes = scalar_strategy_strategy.flatmap(
    scalars_to_trees_with_segments_and_sizes
)
