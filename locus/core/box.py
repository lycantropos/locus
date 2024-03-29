from ground.hints import (Box,
                          Point)


def contains_point(box: Box, point: Point) -> bool:
    return (box.min_x <= point.x <= box.max_x
            and box.min_y <= point.y <= box.max_y)


def overlaps(left: Box, right: Box) -> bool:
    return (right.min_x < left.max_x and left.min_x < right.max_x
            and right.min_y < left.max_y and left.min_y < right.max_y)


def is_subset_of(test: Box, goal: Box) -> bool:
    return (goal.min_x <= test.min_x and test.max_x <= goal.max_x
            and goal.min_y <= test.min_y and test.max_y <= goal.max_y)
