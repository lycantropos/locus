from hypothesis import strategies

integers = strategies.integers()
non_zero_integers = strategies.integers(max_value=-1) | strategies.integers(1)
