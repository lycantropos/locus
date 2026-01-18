from ground.context import Context
from hypothesis import given

from locus.kd import Tree
from tests.hints import ScalarT

from . import strategies


@given(strategies.tree_strategy)
def test_basic(tree: Tree[ScalarT]) -> None:
    result = tree.context

    assert isinstance(result, Context)
