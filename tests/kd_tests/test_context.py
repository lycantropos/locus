from ground.base import Context
from hypothesis import given

from locus.kd import Tree
from . import strategies


@given(strategies.trees)
def test_basic(tree: Tree) -> None:
    result = tree.context

    assert isinstance(result, Context)
