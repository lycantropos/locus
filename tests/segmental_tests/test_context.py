from ground.context import Context
from hypothesis import given

from locus.segmental import Tree
from tests.hints import ScalarT

from . import strategies


@given(strategies.trees)
def test_basic(tree: Tree[ScalarT]) -> None:
    result = tree.context

    assert isinstance(result, Context)
