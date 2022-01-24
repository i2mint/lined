"""Tests for base.py"""

from lined.base import _merge_funcs_and_named_funcs
import pytest


class my_class:
    """A simple class with one method"""

    def add_one(self, x):
        return x + 1


class my_other_class:
    """Another simple class with one method"""

    def add_one(self, x):
        return x + 2


first = my_class()
f = first.add_one

second = my_class()
g = second.add_one

third = my_other_class()
h = third.add_one

i = lambda x: 2 * x


def j(x: int):
    return 2 * x + 1


@pytest.mark.parametrize(
    "funcs_tuple",
    [
        (f,),
        (f, g),
        (f, g, h),
        (f, g, h, i),
        (f, g, h, i, j),
        (f, g, h, i, j, f, g, h, i, j),
    ],
)
def test_merge_funcs_and_named_funcs(funcs_tuple):
    """Test that no function is lost after merging"""
    assert len(_merge_funcs_and_named_funcs(funcs_tuple, named_funcs={})[0]) == len(
        funcs_tuple
    ), "Merging resulted in losing at least one function"
