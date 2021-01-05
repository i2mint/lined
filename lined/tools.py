from functools import partial, wraps
from collections import deque
from typing import Union, Callable, Iterable

from lined.util import func_name


def keys_extractor(keys):
    def extract(x):
        return tuple((x[i] for i in keys))

    return extract


def items(mapping):
    """Get an items generator from a mapping"""
    return mapping.items()


def iterate(iterable: Iterable):
    """Just iterate through a iterable
    Use this to "consume" or "run" an iterator automatically.

    For example, consider the following:

    >>> from lined import Pipeline, iterize, iterate
    >>> pipe = Pipeline(iterize(lambda x: x * 2),
    ...                 iterize(lambda x: print(f"hello {x}")),
    ...            )
    >>>
    >>> for _ in pipe([1, 2, 3]):
    ...     pass
    hello 2
    hello 4
    hello 6

    It could be a bit awkward to have to "consume" the iterable to have it take effect.
    Just calling  ``pipe([1, 2, 3])`` to get those prints seems like a more natural way.
    This is where you can use `iterate`. It basically "launches" that consuming loop for you.

    >>> pipe = Pipeline(iterize(lambda x: x * 2),
    ...                iterize(lambda x: print(f"hello {x}")),
    ...                iterate
    ...               )
    >>>
    >>> pipe([1, 2, 3])
    hello 2
    hello 4
    hello 6

    """
    for _ in iterable:
        pass


# Function transformers ###################################################################


def extra_wraps(func, name=None, doc_prefix=""):
    func.__name__ = name or func_name(func)
    func.__doc__ = doc_prefix + getattr(func, '__name__', '')
    return func


def mywraps(func, name=None, doc_prefix=""):
    def wrapper(wrapped):
        return extra_wraps(wraps(func)(wrapped), name=name, doc_prefix=doc_prefix)

    return wrapper


def iterize(func, name=None):
    """From an Input->Ouput function, makes a Iterator[Input]->Itertor[Output]
    Some call this "vectorization", but it's not really a vector, but an iterable, thus the name.

    >>> f = lambda x: x * 10
    >>> f(2)
    20
    >>> iterized_f = iterize(f)
    >>> list(iterized_f(iter([1,2,3])))
    [10, 20, 30]

    Consider the following pipeline:

    >>> from lined import Pipeline
    >>>
    >>> pipe = Pipeline(lambda x: x * 2,
    ...                 lambda x: f"hello {x}")
    >>> pipe(1)
    'hello 2'

    But what if you wanted to use the pipeline on a "stream" of data. The following wouldn't work:

    >>> try:
    ...     pipe(iter([1,2,3]))
    ... except TypeError as e:
    ...     print(f"{type(e).__name__}: {e}")
    ...
    ...
    TypeError: unsupported operand type(s) for *: 'list_iterator' and 'int'

    Remember that error: You'll surely encounter it at some point.

    The solution to it is (often): ``iterize``,
    which transforms a function that is meant to be applied to a single object,
    into a function that is meant to be applied to an array, or any iterable of such objects.
    (You might be familiar (if you use `numpy` for example) with the related concept of "vectorization",
    or [array programming](https://en.wikipedia.org/wiki/Array_programming).)


    >>> from lined import Pipeline, iterize
    >>> from typing import Iterable
    >>>
    >>> pipe = Pipeline(iterize(lambda x: x * 2),
    ...                 iterize(lambda x: f"hello {x}"))
    >>> iterable = pipe([1, 2, 3])
    >>> assert isinstance(iterable, Iterable)  # see that the result is an iterable
    >>> list(iterable)  # consume the iterable and gather it's items
    ['hello 2', 'hello 4', 'hello 6']
    """
    wrapper = mywraps(func, name=name,
                      doc_prefix=f"generator version of {func_name(func)}:\n")
    return wrapper(partial(map, func))


generator_version = iterize  # back compatibility alias


def singularize_arg_input(func):
    """Make a func(args) function out of a func(*args) one"""

    @mywraps(func, doc_prefix=f"singularize_arg_input version of {func_name(func)}")
    def func_with_single_arg_input(args):
        return func(*args)

    return func_with_single_arg_input


class BufferStats(deque):
    """A callable (fifo) buffer. Calls add input to it, but also returns a function of it's contents.

    What "add" means is configurable (through ``add_new_val`` arg). Default is append, but can be extend etc.

    >>> bs = BufferStats(4, sum)
    >>> list(map(bs, range(7)))
    [0, 1, 3, 6, 10, 14, 18]

    See what happens when you feed the same sequence again:

    >>> list(map(bs, range(7)))
    [15, 12, 9, 6, 10, 14, 18]

    More examples:

    >>> list(map(BufferStats(4, ''.join), 'abcdefgh'))
    ['a', 'ab', 'abc', 'abcd', 'bcde', 'cdef', 'defg', 'efgh']

    >>> from math import prod
    >>> list(map(BufferStats(4, prod), range(7)))
    [0, 0, 0, 0, 24, 120, 360]

    With a different ``add_new_val`` choice.

    >>> bs = BufferStats(4, ''.join, add_new_val=deque.appendleft)
    >>> list(map(bs, 'abcdefgh'))
    ['a', 'ba', 'cba', 'dcba', 'edcb', 'fedc', 'gfed', 'hgfe']

    With ``add_new_val=deque.extend``, data can be fed in chunks.
    In the following, also see how we use iterize to get a function that takes an iterator and returns an iterator

    >>> from lined import iterize
    >>> window_stats = iterize(BufferStats(4, ''.join, add_new_val=deque.extend))
    >>> chks = ['a', 'bc', 'def', 'gh']
    >>> for x in window_stats(chks):
    ...     print(x)
    a
    abc
    cdef
    efgh

    Note: To those who might think that they can optimize this for special cases: Yes you can.
    But SHOULD you? Is it worth the increase in complexity and reduction in flexibility?
    See https://github.com/thorwhalen/umpyre/blob/master/misc/performance_of_rolling_window_stats.md

    """

    def __init__(self, maxlen, func: Callable = sum, add_new_val: Callable = deque.append):
        """

        :param maxlen: Size of the buffer
        :param func: The function to be computed (on buffer contents) and returned when buffer is "called"
        :param add_new_val: The function that adds values on the buffer. Signature must be (self, new_val)
            Is usually a deque method (``deque.append`` by default, but could be ``deque.extend``,
            ``deque.appendleft`` etc.). Can also be any other function that has a valid (self, new_val) signature.
        """
        super().__init__(maxlen=maxlen)
        self.func = func
        if isinstance(add_new_val, str):
            add_new_val = getattr(self, add_new_val)  # add_new_val is a method of deque
        self.add_new_val = add_new_val

    def __call__(self, new_val):
        self.add_new_val(self, new_val)  # add the new value
        return self.func(self)
