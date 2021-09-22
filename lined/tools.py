"""
All kinds of useful tools to use in pipelines.
"""
from functools import partial, wraps
from collections import deque
from typing import Any
from dataclasses import dataclass
from operator import not_

from lined.util import func_name, partial_plus, n_required_args
from lined.simple import Pipe


def negate(
    func,
):  # TODO: Do we want to use wraps(func) to get more than just signature?
    """Get a negated version of a function

    Will return a function with
    the same signature, but whose output is negated (that is, it calls the original
    function getting the `output` but instead of returning it,
    it returns `not output`.

    >>> sum([1, 2, 3])
    6
    >>> sum([-2, 2])
    0
    >>> sum_is_zero = negate(sum)
    >>> sum_is_zero([1, 2, 3])
    False
    >>> sum_is_zero([-2, 2])
    True
    """
    return Pipe(func, not_)


def identity(x):
    """Takes one argument, and returns it as is"""
    return x


def blind(x, output):
    """Takes one argument, and returns it as is.
    The output is meant to be bound by currying (functools.partial)

    >>> true_no_matter_what = partial(blind, output=True)
    >>> false_no_matter_what = partial(blind, output=False)
    >>> true_no_matter_what(42)
    True
    >>> false_no_matter_what(42)
    False
    """
    return output


true_no_matter_what = partial(blind, output=True)
false_no_matter_what = partial(blind, output=False)


def _extract_first_argument(args: tuple, kwargs: dict):
    """
    Returns the tuple (X, _args, _kwargs) where X is the first argument (
    found in either args or kwargs), and _args, _kwargs are the same (with X
    removed)

    >>> _extract_first_argument((1,2,3), {'d': 4})
    (1, [2, 3], {'d': 4})
    >>> _extract_first_argument((), {'d': 4, 'e': 5})
    (4, [], {'e': 5})

    """
    if len(args) > 0:
        first_arg_val, *_args = args
        return first_arg_val, _args, kwargs
    else:
        first_arg_name = next(iter(kwargs), None)
        if first_arg_name is None:
            raise ValueError(
                "You need to have at least one argument (the data (aka 'X'))"
            )
        first_arg_val = kwargs.pop(first_arg_name)
        return first_arg_val, [], kwargs


# ------------------------------------------------------------------------------

from operator import le
from typing import Union, Callable, Generator, Iterable
from itertools import tee


def if_then_else(
    x, if_func=true_no_matter_what, then_func=identity, else_func=identity
):
    """Implement the if-then-else logic as a function.

    >>> if_then_else(
    ...     'world',
    ...     if_func=lambda x: x == 'world',
    ...     then_func="hello {}".format,
    ...     else_func=lambda x: x * 2)
    'hello world'
    >>> if_then_else('bora', if_func=lambda x: x == 'world',
    ... then_func="hello{}".format, else_func=lambda x: x * 2)
    'borabora'

    Really, it's meant to be curried to make functional components.
    For example, to make a function that ensures that a string is
    encapsulated in a tuple, we could do this:

    >>> def is_a_str(x): return isinstance(x, str)
    >>> def make_it_a_tuple(x): return tuple([x])
    >>>
    >>> ensure_tuple_if_string = partial(
    ...     if_then_else,
    ...     if_func=is_a_str,
    ...     then_func=make_it_a_tuple
    ... )
    >>> ensure_tuple_if_string('a string')
    ('a string',)
    >>> ensure_tuple_if_string(['a', 'list'])  # not a string so returned as is
    ['a', 'list']
    """
    if if_func(x):
        return then_func(x)
    else:
        return else_func(x)


def make_it_a_tuple(x):
    return tuple([x])


def is_a_str(x):
    return isinstance(x, str)


cast_to_tuple_if_string = partial(
    if_then_else, if_func=is_a_str, then_func=make_it_a_tuple
)


def is_not_iterable_or_is_a_str(x):
    return not isinstance(x, Iterable) or isinstance(x, str)


cast_to_tuple_if_non_iterable_or_a_string = partial(
    if_then_else, if_func=is_not_iterable_or_is_a_str, then_func=make_it_a_tuple
)


# ------------ Tools for iterables ---------------------------------------------


from itertools import groupby


class Command:
    """Make a no-input callable that will execute a specific function call.

    >>> command = Command(sum, [1, 2, 3])
    >>> command()
    6
    >>> command = Command(print, 'hello', 'world', sep=', ')
    >>> command()
    hello, world

    See: https://en.wikipedia.org/wiki/Command_pattern
    """

    def __init__(self, func, *args, **kwargs):
        self.func, self.args, self.kwargs = func, args, kwargs

    def __call__(self):
        return self.func(*self.args, **self.kwargs)


class ItemsNotSorted(RuntimeError):
    """Use to indicate that two consecutive items where not in the expected
    order"""


def return_instead_of_raising_exceptions(func=None, *, exceptions=(Exception,)):
    """Make a function return its exceptions instead of raising them.

    >>> def foo(x, y):
    ...     return x / y
    >>> f = return_instead_of_raising_exceptions(foo)
    >>> f(6, 2)
    3.0
    >>> f(1,0)  # note that this doesn't raise, but returns the exception (instance)
    ZeroDivisionError('division by zero')

    :param func: The function to transform.
    :param exceptions: The exceptions to handle
        Default is Exception (letting other BaseException instances like
        KeyboardInterrupt still be raised). If you need more exceptions, or less
        exceptions to be handled, enter them here.
    :return:
    """

    def _process_exceptions(exceptions):
        if isinstance(exceptions, type) and issubclass(exceptions, BaseException):
            return (exceptions,)  # needs to be a tuple
        elif isinstance(exceptions, Iterable):
            exceptions = tuple(exceptions)
            assert all(issubclass(e, BaseException) for e in exceptions), (
                "All elements of exceptions must be subclasses of BaseException: "
                "Was {exceptions}"
            )
        else:
            raise TypeError(
                f"exceptions must be a BaseException subclass or iterable thereof: "
                f"{exceptions}"
            )
        return exceptions

    exceptions = _process_exceptions(exceptions)

    if func is None:
        return partial(return_instead_of_raising_exceptions, exceptions=exceptions)

    @wraps(func)
    def func_that_returns_instead_of_raising_exceptions(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except exceptions as exception_instance:
            return exception_instance

    return func_that_returns_instead_of_raising_exceptions


def raise_(exception):
    """raises the given exception (instance or callable that returns one)
    Meant to be hooked to the out put of a function that returns an exception or a
    command to raise one.

    """
    if isinstance(exception, BaseException):
        raise exception
    elif callable(exception):
        raise exception()
    else:
        raise TypeError(
            f"exception must be an BaseException instance or a "
            f"callable that returns one. Was: {exception}"
        )


raise_not_sorted_error = Command(raise_, ItemsNotSorted)

from itertools import groupby


def enumerate_groups(iterable, key=None, start=0):
    """Get enumeration of groups during a groupby call.

    :param iterable: An iterable
    :param key: The key to use in the groupby logic
    :param start: Where to start the enumeration (default is 0)
    :return: A generator of (group_idx, group, item) triples

    >>> iterable = [0, 0, 0, 2, 5, 7, 8, 0, 0, 9, 3, 1]
    >>> assert list(enumerate_groups(iterable, key=lambda x: x > 0)) == [
    ...  (0, False, 0),
    ...  (0, False, 0),
    ...  (0, False, 0),
    ...  (1, True, 2),
    ...  (1, True, 5),
    ...  (1, True, 7),
    ...  (1, True, 8),
    ...  (2, False, 0),
    ...  (2, False, 0),
    ...  (3, True, 9),
    ...  (3, True, 3),
    ...  (3, True, 1),
    ...  ]


    """
    g = groupby(iterable, key)
    for group_idx, (group, grouped_items) in enumerate(g, start):
        for item in grouped_items:
            yield group_idx, group, item


def pairwise(iterable):
    """Yield sliding window pairs

    >>> list(pairwise([1, 2, 3, 4]))
    [(1, 2), (2, 3), (3, 4)]"""
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def raise_exception(exception: Union[Callable, BaseException], *args, **kwargs):
    """Raise an exception (from an exception instance, or a callable that
    makes one"""
    if isinstance(exception, Callable):
        exception = exception(*args, **kwargs)
    raise exception


def consume_until_error(iterable, caught_errors=(Exception,)):
    """Iterable that will simply exit with out error if one of the caught
    errors occurs.

    >>> list(consume_until_error(map(lambda x: 1 / x, [4, 2, 1, 0, -1])))
    [0.25, 0.5, 1.0]
    """
    caught_errors = cast_to_tuple_if_non_iterable_or_a_string(caught_errors) + (
        StopIteration,
    )
    it = iter(iterable)
    while True:
        try:
            yield next(it)
        except caught_errors:
            break


def _validated_comparison_func(key: Callable):
    n_required = n_required_args(key)
    if n_required == 1:

        def comp_func(x, y):
            return key(x) <= key(y)

        return comp_func
    assert n_required == 2, (
        f"key should be a callable with 1 or 2 required " f"arguments"
    )
    return key


def check_sorted_during_iteration(
    iterable: Iterable,
    key: Callable[[Any, Any], bool] = le,
    not_sorted_callback: Union[Callable, BaseException] = raise_not_sorted_error,
) -> Generator:
    r"""Wrap an iterable so that ordering of the elements is checked at runtime.

    :param iterable: Iterable to consume
    :param key: The function that defines what it means to be sorted.
        Could be a Any->bool function, which will act like the key argument
        of builtin sorted for example.
        Could also be an explicit (element, next_element)->bool function that
        returns True iff in the right order
    :param not_sorted_callback: The function to call when two consecutive
    elements are not sorted. For example:
        - raising an error (the default)
        - logging the information, and skiping the offending element (or not)
    :return: A generator consuming the input iterable

    >>> list(check_sorted_during_iteration(iter([1, 2, 3, 4])))
    [1, 2, 3, 4]

    >>> try:
    ...     for i, x in enumerate(
    ...             check_sorted_during_iteration([2, 4, 3, 6]), 1):
    ...         print(x)
    ... except ItemsNotSorted:
    ...     print(
    ...         f"ItemsNotSorted after {i} element (whose value was {x})")
    ...     print(
    ...         "----> Normally, here, you'd put exception handling code")
    ...
    2
    4
    ItemsNotSorted after 2 element (whose value was 4)
    ----> Normally, here, you'd put exception handling code

    Now, mind you, you have total control over what sorted means.
    For example, to define it as strict

    >>> comp = lambda x, y: x > y  # in real life, use operator.gt
    >>> list(check_sorted_during_iteration(iter([4, 3, 2, 1]), key=comp))
    [4, 3, 2, 1]

    Now for a more complex example.
    First we'll define a function that will consume the iterable until an
    error occurs, returning the elements consumed.

    >>> from lined.tools import consume_until_error
    >>> from lined.simple import Pipe
    >>> consume = Pipe(check_sorted_during_iteration, consume_until_error, list)

    >>> iterable = ['a', 'ba', 'cba', 'cba', 'back', 'bacca']
    >>> consume(iterable, lambda x, y: x < y)  # compare with strict <
    ['a', 'ba', 'cba']
    >>> consume(iterable, len)  # compare based on the length
    ['a', 'ba', 'cba', 'cba', 'back', 'bacca']
    >>> consume(iterable, lambda x: x[0])  # compare based on first letter only
    ['a', 'ba', 'cba', 'cba']
    >>> # compare based on whether the previous element is a subset of the next:
    >>> consume(iterable, lambda x, y: set(x).issubset(y))
    ['a', 'ba', 'cba', 'cba', 'back']

    """

    # TODO: Optimization opportunity:
    #   In this implementation, the key of an element is computed twice (once
    #   when element, once when next_element)
    # TODO: key could be generalized to being a Callable[[element,
    #  next_element], bool].
    #   Though note that it's only an interface flexibility since same could
    #   (?) be acheived with a key returning an
    #   instance of a class such that class.__le__(element, next_element) is
    #   what is desired
    key = _validated_comparison_func(key)
    for element, next_element in pairwise(iterable):
        yield element
        if not key(element, next_element):
            not_sorted_callback()
    yield next_element


# ------------------------------------------------------------------------------


def del_fields(d, fields):
    """Returns the same mapping, but with specified fields removed.
    Intended to be applied to a stream of Mappings, using partial to fix fields

    >>> d = [{'a': 1, 'b': 2}, {'a': 11, 'c': 3}]
    >>> list(map(partial(del_fields, fields=['a']), d))
    [{'b': 2}, {'c': 3}]

    """
    if isinstance(fields, str) or not isinstance(fields, Iterable):
        fields = [fields]
    for f in fields:
        d.pop(f, None)
    return d


def add_name(obj, name=None):
    if name is None:
        name = type(obj).__name__
    obj.__name__ = name
    return obj


def keys_extractor(keys):
    """Deprecated: Use operator.itemgetter(*keys) instead."""

    def extract(x):
        return tuple((x[i] for i in keys))

    return extract


def apply_to_single_item(func: Callable, item_idx: int):
    """Get a version of func that applies itself to only the item_idx-th
    element of the input,
    leaving the rest untouched.

    That is, apply_to_single_item(func, 2), for example, is a new_func such that
    ```
        new_func([a, b, c, d, e]) == [a, b, func(c), d, e]
    ```

    :param func: A function to apply to a single element of an iterable (that
    has a [...])
    :param item_idx: The particular item index to apply function to
    :return:

    >>> apply_to_second_item = apply_to_single_item(
    ...     func=lambda x: x * 10, item_idx=1)
    >>> apply_to_second_item([1, 2, 3, 4])
    (1, 20, 2, 3, 4)
    """

    @wraps(func)
    def wrapped(first_arg, *args, **kwargs):
        val_to_apply_func_to = first_arg[item_idx]
        func_output = func(val_to_apply_func_to)
        return tuple([*first_arg[:item_idx], func_output, *first_arg[item_idx:]])

    return wrapped


def items(mapping):
    """Get an items generator from a mapping"""
    return mapping.items()


def iterate(iterable: Iterable):
    """Just iterate through a iterable
    Use this to "consume" or "run" an iterator automatically.

    For example, consider the following:

    >>> from lined import Pipe, iterize, iterate
    >>> pipe = Pipe(
    ...     iterize(lambda x: x * 2),
    ...     iterize(lambda x: print(f"hello {x}")),
    ... )
    >>>
    >>> for _ in pipe([1, 2, 3]):
    ...     pass
    hello 2
    hello 4
    hello 6

    It could be a bit awkward to have to "consume" the iterable to have it
    take effect.
    Just calling  ``pipe([1, 2, 3])`` to get those prints seems like a more
    natural way.
    This is where you can use `iterate`. It basically "launches" that
    consuming loop for you.

    >>> pipe = Pipe(
    ...     iterize(lambda x: x * 2),
    ...     iterize(lambda x: print(f"hello {x}")),
    ...     iterate
    ... )
    >>>
    >>> pipe([1, 2, 3])
    hello 2
    hello 4
    hello 6

    """
    for _ in iterable:
        pass


def append_output_to_input(func, appender=lambda x, output: (x, output)):
    """Decorator that makes the function into a function returning its input with output

    ┌─────────────┐
    │    input    │
    └─────────────┘
         │
         ▼
    ┌─────────────┐
    │    func     │
    └─────────────┘
         │
         ▼
    ┌─────────────────┐
    │ (input, output) │
    └─────────────────┘

    >>> func = lambda x: f"hello {x}"
    >>> func('world')
    'hello world'
    >>> new_func = append_output_to_input(func)
    >>> new_func('world')
    ('world', 'hello world')
    """
    assert n_required_args(func) == 1

    @wraps(func)
    def _func(x):
        output = func(x)
        return appender(x, output)

    return _func


def side_call(x, callback):
    """Identity function that calls a callaback function before returning the
    input as is (unless the input is mutable and the callback changes it).

    >>> from lined import Pipe
    >>> add2 = lambda x: x + 2
    >>> add2(40)
    42
    >>> from functools import partial
    >>> logger = partial(side_call, callback=lambda x: print(f"input is {x}"))
    >>> logged_add2 = Pipe(logger, add2)
    >>> logged_add2(40)
    input is 40
    42
    """
    callback(x)
    return x


print_and_pass_on = partial_plus(
    side_call,
    callback=print,
    __name__="print_and_pass_on",
    __doc__="Passes input through to output, but prints before outputing",
)

# Function transformers
# ###################################################################


def extra_wraps(func, name=None, doc_prefix=""):
    func.__name__ = name or func_name(func)
    func.__doc__ = doc_prefix + getattr(func, "__name__", "")
    return func


def mywraps(func, name=None, doc_prefix=""):
    def wrapper(wrapped):
        return extra_wraps(wraps(func)(wrapped), name=name, doc_prefix=doc_prefix)

    return wrapper


def tail_io(func):
    """Will apply function only to the tail of tuple inputs, still passing
    the header on.
    That is, from a ``x -> func(x)`` function, you get a ``(*header, x) -> (
    *header, func(x))`` function.

    >>> def foo(x):
    ...    return x * 2
    >>>
    >>> foo('boo')
    'booboo'
    >>> new_foo = tail_io(foo)
    >>> new_foo((7, 'boo'))
    (7, 'booboo')
    >>> new_foo(('all', 'items', 'but', 'the', 'last', 'are', 'just',
    ...          'passed', 'on', 'boo'))
    ('all', 'items', 'but', 'the', 'last', 'are', 'just', 'passed', 'on', 'booboo')

    """

    @mywraps(func)
    def _func(input):
        *header, real_input = input
        out = func(real_input)
        return (*header, out)

    return _func


def iterize(func, name=None):
    """From an Input->Ouput function, makes a Iterator[Input]->Itertor[Output]
    Some call this "vectorization", but it's not really a vector, but an
    iterable, thus the name.

    >>> f = lambda x: x * 10
    >>> f(2)
    20
    >>> iterized_f = iterize(f)
    >>> list(iterized_f(iter([1,2,3])))
    [10, 20, 30]

    Consider the following pipeline:

    >>> from lined import Pipe
    >>>
    >>> pipe = Pipe(lambda x: x * 2,
    ...                 lambda x: f"hello {x}")
    >>> pipe(1)
    'hello 2'

    But what if you wanted to use the pipeline on a "stream" of data. The
    following wouldn't work:

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
    into a function that is meant to be applied to an array, or any iterable
    of such objects.
    (You might be familiar (if you use `numpy` for example) with the related
    concept of "vectorization",
    or [array programming](https://en.wikipedia.org/wiki/Array_programming).)


    >>> from lined import Pipe, iterize
    >>> from typing import Iterable
    >>>
    >>> pipe = Pipe(iterize(lambda x: x * 2),
    ...                 iterize(lambda x: f"hello {x}"))
    >>> iterable = pipe([1, 2, 3])
    >>> # see that the result is an iterable
    >>> assert isinstance(iterable, Iterable)
    >>> list(iterable)  # consume the iterable and gather it's items
    ['hello 2', 'hello 4', 'hello 6']
    """
    # # TODO: Try replacing with partial_plus instead
    # wrapper = mywraps(
    #     func, name=name, doc_prefix=f"generator version of {func_name(func)}:\n"
    # )
    #
    # _func = partial(map, func)
    # new_sig = Sig(map).normalize_kind(kind=Parameter.POSITIONAL_ONLY)
    #
    # @wrapper
    # @new_sig
    # def __func(*args):
    #     return _func(*args)
    #
    # __func._iterized = True
    # return __func

    # the simpler earlier version has problems with LineParametrized
    #   TypeError: map() takes no keyword arguments
    # because
    #   args, kwargs = Sig(func).source_args_and_kwargs(*args, **kwargs)
    # made kwargs that made map partial choke.

    wrapper = mywraps(
        func, name=name, doc_prefix=f"generator version of {func_name(func)}:\n"
    )
    return wrapper(partial(map, func))


def wrap_first_arg_in_list(func):
    """Takes a func(X,...) function and returns a func([X],...) function."""

    @wraps(func)
    def _func(*args, **kwargs):
        first_arg_val, args, kwargs = _extract_first_argument(args, kwargs)
        return func([first_arg_val], *args, **kwargs)

    return _func


def deiterize(func):
    """The inverse of iterize.
    Takes an "iterized" (a.k.a. "vectorized") function (i.e. a function that
    works on iterables), and
    That is, takes a func(X,...) function and returns a next(iter(func([X],
    ...))) function."""
    return Pipe(wrap_first_arg_in_list(func), iter, next)


generator_version = iterize  # back compatibility alias


def mk_filter(filter_func=None):
    return partial_plus(
        filter,
        filter_func,
        __name__="mk_filter",
        __doc__="Makes a filter with a fixed filt func.",
    )


def map_star(func):
    """Make a func(args) function out of a func(*args) o.
    Also known as singularize_arg_input.
    In a way, the opposite of map_starexpanded_args.

    >>> def foo(a, b):
    ...     return a + b
    >>> singularized_foo = map_star(foo)
    >>> singularized_foo((2, 3))
    5
    >>> assert singularized_foo([2, 3]) == singularized_foo({2, 3}) == foo(2, 3)
    """

    @mywraps(func, doc_prefix=f"map_star version of {func_name(func)}")
    def func_with_single_arg_input(args):
        return func(*args)

    return func_with_single_arg_input


singularize_arg_input = map_star  # alias


def expanded_args(func):
    """Make's a func(*args) function out of a func(args) one.
    In a way, the opposite of map_star.

    >>> sum([1,2,3,4])
    10
    >>> mysum = expanded_args(sum)
    >>> mysum(1, 2, 3, 4)
    10

    """

    @mywraps(func, doc_prefix=f"expanded_args version of {func_name(func)}")
    def _func(*args):
        return func(args)

    return _func


class Enumerate:
    """Decorator a function so it enumerates the number of calls.
    Or in general, returns (cursor, func(x)) instead of just func(x),
    where the start and step of the cursor can
    be defined (default is start=0 and step=1)

    >>> def foo(x):
    ...    return x * 2
    >>> new_foo = Enumerate(foo)
    >>> new_foo('ha')
    (0, 'haha')
    >>> new_foo('ho')
    (1, 'hoho')
    >>> enum_foo_with_step = Enumerate(foo, start=3, step=7)
    >>> enum_foo_with_step('z')
    (3, 'zz')
    >>> enum_foo_with_step(11)
    (10, 22)
    """

    def __init__(self, func, start=0, step=1):
        self.func = func
        self.cursor = start
        self.step = step

    def __call__(self, *args, **kwargs):
        current_cursor = self.cursor
        out = self.func(*args, **kwargs)
        self.cursor += self.step
        return current_cursor, out


def with_cursor(func, start=0, step=1):
    """Decorator a function so it enumerates the number of calls.
    Or in general, returns (cursor, func(x)) instead of just func(x),
    where the start and step of the cursor can
    be defined (default is start=0 and step=1)

    >>> def foo(x):
    ...    return x * 2
    >>> new_foo = with_cursor(foo)
    >>> new_foo('ha')
    (0, 'haha')
    >>> new_foo('ho')
    (1, 'hoho')
    >>> enum_foo_with_step = with_cursor(foo, start=3, step=7)
    >>> enum_foo_with_step('z')
    (3, 'zz')
    >>> enum_foo_with_step(11)
    (10, 22)
    """

    @wraps(func)
    def _func(*args, **kwargs):
        current_cursor = _func.cursor
        out = func(*args, **kwargs)
        _func.cursor += step
        return current_cursor, out

    _func.cursor = start
    return _func


Stats = Any

from typing import cast

_no_value_specified_sentinel = cast(int, object())


# _no_value_specified_sentinel = object()


class BufferStats(deque):
    """A callable (fifo) buffer. Calls add input to it, but also returns some results
    computed from it's contents.

    What "add" means is configurable (through ``add_new_val`` arg). Default
    is append, but can be extend etc.

    >>> bs = BufferStats(maxlen=4, func=sum)
    >>> list(map(bs, range(7)))
    [0, 1, 3, 6, 10, 14, 18]

    See what happens when you feed the same sequence again:

    >>> list(map(bs, range(7)))
    [15, 12, 9, 6, 10, 14, 18]

    More examples:

    >>> list(map(BufferStats(maxlen=4, func=''.join), 'abcdefgh'))
    ['a', 'ab', 'abc', 'abcd', 'bcde', 'cdef', 'defg', 'efgh']

    >>> from math import prod
    >>> list(map(BufferStats(maxlen=4, func=prod), range(7)))
    [0, 0, 0, 0, 24, 120, 360]

    With a different ``add_new_val`` choice.

    >>> bs = BufferStats(maxlen=4, func=''.join, add_new_val=deque.appendleft)
    >>> list(map(bs, 'abcdefgh'))
    ['a', 'ba', 'cba', 'dcba', 'edcb', 'fedc', 'gfed', 'hgfe']

    With ``add_new_val=deque.extend``, data can be fed in chunks.
    In the following, also see how we use iterize to get a function that
    takes an iterator and returns an iterator

    >>> from lined import iterize
    >>> window_stats = iterize(BufferStats(
    ... maxlen=4, func=''.join, add_new_val=deque.extend))
    >>> chks = ['a', 'bc', 'def', 'gh']
    >>> for x in window_stats(chks):
    ...     print(x)
    a
    abc
    cdef
    efgh

    Note: To those who might think that they can optimize this for special
    cases: Yes you can.
    But SHOULD you? Is it worth the increase in complexity and reduction in
    flexibility?
    See https://github.com/thorwhalen/umpyre/blob/master/misc
    /performance_of_rolling_window_stats.md

    """

    # __name__ = 'BufferStats'

    def __init__(
        self,
        values=(),
        maxlen: int = _no_value_specified_sentinel,
        func: Callable = sum,
        add_new_val: Callable = deque.append,
    ):
        """

        :param maxlen: Size of the buffer
        :param func: The function to be computed (on buffer contents) and
        returned when buffer is "called"
        :param add_new_val: The function that adds values on the buffer.
        Signature must be (self, new_val)
            Is usually a deque method (``deque.append`` by default, but could
            be ``deque.extend``, ``deque.appendleft`` etc.).
            Can also be any other function that
            has a valid (self, new_val) signature.
        """
        if maxlen is _no_value_specified_sentinel:
            raise TypeError("You are required to specify maxlen")
        if not isinstance(maxlen, int):
            raise TypeError(f"maxlen must be an integer, was: {maxlen}")

        super().__init__(values, maxlen=maxlen)
        self.func = func
        if isinstance(add_new_val, str):
            # assume add_new_val is a method of deque:
            add_new_val = getattr(self, add_new_val)
        self.add_new_val = add_new_val
        self.__name__ = "BufferStats"

    def __call__(self, new_val) -> Stats:
        self.add_new_val(self, new_val)  # add the new value
        return self.func(self)


def is_not_none(x):
    return x is not None


def return_buffer_on_stats_condition(
    stats: Stats, buffer: Iterable, cond: Callable = is_not_none, else_val=None
):
    """

    >>> return_buffer_on_stats_condition(stats=3, buffer=[1,2,3,4], cond=lambda x: x%2 == 1)
    [1, 2, 3, 4]
    >>> return_buffer_on_stats_condition(stats=3, buffer=[1,2,3,4], cond=lambda x: x%2 == 0, else_val='3 is not even!')
    '3 is not even!'
    """

    if cond(stats):
        return buffer
    else:
        return else_val


# @add_name
@dataclass
class Segmenter:
    """

    >>> gen = iter(range(200))
    >>> bs = BufferStats(maxlen=10, func=sum)
    >>> return_if_stats_is_odd = partial(return_buffer_on_stats_condition, cond=lambda x: x%2 == 1, else_val='The sum is not odd!')
    >>> seg = Segmenter(buffer=bs, stats_buffer_callback=return_if_stats_is_odd)
    >>> seg(new_val=1) # since the sum of the values in the buffer [1] is odd, the buffer is returned
    [1]

    Adding 1 + 2 is still odd so:

    >>> seg(new_val=2)
    [1, 2]

    Now since 1 + 2 + 5 is even, the else_val of return_if_stats_is_odd is returned instead

    >>> seg(new_val=5)
    'The sum is not odd!'
    """

    buffer: BufferStats
    stats_buffer_callback: Callable[
        [Stats, Iterable], Any
    ] = return_buffer_on_stats_condition
    __name__ = "Segmenter"

    def __call__(self, new_val):
        stats = self.buffer(new_val)
        return self.stats_buffer_callback(stats, list(self.buffer))
