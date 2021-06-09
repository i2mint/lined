from functools import partial
from typing import Callable

writable_function_dunders = {
    '__annotations__',
    '__call__',
    '__defaults__',
    '__dict__',
    '__doc__',
    '__globals__',
    '__kwdefaults__',
    '__name__',
    '__qualname__',
}


def partial_plus(func, *args, **kwargs):
    """Like partial, but with the ability to add 'normal function' stuff (name, doc) to the curried function.

    Note: if no writable_function_dunders is specified will just act as the builtin partial (which it calls first).

    >>> def foo(a, b): return a + b
    >>> f = partial_plus(foo, b=2, __name__='bar', __doc__='foo, but with b=2')
    >>> f.__name__
    'bar'
    >>> f.__doc__
    'foo, but with b=2'
    """
    dunders_in_kwargs = writable_function_dunders.intersection(kwargs)

    def gen():
        for dunder in dunders_in_kwargs:
            dunder_val = kwargs.pop(dunder)
            yield dunder, dunder_val

    dunders_to_write = dict(gen())  # will remove dunders from kwargs
    partial_func = partial(func, *args, **kwargs)
    for dunder, dunder_val in dunders_to_write.items():
        setattr(partial_func, dunder, dunder_val)
    return partial_func


def incremental_str_maker(str_format='{:03.f}'):
    """Make a function that will produce a (incrementally) new string at every call."""
    i = 0

    def mk_next_str():
        nonlocal i
        i += 1
        return str_format.format(i)

    return mk_next_str


unnamed_pipeline = incremental_str_maker(str_format='UnnamedPipeline{:03.0f}')
unnamed_func_name = incremental_str_maker(str_format='unnamed_func_{:03.0f}')


def func_name(func):
    """The func.__name__ of a callable func, or makes and returns one if that fails.
    To make one, it calls unamed_func_name which produces incremental names to reduce the chances of clashing"""
    try:
        name = func.__name__
        if name == '<lambda>':
            return unnamed_func_name()
        return name
    except AttributeError:
        return unnamed_func_name()


def dot_to_ascii(dot: str, fancy: bool = True):
    """Convert a dot string to an ascii rendering of the diagram.

    Needs a connection to the internet to work.


    >>> graph_dot = '''
    ...     graph {
    ...         rankdir=LR
    ...         0 -- {1 2}
    ...         1 -- {2}
    ...         2 -> {0 1 3}
    ...         3
    ...     }
    ... '''
    >>>
    >>> graph_ascii = dot_to_ascii(graph_dot)  # doctest: +SKIP
    >>>
    >>> print(graph_ascii)  # doctest: +SKIP
    <BLANKLINE>
                     ┌─────────┐
                     ▼         │
         ┌───┐     ┌───┐     ┌───┐     ┌───┐
      ┌▶ │ 0 │ ─── │ 1 │ ─── │   │ ──▶ │ 3 │
      │  └───┘     └───┘     │   │     └───┘
      │    │                 │   │
      │    └──────────────── │ 2 │
      │                      │   │
      │                      │   │
      └───────────────────── │   │
                             └───┘
    <BLANKLINE>

    """
    import requests

    url = 'https://dot-to-ascii.ggerganov.com/dot-to-ascii.php'
    boxart = 0

    # use nice box drawing char instead of + , | , -
    if fancy:
        boxart = 1

    stripped_dot_str = dot.strip()
    if not (
        stripped_dot_str.startswith('graph') or stripped_dot_str.startswith('digraph')
    ):
        dot = 'graph {\n' + dot + '\n}'


    params = {
        'boxart': boxart,
        'src': dot,
    }

    response = requests.get(url, params=params).text

    if response == '':
        raise SyntaxError('DOT string is not formatted correctly')

    return response


# ───────────────────────────────────────────────────────────────────────────────────────

from inspect import signature, Parameter


def param_is_required(param: Parameter) -> bool:
    return param.default == Parameter.empty and param.kind not in {
        Parameter.VAR_POSITIONAL,
        Parameter.VAR_KEYWORD,
    }


def n_required_args(func: Callable) -> int:
    """Number of required arguments.

    A required argument is one that doesn't have a default, nor is VAR_POSITIONAL (*args) or VAR_KEYWORD (**kwargs).
    Note: Sometimes a minimum number of arguments in VAR_POSITIONAL and VAR_KEYWORD are in fact required,
    but we can't see this from the signature, so we can't tell you about that! You do the math.

    >>> n_required_args(lambda x, y, z=None, *args, **kwargs: ...)
    2

    """
    return sum(map(param_is_required, signature(func).parameters.values()))
