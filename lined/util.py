"""Util functions"""

from functools import partial, partialmethod
from typing import Callable
from types import MethodType
import itertools

from inspect import Signature, Parameter, signature


dflt_signature = Signature(
    [
        Parameter(name="args", kind=Parameter.VAR_POSITIONAL),
        Parameter(name="kwargs", kind=Parameter.VAR_KEYWORD),
    ]
)


def signature_from_first_and_last_func(first_func, last_func):
    """Returns a signature using the first_func for it's params and last_func for
    return annotations, both only if possible"""
    try:
        input_params = signature(first_func).parameters.values()
    except ValueError:  # function doesn't have a signature, so take default
        input_params = dflt_signature.parameters.values()
    try:
        return_annotation = signature(last_func).return_annotation
    except ValueError:  # function doesn't have a signature, so take default
        return_annotation = dflt_signature.return_annotation
    return Signature(input_params, return_annotation=return_annotation)


writable_function_dunders = {
    "__annotations__",
    "__call__",
    "__defaults__",
    "__dict__",
    "__doc__",
    "__globals__",
    "__kwdefaults__",
    "__name__",
    "__qualname__",
}


def partial_plus(func, *args, **kwargs):
    """Like partial, but with the ability to add 'normal function' stuff
    (name, doc) to the curried function.

    Note: if no writable_function_dunders is specified will just act as the
    builtin partial (which it calls first).

    >>> def foo(a, b): return a + b
    >>> f = partial_plus(foo, b=2, __name__='bar', __doc__='foo, but with b=2')
    >>> f.__name__
    'bar'
    >>> f.__doc__
    'foo, but with b=2'
    """
    kwargs["__name__"] = kwargs.get("__name__", func_name(func))

    dunders_in_kwargs = writable_function_dunders.intersection(kwargs)

    def gen():
        for dunder in dunders_in_kwargs:
            dunder_val = kwargs.pop(dunder)
            yield dunder, dunder_val

    dunders_to_write = dict(gen())  # will remove dunders from kwargs

    # partial_func = CachedInstancePartial(func, *args, **kwargs)
    partial_func = partial(func, *args, **kwargs)

    for dunder, dunder_val in dunders_to_write.items():
        setattr(partial_func, dunder, dunder_val)

    return partial_func


def incremental_str_maker(str_format="{:03.f}"):
    """Make a function that will produce a (incrementally) new string at every call."""
    i = 0

    def mk_next_str():
        nonlocal i
        i += 1
        return str_format.format(i)

    return mk_next_str


unnamed_pipeline = incremental_str_maker(str_format="UnnamedPipeline{:03.0f}")
unnamed_func_name = incremental_str_maker(str_format="unnamed_func_{:03.0f}")


def func_name(func):
    """The func.__name__ of a callable func, or makes and returns one if that fails.
    To make one, it calls unamed_func_name which produces incremental names to reduce the chances of clashing"""
    try:
        name = func.__name__
        if name == "<lambda>":
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

    url = "https://dot-to-ascii.ggerganov.com/dot-to-ascii.php"
    boxart = 0

    # use nice box drawing char instead of + , | , -
    if fancy:
        boxart = 1

    stripped_dot_str = dot.strip()
    if not (
        stripped_dot_str.startswith("graph") or stripped_dot_str.startswith("digraph")
    ):
        dot = "graph {\n" + dot + "\n}"

    params = {
        "boxart": boxart,
        "src": dot,
    }

    try:
        response = requests.get(url, params=params).text
    except requests.exceptions.ConnectionError:
        return "ConnectionError: You need the internet to convert dot into ascii!"

    if response == "":
        raise SyntaxError("DOT string is not formatted correctly")

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


# ───────────────────────────────────────────────────────────────────────────────────────
# Vendorized from boltons (https://github.com/mahmoud/boltons)

make_method = lambda desc, obj, obj_type: MethodType(desc, obj)


def mro_items(type_obj):
    """Takes a type and returns an iterator over all class variables
    throughout the type hierarchy (respecting the MRO).

    >>> sorted(set([k for k, v in mro_items(int) if not k.startswith('__')
    ...     and 'bytes' not in k and not callable(v)]))
    ['denominator', 'imag', 'numerator', 'real']
    """
    # TODO: handle slots?
    return itertools.chain.from_iterable(ct.__dict__.items() for ct in type_obj.__mro__)


class CachedInstancePartial(partial):
    """The ``CachedInstancePartial`` is virtually the same as
    :class:`InstancePartial`, adding support for method-usage to
    :class:`functools.partial`, except that upon first access, it
    caches the bound method on the associated object, speeding it up
    for future accesses, and bringing the method call overhead to
    about the same as non-``partial`` methods.

    See the :class:`InstancePartial` docstring for more details.
    """

    def __get__(self, obj, obj_type):
        # These assignments could've been in __init__, but there was
        # no simple way to do it without breaking one of PyPy or Py3.
        self.__name__ = None
        self.__doc__ = self.func.__doc__
        self.__module__ = self.func.__module__

        name = self.__name__
        if name is None:
            for k, v in mro_items(obj_type):
                if v is self:
                    self.__name__ = name = k
        if obj is None:
            return make_method(self, obj, obj_type)
        try:
            # since this is a data descriptor, this block
            # is probably only hit once (per object)
            return obj.__dict__[name]
        except KeyError:
            obj.__dict__[name] = ret = make_method(self, obj, obj_type)
            return ret
