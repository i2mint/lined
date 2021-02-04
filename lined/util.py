from functools import partial

writable_function_dunders = {
    '__annotations__',
    '__call__',
    '__defaults__',
    '__dict__',
    '__doc__',
    '__globals__',
    '__kwdefaults__',
    '__name__',
    '__qualname__'}


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
        if name == '<lambda>':
            return unnamed_func_name()
        return name
    except AttributeError:
        return unnamed_func_name()
