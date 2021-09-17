"""Simple lightweight utils"""

from inspect import signature, Signature, Parameter

dflt_signature = Signature(
    [
        Parameter(name="args", kind=Parameter.VAR_POSITIONAL),
        Parameter(name="kwargs", kind=Parameter.VAR_KEYWORD),
    ]
)


def signature_from_first_and_last_func(first_func, last_func):
    try:
        input_params = signature(first_func).parameters.values()
    except ValueError:  # function doesn't have a signature, so take default
        input_params = dflt_signature.parameters.values()
    try:
        return_annotation = signature(last_func).return_annotation
    except ValueError:  # function doesn't have a signature, so take default
        return_annotation = dflt_signature.return_annotation
    return Signature(input_params, return_annotation=return_annotation)


def compose(*funcs):
    """

    :param funcs:
    :return:

    >>> def foo(a, b=2):
    ...     return a + b
    >>> f = compose(foo, lambda x: print(f"x: {x}"))
    >>> f(3)
    x: 5

    Notes:
        - composed functions are normal functions (have a __name__ etc.) but are not pickalable. See Pipe for that.
    """

    def composed_funcs(*args, **kwargs):
        out = composed_funcs.first_func(*args, **kwargs)
        for func in composed_funcs.other_funcs:
            out = func(out)
        return out

    n_funcs = len(funcs)
    if n_funcs == 0:
        raise ValueError("You need to specify at least one function!")
    elif n_funcs == 1:
        first_func = last_func = funcs[0]
        other_funcs = ()
    else:
        first_func, *other_funcs = funcs
        last_func = other_funcs[-1]

    composed_funcs.first_func = first_func
    composed_funcs.other_funcs = other_funcs
    composed_funcs.__signature__ = signature_from_first_and_last_func(
        first_func, last_func
    )

    return composed_funcs


# Pipe code is completely independent. If you only need simple pipelines, use this, or even copy/paste it where needed.
# TODO: Give it a __name__ and make it more like a "normal" function so it works well when so assumed
class Pipe:
    """Simple function composition. That is, gives you a callable that implements input -> f_1 -> ... -> f_n -> output.

    >>> def foo(a, b=2):
    ...     return a + b
    >>> f = Pipe(foo, lambda x: print(f"x: {x}"))
    >>> f(3)
    x: 5

    Notes:
        - Pipe instances don't have a __name__ etc. So some expectations of normal functions are not met.
        - Pipe instance are pickalable (as long as the functions that compose them are)
    """

    def __init__(self, *funcs):

        n_funcs = len(funcs)
        other_funcs = ()
        if n_funcs == 0:
            raise ValueError("You need to specify at least one function!")
        elif n_funcs == 1:
            first_func = last_func = funcs[0]
        else:
            first_func, *other_funcs, last_func = funcs

        self.__signature__ = signature_from_first_and_last_func(first_func, last_func)
        self.first_func = first_func
        self.other_funcs = tuple(other_funcs) + (last_func,)

    def __call__(self, *args, **kwargs):
        out = self.first_func(*args, **kwargs)
        for func in self.other_funcs:
            out = func(out)
        return out
