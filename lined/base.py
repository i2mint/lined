"""

>>> chunkers = {'a': lambda x: x[0] + x[1],
...             'b': lambda x: x[0] * x[1]}
>>> featurizers = {'a': lambda z: str(z),
...                'b': lambda z: [z] * 3}
>>> multi_chunker = mk_multi_func(**chunkers)
>>> assert multi_chunker({'a': (1, 2), 'b': (3, 4)}) == {'a': 3, 'b': 12}
>>> multi_featurizer = mk_multi_func(**featurizers)
>>> assert multi_featurizer({'a': 3, 'b': 12}) == {'a': '3', 'b': [12, 12, 12]}
>>> my_pipe = Pipeline(multi_chunker, multi_featurizer)
>>> assert my_pipe({'a': (1, 2), 'b': (3, 4)}) == {'a': '3', 'b': [12, 12, 12]}
"""

from typing import Callable, Dict, Iterable, Optional
from inspect import Signature, signature

# MultiFuncSpec = Dict[str, Callable]
MultiFunc = Callable[[Dict], Dict]


class Pipeline:
    def __init__(self, *funcs: Iterable[Callable]):
        """Performs function composition.
        That is, get a callable that is equivalent to a chain of callables.
        For example, if `f`, `h`, and `g` are three functions, the function
        ```
            c = Compose(f, h, g)
        ```
        is such that, for any valid inputs `args, kwargs` of `f`,
        ```
        c(*args, **kwargs) == g(h(f(*args, **kwargs)))
        ```
        (assuming the functions are deterministic of course).

        A really simple example:

        >>> p = Pipeline(sum, str)
        >>> p([2, 3])
        '5'

        >>> def first(a, b=1):
        ...     return a * b
        >>>
        >>> def last(c) -> float:
        ...     return c + 10
        >>>
        >>> f = Pipeline(first, last)
        >>>
        >>> assert f(2) == 12
        >>> assert f(2, 10) == 30

        Let's check out the signature of f:

        >>> from inspect import signature
        >>>
        >>> assert str(signature(f)) == '(a, b=1) -> float'
        >>> assert signature(f).parameters == signature(first).parameters
        >>> assert signature(f).return_annotation == signature(last).return_annotation == float

        Border case: One function only

        >>> same_as_first = Pipeline(first)
        >>> assert same_as_first(42) == first(42)
        """
        self.funcs = funcs

        # really, it would make sense that this is the identity, but we'll implement only when needed
        assert len(self.funcs) > 0, "You need to specify at least one function!"

        self.__signature__ = _signature_of_pipeline(*self.funcs)

    def __call__(self, *args, **kwargs):
        first_func, *other_funcs = self.funcs
        out = first_func(*args, **kwargs)
        for func in other_funcs:
            out = func(out)
        return out


def _signature_of_pipeline(*funcs):
    n_funcs = len(funcs)
    if n_funcs == 0:
        raise ValueError("You need to specify at least one function!")
    elif n_funcs == 1:
        first_func = last_func = funcs[0]
    else:
        first_func, *_, last_func = funcs
    # Finally, let's make the __call__ have a nice signature.
    # Argument information from first func and return annotation from last func
    try:
        input_params = signature(first_func).parameters.values()
        try:
            return_annotation = signature(last_func).return_annotation
        except ValueError:
            return_annotation = Signature.empty
        return Signature(input_params, return_annotation=return_annotation)
    except ValueError:
        return None


def mk_multi_func(named_funcs_dict: Optional[Dict] = None, /, **named_funcs) -> MultiFunc:
    """Make a multi-channel function from a {name: func, ...} specification.

    >>> multi_func = mk_multi_func(say_hello=lambda x: f"hello {x}", say_goodbye=lambda x: f"goodbye {x}")
    >>> multi_func({'say_hello': 'world', 'say_goodbye': 'Lenin'})
    {'say_hello': 'hello world', 'say_goodbye': 'goodbye Lenin'}

    :param spec: A map between a name (str) and a function associated to that name
    :return: A function that takes a dict as an (multi-channel) input and a dict as a (multi-channel) output

    Q: Why can I specify the specs both with named_funcs_dict and **named_funcs?
    A: Look at the ``dict(...)`` interface. You see the same thing there.
    Different reason though (here we assert that the keys don't overlap).
    Usually named_funcs is more convenient, but if you need to use keys that are not valid python variable names,
    you can always use named_funcs_dict to express that!

    >>> multi_func = mk_multi_func({'x+y': lambda d: f"sum is {d}", 'x*y': lambda d: f"prod is {d}"})
    >>> multi_func({'x+y': 5, 'x*y': 6})
    {'x+y': 'sum is 5', 'x*y': 'prod is 6'}

    You can also use both. Like with ``dict(...)``.

    """

    named_funcs_dict = named_funcs_dict or {}
    assert named_funcs_dict.keys().isdisjoint(named_funcs), \
        f"named_funcs_dict and named_funcs can't share keys. Yet they share {named_funcs_dict.keys() & named_funcs}"
    named_funcs = dict(named_funcs_dict, **named_funcs)

    def multi_func(d: dict):
        def gen():
            for key, func in named_funcs.items():
                yield key, func(d[key])

        return dict(gen())

    return multi_func
