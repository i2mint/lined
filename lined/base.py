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

from typing import Callable, Dict, Iterable

# MultiFuncSpec = Dict[str, Callable]
MultiFunc = Callable[[Dict], Dict]


class Pipeline:
    """
    >>> p = Pipeline(lambda x: x[0] + x[1],
    ...              lambda z: str(z))
    >>> p((2, 3))
    '5'
    """

    def __init__(self, *funcs: Iterable[Callable]):
        self.funcs = funcs

    def __call__(self, x):
        funcs = iter(self.funcs)
        first_func = next(funcs)
        y = first_func(x)
        for func in funcs:
            y = func(y)
        return y


def mk_multi_func(**named_func) -> MultiFunc:
    """Make a multi-channel function from a {name: func, ...} specification.

    >>> multi_func = mk_multi_func(say_hello=lambda x: f"hello {x}", say_goodbye=lambda x: f"goodby {x}")
    >>> multi_func({'say_hello': 'world', 'say_goodbye': 'Lenin'})
    {'say_hello': 'hello world', 'say_goodbye': 'goodby Lenin'}

    :param spec: A map between a name (str) and a function associated to that name
    :return: A function that takes a dict as an (multi-channel) input and a dict as a (multi-channel) output
    """

    def multi_func(d: dict):
        def gen():
            for key, func in named_func.items():
                yield key, func(d[key])

        return dict(gen())

    return multi_func
