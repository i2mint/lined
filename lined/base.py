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

from functools import wraps
from typing import Callable, Dict, Iterable, Optional, Union
from inspect import Signature, signature
from itertools import starmap
from dataclasses import dataclass

from lined.util import unnamed_pipeline, func_name

# MultiFuncSpec = Dict[str, Callable]
MultiFunc = Callable[[Dict], Dict]
Funcs = Union[Iterable[Callable], Callable]
LayeredFuncs = Iterable[Funcs]


# def fnode(func, name=None):
#     @wraps(func)
#     def func_node(*args, **kwargs):
#         return func(*args, **kwargs)
#
#     func_node.__name__ = name or func_name(func)
#     return func_node


@dataclass
class Fnode:
    func: Callable
    __name__: Optional[str] = None

    def __post_init__(self):
        wraps(self.func)(self)
        self.__name__ = self.__name__ or func_name(self.func)

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)


def fnode(func, name=None):
    return Fnode(func, name)


class Line:
    def __init__(self, *funcs: Funcs, name=None, input_name=None, output_name=None):
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

        :param funcs: The functions of the pipeline
        :param name: The name of the pipeline
        :param input_name: The name of an input
        :param output_name: The name of an output
        A really simple example:

        >>> p = Pipeline(sum, str)
        >>> p([2, 3])
        '5'

        A still quite simple example:

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


        >>> from functools import partial
        >>> pipe = Pipeline(sum, str, print, name='MyPipeline', input_name='x', output_name='y')
        >>> pipe
        Pipeline(sum, str, print, name='MyPipeline', input_name='x', output_name='y')

        """
        self.funcs = funcs
        self.input_name = input_name
        self.output_name = output_name
        # really, it would make sense that this is the identity, but we'll implement only when needed
        assert len(self.funcs) > 0, "You need to specify at least one function!"
        self.funcs = tuple(map(fnode, self.funcs))
        self.__signature__ = _signature_of_pipeline(*self.funcs)
        if name is not None:
            self.__name__ = name
        else:
            self.__name__ = unnamed_pipeline()

    def __repr__(self):
        funcs_str = ', '.join((f.__name__ for f in self.funcs))
        suffix = ''
        if self.input_name is not None:
            suffix += f", input_name='{self.input_name}'"
        if self.output_name is not None:
            suffix += f", output_name='{self.output_name}'"
        return f"{self.__class__.__name__}({funcs_str}, name='{self.__name__}'{suffix})"

    def __call__(self, *args, **kwargs):
        first_func, *other_funcs = self.funcs
        out = first_func(*args, **kwargs)
        for func in other_funcs:
            out = func(out)
        return out

    def __len__(self):
        return len(self.funcs)

    def __getitem__(self, k):
        """Get a sub-pipeline"""
        if isinstance(k, (int, slice)):
            item_str = ""
            funcs = ()
            if isinstance(k, int):
                funcs = (self.funcs[k],)
                item_str = str(k)
            elif isinstance(k, slice):
                assert k.step is None, f"slices with steps are not handled: {k}"
                funcs = self.funcs[k]
                item_str = f'{k.start}:{k.stop}'
            return self.__class__(*funcs, name=f"{self.__name__}[item_str]")
        else:
            raise TypeError(f"Don't know how to handle that type of key: {k}")

    def dot_digraph_body(self, prefix=None, **kwargs):
        fnode_shape = kwargs.get('fnode_shape', "box")
        vnode_shape = kwargs.get('fnode_shape', "oval")

        if prefix is None:
            if len(self.funcs) <= 7:
                yield 'rankdir="LR"'
        else:
            yield prefix

        if self.input_name is not None:
            yield f'{self.input_name} [shape="circle"]'
            yield f'{self.input_name} -> {self.funcs[0].__name__}'

        for f in self.funcs:
            yield f'{f.__name__} [shape="{fnode_shape}"]'

        for f, ff in zip(self.funcs[:-1], self.funcs[1:]):
            yield f'{f.__name__} -> {ff.__name__}'

        if self.output_name is not None:
            yield f'{self.output_name} [shape="{vnode_shape}"]'
            yield f'{self.funcs[-1].__name__} -> {self.output_name}'

    def dot_digraph(self, prefix=None, **kwargs):
        try:
            import graphviz
        except (ModuleNotFoundError, ImportError) as e:
            raise ModuleNotFoundError(f"{e}\nYou may not have graphviz installed. "
                                      f"See https://pypi.org/project/graphviz/.")

        body = list(self.dot_digraph_body(prefix=prefix, **kwargs))
        return graphviz.Digraph(body=body)


Pipeline = Line  # for back-compatibility

from dataclasses import dataclass
from typing import Any


@dataclass
class Sentinel:
    """To make sentinels holding an optional value"""
    val: Any = None

    @classmethod
    def this_is_not(cls, obj):
        return not isinstance(obj, cls)

    @classmethod
    def filter_in(cls, condition, sentinel_val=None):
        assert isinstance(condition, Callable), f"condition need to be callable, but was {condition}"

        def filt(x):
            if condition(x):
                return x
            else:
                return cls(sentinel_val)

        return filt

    @classmethod
    def filter_out(cls, condition, sentinel_val=None):
        assert isinstance(condition, Callable), f"condition need to be callable, but was {condition}"

        def filt(x):
            if not condition(x):
                return x
            else:
                return cls(sentinel_val)

        return filt


class Conditions:
    @staticmethod
    def excluded_val(excluded_val):
        def condition(x):
            return x == excluded_val

        return condition

    @staticmethod
    def exclude_type(excluded_type):
        def condition(x):
            return not isinstance(x, excluded_type)

        return condition

    @staticmethod
    def include_type(excluded_type):
        def condition(x):
            return isinstance(x, excluded_type)

        return condition


class SentineledPipeline(Pipeline):
    """A pipeline that can be interrupted by a sentinel.

    Sentinels are useful to interrupt the pipeline computation.

    Say, for example, you know if the length of an input iterable divided by three is 1 or 2.
    You wouldn't want to divide by 0 or have a loop choke on an input that doesn't have a length.
    So you do this:

    >>> pipe = SentineledPipeline(
    ...     lambda x: (hasattr(x, '__len__') and x) or Sentinel('no length'), # returns x if it has a length, and None if not
    ...     len,
    ...     lambda x: x % 3,
    ... )
    >>> pipe([1,2,3,4])
    1
    >>> pipe(1)
    Sentinel(val='no length')
    >>> # which allows us to do things like:
    >>> list(filter(Sentinel.this_is_not, map(pipe, [[1,2,3,4], None, 1, [1,2,3]])))
    [1, 0]
    """

    def __call__(self, *args, **kwargs):
        first_func, *other_funcs = self.funcs
        out = first_func(*args, **kwargs)
        if not isinstance(out, Sentinel):
            for func in other_funcs:
                out = func(out)
                if isinstance(out, Sentinel):
                    break
        return out


def inject_names_if_missing(funcs):
    for func in funcs:
        func.__name__ = func_name(func)
    return funcs


def stack(*funcs):
    def call(func, arg):
        return func(arg)

    def stacked_funcs(input_tuple):
        assert len(funcs) == len(input_tuple), \
            "the length of input_tuple ({len(input_tuple)} should be the same length" \
            " (len{funcs}) as the funcs: {input_tuple}"
        return tuple(starmap(call, zip(funcs, input_tuple)))

    return stacked_funcs


class LayeredPipeline(Pipeline):
    def __init__(self, *funcs: LayeredFuncs, name=None):
        def _funcs():
            for func in funcs:
                if isinstance(func, Callable):
                    yield func
                elif isinstance(func, (list, tuple, set)):
                    yield LayeredPipeline(*func)
                else:
                    raise ValueError(f"Don't know how to deal with this func: {func}")

        super().__init__(*_funcs(), name=name)


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


from collections import defaultdict


# Class to represent a graph
class Digraph:
    def __init__(self, nodes_adjacent_to=None):
        nodes_adjacent_to = nodes_adjacent_to or dict()
        self.nodes_adjacent_to = defaultdict(list, nodes_adjacent_to)  # adjacency list (look it up)
        # self.n_vertices = vertices  # No. of vertices

    # function to add an edge to graph
    def add_edge(self, u, v):
        self.nodes_adjacent_to[u].append(v)

        # A recursive function used by topologicalSort

    def _helper(self, v, visited, stack):

        # Mark the current node as visited.
        visited[v] = True

        # Recur for all the vertices adjacent to this vertex
        for i in self.nodes_adjacent_to[v]:
            if visited[i] == False:
                self._helper(i, visited, stack)

                # Push current vertex to stack which stores result
        stack.insert(0, v)

        # The function to do Topological Sort. It uses recursive

    # topologicalSortUtil()
    def topological_sort(self):
        # Mark all the vertices as not visited
        visited = [False] * self.n_vertices
        stack = []

        # Call the recursive helper function to store Topological
        # Sort starting from all vertices one by one
        for i in range(self.n_vertices):
            if visited[i] == False:
                self._helper(i, visited, stack)

                # Print contents of stack
        return stack
