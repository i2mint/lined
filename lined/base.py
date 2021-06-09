"""
The base objects of lined.
"""

from functools import wraps
from typing import Callable, Dict, Iterable, Optional, Union, Mapping
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


_line_init_reserved_names = {'pipeline_name', 'input_name', 'output_name'}


def _func_to_name_func_pair(func):
    """The func.__name__ of a callable func, or makes and returns one if that fails.
    To make one, it calls unamed_func_name which produces incremental names to reduce the chances of clashing"""
    if isinstance(func, tuple) and len(func) == 2:
        name, func = func
        return name, func
    else:
        return func_name(func), func


def _merge_funcs_and_named_funcs(funcs, named_funcs):
    """Add the funcs of named_funcs to funcs tuple and visa versa, making two aligned collections of functions."""
    assert _line_init_reserved_names.isdisjoint(
        named_funcs
    ), f"Can't name a function with any of the following strings: {', '.join(_line_init_reserved_names)}"
    funcs_obtained_from_named_funcs = tuple(named_funcs.values())
    named_funcs_obtained_from_funcs = dict(map(_func_to_name_func_pair, funcs))
    assert named_funcs_obtained_from_funcs.keys().isdisjoint(
        named_funcs
    ), f"Some names clashed: {', '.join(set(named_funcs_obtained_from_funcs).intersection(named_funcs))}"
    return (
        funcs + funcs_obtained_from_named_funcs,
        dict(named_funcs_obtained_from_funcs, **named_funcs),
    )


# TODO: Deprecate of named_funcs? Use (name, func) mechanism only?
# TODO: add validation option (e.g. all downstream functions single-argumented)
# TODO: Handle names with spaces
# TODO: Use .name instead of __name__ for pipeline_name?
class Line:
    def __init__(
        self,
        *funcs: Funcs,
        pipeline_name=None,
        input_name=None,
        output_name=None,
        **named_funcs,
    ):
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
        :param pipeline_name: The name of the pipeline
        :param input_name: The name of an input
        :param output_name: The name of an output
        A really simple example:

        >>> p = Line(sum, str)
        >>> p([2, 3])
        '5'

        A still quite simple example:

        >>> def first(a, b=1):
        ...     return a * b
        >>>
        >>> def last(c) -> float:
        ...     return c + 10
        >>>
        >>> f = Line(first, last)
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

        >>> same_as_first = Line(first)
        >>> assert same_as_first(42) == first(42)

        >>> from functools import partial
        >>> pipe = Line(sum, str, print, pipeline_name='MyPipeline', input_name='x', output_name='y')
        >>> pipe
        MyPipeline(sum, str, print, input_name='x', output_name='y')


        """
        funcs, named_funcs = _merge_funcs_and_named_funcs(funcs, named_funcs)

        self.input_name = input_name
        self.output_name = output_name
        # really, it would make sense that this is the identity, but we'll implement only when needed
        assert len(funcs) > 0, 'You need to specify at least one function!'
        self.funcs = tuple(fnode(func, name) for name, func in named_funcs.items())
        self.named_funcs = {
            name: fnode_func for name, fnode_func in zip(named_funcs, self.funcs)
        }
        assert all(
            f == ff for f, ff in zip(self.funcs, self.named_funcs.values())
        ), f'funcs and named_funcs are not aligned after merging'
        if pipeline_name is not None:
            self.__name__ = pipeline_name
        self.__signature__ = _signature_of_pipeline(*self.funcs)

    # Note: Did this to lighten __init__, but made signature(Line) not work
    # @property
    # def __signature__(self):
    #     return _signature_of_pipeline(*self.funcs)

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
            item_str = ''
            funcs = ()
            if isinstance(k, int):
                funcs = (self.funcs[k],)
                item_str = str(k)
            elif isinstance(k, slice):
                assert k.step is None, f'slices with steps are not handled: {k}'
                funcs = self.funcs[k]
                item_str = f'{k.start}:{k.stop}'
            return self.__class__(*funcs, name=f'{self.__name__}[item_str]')
        else:
            raise TypeError(f"Don't know how to handle that type of key: {k}")

    def dot_digraph_body(
        self,
        prefix=None,
        fnode_shape='box',
        vnode_shape='none',
        input_node=True,
        output_node=False,
        edges_gen=True,
        **kwargs,
    ):

        if len(self.funcs) == 0:
            return  # no functions, so just return

        if prefix is None:
            if len(self.funcs) <= 7:
                yield 'rankdir="LR"'
        else:
            yield prefix

        func_names = list(self.named_funcs)

        if input_node:
            if input_node is True:
                for argname in signature(self.funcs[0]).parameters:
                    yield f'{argname} [shape="{vnode_shape}"]'
                    yield f'{argname} -> {func_names[0]}'
            elif input_node == str:
                input_node = self.input_name or 'input'
                yield f'{input_node} [shape="{vnode_shape}"]'
                yield f'{input_node} -> {func_names[0]}'

        for fname in func_names:
            yield f'{fname} [shape="{fnode_shape}"]'

        if edges_gen:
            if edges_gen is True:
                for from_fname, to_fname in zip(func_names[:-1], func_names[1:]):
                    yield f'{from_fname} -> {to_fname}'
            else:
                yield from edges_gen

        if output_node is None and self.output_name is not None:
            output_node = self.output_name
        if output_node:
            if output_node is True:
                output_node = self.output_name or 'output'
            yield f'{output_node} [shape="{vnode_shape}"]'
            yield f'{func_names[-1]} -> {self.output_name}'

    @wraps(dot_digraph_body)
    def dot_digraph_ascii(self, *args, **kwargs):
        """Get an ascii art string that represents the pipeline"""
        from lined.util import dot_to_ascii

        return dot_to_ascii('\n'.join(self.dot_digraph_body(*args, **kwargs)))

    @wraps(dot_digraph_body)
    def dot_digraph(self, *args, **kwargs):
        try:
            import graphviz
        except (ModuleNotFoundError, ImportError) as e:
            raise ModuleNotFoundError(
                f'{e}\nYou may not have graphviz installed. '
                f'See https://pypi.org/project/graphviz/.'
            )

        body = list(self.dot_digraph_body(*args, **kwargs))
        return graphviz.Digraph(body=body)

    def __repr__(self):
        funcs_str = ', '.join((fname for fname in self.named_funcs))
        suffix = ''
        if self.input_name is not None:
            suffix += f", input_name='{self.input_name}'"
        if self.output_name is not None:
            suffix += f", output_name='{self.output_name}'"
        return f'{self._name_of_instance()}({funcs_str}{suffix})'

    def _name_of_instance(self):
        return getattr(self, '__name__', self.__class__.__name__)


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
        assert isinstance(
            condition, Callable
        ), f'condition need to be callable, but was {condition}'

        def filt(x):
            if condition(x):
                return x
            else:
                return cls(sentinel_val)

        return filt

    @classmethod
    def filter_out(cls, condition, sentinel_val=None):
        assert isinstance(
            condition, Callable
        ), f'condition need to be callable, but was {condition}'

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


# ---------------------------------------------------------------------------------------

from i2.signatures import call_forgivingly, Sig


class NoSuchConfig:
    def __bool__(self):
        return False

    def __repr__(self):
        return 'no_such_config'


no_such_config = NoSuchConfig()


class Configs(dict):
    """A dict, whose keys can be access as if they were attributes.
    Also defaults to sentinel _default when

    >>> s = Configs()

    Write it as you do with attributes or dict keys,
    get it as an attribute and a dict keys.

    >>> s.foo = 'bar'
    >>> assert s.foo == 'bar'
    >>> assert s['foo'] == 'bar'
    >>> s['hello'] = 'world'
    >>> assert s.hello == 'world'
    >>> assert s['hello'] == 'world'
    >>> hasattr(s, 'hello')
    True

    >>> s['does not exist']
    no_such_config
    """

    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError('No such attribute: ' + name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError('No such attribute: ' + name)

    def __missing__(self, name):
        return self._key_missing_callback()

    def _key_missing_callback(self):
        """Override if needed. Can also just raise to raise KeyError"""
        return no_such_config


from collections import ChainMap

dflt_configs = dict(
    fnode_shape='box',
    vnode_shape='none',
    display_all_arguments=True,
    edge_kind='to_args_on_edge',
    input_node=True,
    output_node='output',
)


class LineParametrized(Line):
    r"""A pipeline that exposes all inputs of all the functions of the pipeline.

    For example, say you have two functions `f(a, b=1)` and `g(x, y=2)`.

    `Line(f, g)` would be a function with `a` and `b` as your inputs.

    ```
            ┌───┐     ┌───┐
     a  ──▶ │ f │ ──▶ │ g │ ──▶  output
            └───┘     └───┘
              ▲
              │
              │

              b
    ```

    On the other hand, `LineParametrized(f, g)` will give you control over `y`.

            ┌───┐  x   ┌───┐
     a  ──▶ │ f │ ───▶ │ g │ ──▶  output
            └───┘      └───┘
              ▲          ▲
              │          │
              │          │

              b         y=

    >>> def add(a, b=0): return a + b
    >>> def times(x, y=2): return x * y
    >>> def exp(r, e=3):  return r ** e
    >>> f = LineParametrized(add, times, exp)
    >>> assert f(2) == 64  # as before
    >>> assert f(2, 3) == 1000  # as before, but now you can do this...
    >>> assert f(2, 3, y=1) == 125  # ((2 + 3) * 1) ** 3 == 125
    >>> assert f(2, 3, e=1) == 10  # ((2 + 3) * 2) ** 1 == 10
    >>> assert f(2, 3, y=3, e=1) == 15  # ((2 + 3) * 3) ** 1 == 15
    >>> assert f(2, y=10, e=1) == 20  # (skipping b here! using default b=0) ((2 + 0) * 10) ** 1 == 20

    >>> from inspect import signature
    >>> signature(f)
    <Sig (a, x, r, b=0, y=2, e=3)>

    >>> print(f.dot_digraph_ascii())
            ┌─────┐  x   ┌───────┐  r   ┌─────┐
     a  ──▶ │ add │ ───▶ │ times │ ───▶ │ exp │ ──▶  output
            └─────┘      └───────┘      └─────┘
              ▲            ▲              ▲
              │            │              │
              │            │              │
    <BLANKLINE>
               b            y=            e=
    <BLANKLINE>

    >>> print(f.dot_digraph_ascii(edge_kind='simple'))
            ┌─────┐     ┌───────┐     ┌─────┐
     a  ──▶ │ add │ ──▶ │ times │ ──▶ │ exp │ ──▶  output
            └─────┘     └───────┘     └─────┘
              ▲
              │
              │
    <BLANKLINE>
               b
    <BLANKLINE>

    >>> print(f.dot_digraph_ascii(edge_kind='simple_on_edge'))
            ┌─────┐  x   ┌───────┐  r   ┌─────┐
     a  ──▶ │ add │ ───▶ │ times │ ───▶ │ exp │ ──▶  output
            └─────┘      └───────┘      └─────┘
              ▲
              │
              │
    <BLANKLINE>
               b
    <BLANKLINE>

    >>> print(f.dot_digraph_ascii(edge_kind='to_args'))
    <BLANKLINE>
                            x            r
    <BLANKLINE>
                          │             │
                          │             │
                          ▼             ▼
            ┌─────┐     ┌───────┐     ┌─────┐
     a  ──▶ │ add │ ──▶ │ times │ ──▶ │ exp │ ──▶  output
            └─────┘     └───────┘     └─────┘
              ▲           ▲             ▲
              │           │             │
              │           │             │
    <BLANKLINE>
               b           y=           e=
    <BLANKLINE>
    """

    @wraps(Line.__init__)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__signature__ = Sig.from_objs(*self.funcs)

    def __call__(self, *args, **kwargs):
        first_func, *other_funcs = self.funcs
        out = call_forgivingly(first_func, *args, **kwargs)
        for func in other_funcs:
            out = call_forgivingly(func, out, **kwargs)
        return out

    # @property
    # def __signature__(self):
    #     return Sig.from_objs(*self.funcs)

    # TODO: Try merging Line.dot_diagraph_body and this, for reuse
    def dot_digraph_body(
        self, prefix=None, edge_kind='to_args_on_edge', convention=None, **kwargs
    ):

        c = Configs(
            ChainMap(
                convention or {},
                dict(kwargs, edge_kind=edge_kind, prefix=prefix),
                dflt_configs,
            )
        )
        if prefix is None:
            if len(self.funcs) <= 7:
                yield 'rankdir="LR"'
        else:
            yield prefix

        func_names = list(self.named_funcs)

        if c.input_node:
            if c.input_node is True:
                for argname in signature(self.funcs[0]).parameters:
                    yield f'{argname} [shape="{c.vnode_shape}"]'
                    yield f'{argname} -> {func_names[0]}'
            elif c.input_node == str:
                input_node = self.input_name or 'input'
                yield f'{input_node} [shape="{c.vnode_shape}"]'
                yield f'{input_node} -> {func_names[0]}'

        for i, (fname, func) in enumerate(self.named_funcs.items()):
            yield f'{fname} [shape="{c.fnode_shape}"]'

            sig = Sig(func)
            if i > 0:
                first_arg = next(iter(sig.names), None)

                on_edge = c.edge_kind.endswith('on_edge')
                if on_edge:
                    yield f'{func_names[i - 1]} -> {fname} [label="{first_arg}"]'
                else:
                    yield f'{func_names[i - 1]} -> {fname}'

                if c.edge_kind.startswith('to_args'):
                    if c.display_all_arguments:
                        for i, argname in enumerate(sig.names):
                            if on_edge and i == 0:
                                continue  # skip first arg if on_edge mode
                            if argname not in sig.defaults:
                                yield f'{argname} [shape="{c.vnode_shape}"]'
                            else:
                                argname_with_equals = argname + '='
                                yield (
                                    f'{argname} [shape="{c.vnode_shape}" label="'
                                    f'{argname_with_equals}"]'
                                )
                            yield f'{argname} -> {fname}'

        if c.output_node:
            yield f'{c.output_node} [shape="{c.vnode_shape}"]'
            yield f'{func_names[-1]} -> {c.output_node}'

    @wraps(dot_digraph_body)
    def dot_digraph_ascii(self, *args, **kwargs):
        """Get an ascii art string that represents the pipeline"""
        return super().dot_digraph_ascii(*args, **kwargs)

    @wraps(dot_digraph_body)
    def dot_digraph(self, *args, **kwargs):
        return super().dot_digraph(*args, **kwargs)


# ---------------------------------------------------------------------------------------


class LineSentineled(Line):
    """A pipeline that can be interrupted by a sentinel.

    Sentinels are useful to interrupt the pipeline computation.

    Say, for example, you know if the length of an input iterable divided by three is
    1 or 2.
    You wouldn't want to divide by 0 or have a loop choke on an input that doesn't
    have a length.

    So you do this:

    >>> pipe = LineSentineled(
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


SentineledPipeline = LineSentineled  # back-compatibility alias


def inject_names_if_missing(funcs):
    for func in funcs:
        func.__name__ = func_name(func)
    return funcs


def stack(*funcs):
    def call(func, arg):
        return func(arg)

    def stacked_funcs(input_tuple):
        assert len(funcs) == len(input_tuple), (
            'the length of input_tuple ({len(input_tuple)} should be the same length'
            ' (len{funcs}) as the funcs: {input_tuple}'
        )
        return tuple(starmap(call, zip(funcs, input_tuple)))

    return stacked_funcs


# TODO: Need tests and the new args of Line (named_funcs...)
class LayeredPipeline(Line):
    def __init__(self, *funcs: LayeredFuncs, pipeline_name=None):
        def _funcs():
            for func in funcs:
                if isinstance(func, Callable):
                    yield func
                elif isinstance(func, (list, tuple, set)):
                    yield LayeredPipeline(*func)
                else:
                    raise ValueError(f"Don't know how to deal with this func: {func}")

        super().__init__(*_funcs(), pipeline_name=pipeline_name)


def _signature_of_pipeline(*funcs):
    n_funcs = len(funcs)
    if n_funcs == 0:
        raise ValueError('You need to specify at least one function!')
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

    Here's a more significant example.

    >>> chunkers = {'a': lambda x: x[0] + x[1],
    ...             'b': lambda x: x[0] * x[1]}
    >>> featurizers = {'a': lambda z: str(z),
    ...                'b': lambda z: [z] * 3}
    >>> multi_chunker = mk_multi_func(**chunkers)
    >>> multi_chunker({'a': (1, 2), 'b': (3, 4)})
    {'a': 3, 'b': 12}
    >>> multi_featurizer = mk_multi_func(**featurizers)
    >>> multi_featurizer({'a': 3, 'b': 12})
    {'a': '3', 'b': [12, 12, 12]}
    >>> my_pipe = Line(multi_chunker, multi_featurizer)
    >>> my_pipe({'a': (1, 2), 'b': (3, 4)})
    {'a': '(1, 2)', 'b': [(3, 4), (3, 4), (3, 4)]}
    """

    named_funcs_dict = named_funcs_dict or {}
    assert named_funcs_dict.keys().isdisjoint(
        named_funcs
    ), f"named_funcs_dict and named_funcs can't share keys. Yet they share {named_funcs_dict.keys() & named_funcs}"
    named_funcs = dict(named_funcs_dict, **named_funcs)
    assert all(
        map(callable, named_funcs.values())
    ), f'At least one value of named_funcs was not callable'

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
        self.nodes_adjacent_to = defaultdict(
            list, nodes_adjacent_to
        )  # adjacency list (look it up)
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
