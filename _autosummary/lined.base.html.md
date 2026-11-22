# lined.base

The base objects of lined.

### Functions

| `fnode`(func[, name, first_arg_position_only])                                                     |                                                                 |
|----------------------------------------------------------------------------------------------------|-----------------------------------------------------------------|
| `inject_names_if_missing`(funcs)                                                                   |                                                                 |
| [`log_calls`](#lined.base.log_calls)(line[, logger, what_to_log])            | Log the calls of every step of a pipeline.                      |
| [`name_with_varkind_and_default_marker`](#lined.base.name_with_varkind_and_default_marker)(param)       | Returns a string representation for the Parameter object        |
| `normalize_func`(func)                                                                             |                                                                 |
| `stack`(\*funcs)                                                                                   |                                                                 |
| [`uniquize_funcs_names`](#lined.base.uniquize_funcs_names)(...[, suffic_counter_start]) | Check and make sure that the functions names are all different. |

### Classes

| `Conditions`()                                                                                  |                                                                          |
|-------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------|
| [`Configs`](#lined.base.Configs)                                        | A dict, whose keys can be access as if they were attributes.             |
| [`Digraph`](#lined.base.Digraph)([nodes_adjacent_to])                   | This class is experiemental and will probably move to meshed.            |
| [`Fnode`](#lined.base.Fnode)(func[, name, first_arg_position_only])   | A function wrapper to be used in pipelines.                              |
| `LayeredPipeline`(\*funcs[, pipeline_name])                                                     |                                                                          |
| `Line`(\*funcs[, pipeline_name, input_name, ...])                                               |                                                                          |
| [`LineParametrized`](#lined.base.LineParametrized)(default_conflict_method, ...) | A pipeline that exposes all inputs of all the functions of the pipeline. |
| [`LineSentineled`](#lined.base.LineSentineled)(\*funcs[, pipeline_name, ...])  | A pipeline that can be interrupted by a sentinel.                        |
| `NoSuchConfig`()                                                                                |                                                                          |
| `Pipeline`                                                                                      |                                                                          |
| [`Sentinel`](#lined.base.Sentinel)([val])                                | To make sentinels holding an optional value                              |
| [`SentineledPipeline`](#lined.base.SentineledPipeline)                             |                                                                          |

### *class* lined.base.Configs

Bases: [`dict`](https://docs.python.org/3/builtins/stdtypes.html#dict)

A dict, whose keys can be access as if they were attributes.
Also defaults to sentinel \_default when

```pycon
>>> s = Configs()
```

Write it as you do with attributes or dict keys,
get it as an attribute and a dict keys.

```pycon
>>> s.foo = 'bar'
>>> assert s.foo == 'bar'
>>> assert s['foo'] == 'bar'
>>> s['hello'] = 'world'
>>> assert s.hello == 'world'
>>> assert s['hello'] == 'world'
>>> hasattr(s, 'hello')
True
```

```pycon
>>> s['does not exist']
no_such_config
```

### *class* lined.base.Digraph(nodes_adjacent_to=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

This class is experiemental and will probably move to meshed.

### *class* lined.base.Fnode(func, name=None, first_arg_position_only=False)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

A function wrapper to be used in pipelines.

```pycon
>>> import pickle
>>>
>>> Sig(pickle.dumps)
<Sig (obj, protocol=None, *, fix_imports=True, buffer_callback=None)>
>>> fn = Fnode(pickle.dumps)
>>> Sig(fn)
<Sig (obj, protocol=None, *, fix_imports=True, buffer_callback=None)>
```

The `first_arg_position_only=True` flag will set the first argument of the
Fnode instance signature to be POSITION_ONLY.
This is a useful normalization for inner nodes of a pipeline.

```pycon
>>> fn = Fnode(pickle.dumps, name='my_pickler', first_arg_position_only=True)
>>> Sig(fn)
<Sig (obj, /, protocol=None, *, fix_imports=True, buffer_callback=None)>
>>> unpickled_fn = pickle.loads(pickle.dumps(fn))
```

```pycon
>>> unpickled_fn
my_pickler(obj, /, protocol=None, *, fix_imports=True, buffer_callback=None)
```

### *class* lined.base.LineParametrized(default_conflict_method, strict, \*funcs, pipeline_name=None, input_name=None, output_name=None)

Bases: `Line`

A pipeline that exposes all inputs of all the functions of the pipeline.

For example, say you have two functions `f(a, b=1)` and `g(x, y=2)`.

`Line(f, g)` would be a function with `a` and `b` as your inputs.

On the other hand, `LineParametrized(f, g)` will give you control over `y`.

```pycon
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
```

```pycon
>>> from inspect import signature
>>> signature(f)
<Sig (a, b=0, *, y=2, e=3)>
```

Note that `y` and `e` are keyword-only arguments.
All arguments that are not from the first function will be keyword only
(except for the first argument of these functions, which do not appear at all
since they’re used as the “connecting arguments”).

Note in the above signature, that `x` and `r` are missing.
That’s because these are “connecting” arguments.
‘x’ comes from `add` and is fed to `times`.
`r` comes from `times` and is fed to `exp`.

```pycon
>>> print(f.dot_digraph_ascii())
        ┌─────┐  x   ┌───────┐  r   ┌─────┐
 a  ──▶ │ add │ ───▶ │ times │ ───▶ │ exp │ ──▶  output
        └─────┘      └───────┘      └─────┘
          ▲            ▲              ▲
          │            │              │
          │            │              │

          b=            y=            e=
```

```pycon
>>> print(f.dot_digraph_ascii(edge_kind='simple'))
        ┌─────┐     ┌───────┐     ┌─────┐
 a  ──▶ │ add │ ──▶ │ times │ ──▶ │ exp │ ──▶  output
        └─────┘     └───────┘     └─────┘
          ▲
          │
          │

          b=
```

```pycon
>>> print(f.dot_digraph_ascii(edge_kind='simple_on_edge'))
        ┌─────┐  x   ┌───────┐  r   ┌─────┐
 a  ──▶ │ add │ ───▶ │ times │ ───▶ │ exp │ ──▶  output
        └─────┘      └───────┘      └─────┘
          ▲
          │
          │

          b=
```

```pycon
>>> print(f.dot_digraph_ascii(edge_kind='to_args'))

                        x            r

                      │             │
                      │             │
                      ▼             ▼
        ┌─────┐     ┌───────┐     ┌─────┐
 a  ──▶ │ add │ ──▶ │ times │ ──▶ │ exp │ ──▶  output
        └─────┘     └───────┘     └─────┘
          ▲           ▲             ▲
          │           │             │
          │           │             │

          b=           y=           e=
```

### *class* lined.base.LineSentineled(\*funcs, pipeline_name=None, input_name=None, output_name=None)

Bases: `Line`

A pipeline that can be interrupted by a sentinel.

Sentinels are useful to interrupt the pipeline computation.

Say, for example, you know if the length of an input iterable divided by three is
1 or 2.
You wouldn’t want to divide by 0 or have a loop choke on an input that doesn’t
have a length.

So you do this:

```pycon
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
```

### *class* lined.base.Sentinel(val=None)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

To make sentinels holding an optional value

### lined.base.SentineledPipeline

alias of [`LineSentineled`](#lined.base.LineSentineled)

### lined.base.log_calls(line, logger=<built-in function print>, what_to_log=<function \_call_signature>)

Log the calls of every step of a pipeline.

* **Parameters:**
  * **line** (`Line`) – A Line object
  * **logger** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)) – A callable that will be called on the output of what_to_log.
    Default is print.
  * **what_to_log** – A callable that will be called on (func, args, kwargs) to
    produce the data that will be pased on to logger.
    By default it will produce a string representation of the call.
* **Returns:**
  A pipeline that will log the calls made at every step

```pycon
>>> from math import log2
>>> pipe = Line(sum, log2, str)
>>> logged_pipe = log_calls(pipe)
>>> t = logged_pipe([1, 3, 4])
sum([1, 3, 4], )
log2(8, )
str(3.0, )
>>> t
'3.0'
```

We’re using the `(logger=print, what_to_log=_call_signature)` default here,
but other pair can be useful in some situations.

For example, some of these calls may involve objects whose string representation
is no informative, or two large to be useful.
In this case, one could instead set `(logger, what_to_log)` to serialize the
calls and save in a DB or pickle files that could then be studied.

#### NOTE
This function logs all the calls.

If you want to log only some calls, you might want to use lined.tools.side_call
or `lined.tools.print_and_pass_on`.
You can also provide a custome `(logger, what_to_log)` pair that will do something
special according to the function (namely, log the call or not).

### lined.base.name_with_varkind_and_default_marker(param)

Returns a string representation for the Parameter object

* **Return type:**
  [`str`](https://docs.python.org/3/builtins/stdtypes.html#str)

### lined.base.uniquize_funcs_names(named_funcs_obtained_from_funcs, suffic_counter_start=2)

Check and make sure that the functions names are all different. This is achieved by adding a suffix to
the duplicate names
