# lined.util

Util functions

### Functions

| `clean`(varStr)                                                                          |                                                                                                                      |
|------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------|
| [`dot_to_ascii`](#lined.util.dot_to_ascii)(dot[, fancy])              | Convert a dot string to an ascii rendering of the diagram.                                                           |
| [`ensure_numerical_keys`](#lined.util.ensure_numerical_keys)(k[, names])       | Returns a numerical version of the input key.                                                                        |
| [`func_name`](#lined.util.func_name)(func)                         | The func._\_name_\_ of a callable func, or makes and returns one if that fails.                                      |
| [`incremental_str_maker`](#lined.util.incremental_str_maker)([str_format])     | Make a function that will produce a (incrementally) new string at every call.                                        |
| `make_method`(desc, obj, obj_type)                                                       |                                                                                                                      |
| [`mro_items`](#lined.util.mro_items)(type_obj)                     | Takes a type and returns an iterator over all class variables throughout the type hierarchy (respecting the MRO).    |
| [`n_required_args`](#lined.util.n_required_args)(func)                   | Number of required arguments.                                                                                        |
| `name_to_id`(s[, repl_str])                                                              |                                                                                                                      |
| `param_is_required`(param)                                                               |                                                                                                                      |
| [`partial_plus`](#lined.util.partial_plus)(func, \*args, \*\*kwargs)  | Like partial, but with the ability to add 'normal function' stuff (name, doc) to the curried function.               |
| [`signature_from_first_and_last_func`](#lined.util.signature_from_first_and_last_func)(...) | Returns a signature using the first_func for it's params and last_func for return annotations, both only if possible |
| `unnamed_func_name`()                                                                    |                                                                                                                      |
| `unnamed_pipeline`()                                                                     |                                                                                                                      |

### Classes

| [`CachedInstancePartial`](#lined.util.CachedInstancePartial)   | The `CachedInstancePartial` is virtually the same as `InstancePartial`, adding support for method-usage to `functools.partial`, except that upon first access, it caches the bound method on the associated object, speeding it up for future accesses, and bringing the method call overhead to about the same as non-`partial` methods.   |
|--------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

### *class* lined.util.CachedInstancePartial

Bases: `partial`

The `CachedInstancePartial` is virtually the same as
`InstancePartial`, adding support for method-usage to
`functools.partial`, except that upon first access, it
caches the bound method on the associated object, speeding it up
for future accesses, and bringing the method call overhead to
about the same as non-`partial` methods.

See the `InstancePartial` docstring for more details.

### lined.util.dot_to_ascii(dot, fancy=True)

Convert a dot string to an ascii rendering of the diagram.

Needs a connection to the internet to work.

```pycon
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
>>> graph_ascii = dot_to_ascii(graph_dot)
>>>
>>> print(graph_ascii)

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
```

### lined.util.ensure_numerical_keys(k, names=None)

Returns a numerical version of the input key.
If `k` is None or an integer, will be returned as is.
If `k` is a string, it will return the (int) index of `k` in `names`.
If `k` is a slice, it will convert the k.start and k.stop to ints (or None).

```pycon
>>> ensure_numerical_keys(42)
42
>>> ensure_numerical_keys('bob', ['alice', 'bob', 'charlie'])
1
>>> ensure_numerical_keys(slice('bob', 'charlie'), ['alice', 'bob', 'charlie'])
slice(1, 2, None)
>>> ensure_numerical_keys(slice('bob'), ['alice', 'bob', 'charlie'])
slice(None, 1, None)
```

### lined.util.func_name(func)

The func._\_name_\_ of a callable func, or makes and returns one if that fails.
To make one, it calls unamed_func_name which produces incremental names to reduce the chances of clashing

### lined.util.incremental_str_maker(str_format='{:03.f}')

Make a function that will produce a (incrementally) new string at every call.

### lined.util.mro_items(type_obj)

Takes a type and returns an iterator over all class variables
throughout the type hierarchy (respecting the MRO).

```pycon
>>> sorted(set([k for k, v in mro_items(int) if not k.startswith('__')
...     and 'bytes' not in k and not callable(v)]))
['denominator', 'imag', 'numerator', 'real']
```

### lined.util.n_required_args(func)

Number of required arguments.

A required argument is one that doesn’t have a default, nor is VAR_POSITIONAL (\*args) or VAR_KEYWORD (\*\*kwargs).

#### NOTE
Sometimes a minimum number of arguments in VAR_POSITIONAL and VAR_KEYWORD are in fact required,
but we can’t see this from the signature, so we can’t tell you about that! You do the math.

```pycon
>>> n_required_args(lambda x, y, z=None, *args, **kwargs: ...)
2
```

* **Return type:**
  [`int`](https://docs.python.org/3/builtins/functions.html#int)

### lined.util.partial_plus(func, \*args, \*\*kwargs)

Like partial, but with the ability to add ‘normal function’ stuff
(name, doc) to the curried function.

#### NOTE
if no writable_function_dunders is specified will just act as the
builtin partial (which it calls first).

```pycon
>>> def foo(a, b): return a + b
>>> f = partial_plus(foo, b=2, __name__='bar', __doc__='foo, but with b=2')
>>> f.__name__
'bar'
>>> f.__doc__
'foo, but with b=2'
```

### lined.util.signature_from_first_and_last_func(first_func, last_func)

Returns a signature using the first_func for it’s params and last_func for
return annotations, both only if possible
