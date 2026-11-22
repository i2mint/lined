# lined.tools

All kinds of useful tools to use in pipelines.

### Functions

| `add_name`(obj[, name])                                                                            |                                                                                                                                                                         |
|----------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`append_output_to_input`](#lined.tools.append_output_to_input)(func[, appender])          | Decorator that makes the function into a function returning its input with output                                                                                       |
| [`apply_to_single_item`](#lined.tools.apply_to_single_item)(func, item_idx)              | Get a version of func that applies itself to only the item_idx-th element of the input, leaving the rest untouched.                                                     |
| [`blind`](#lined.tools.blind)(x, output)                                  | Takes one argument, and returns it as is.                                                                                                                               |
| [`call`](#lined.tools.call)(func)                                        | Just call the input function with not arguments.                                                                                                                        |
| [`check_sorted_during_iteration`](#lined.tools.check_sorted_during_iteration)(iterable[, ...])    | Wrap an iterable so that ordering of the elements is checked at runtime.                                                                                                |
| [`consume_until_error`](#lined.tools.consume_until_error)(iterable[, caught_errors])    | Iterable that will simply exit with out error if one of the caught errors occurs.                                                                                       |
| [`deiterize`](#lined.tools.deiterize)(func)                                   | The inverse of iterize.                                                                                                                                                 |
| [`del_fields`](#lined.tools.del_fields)(d, fields)                             | Returns the same mapping, but with specified fields removed.                                                                                                            |
| [`dictify`](#lined.tools.dictify)(func[, copy_dict, name])                  | Makes a version of the input func that should be called on dictionaries and will return dictionaries.                                                                   |
| [`enumerate_groups`](#lined.tools.enumerate_groups)(iterable[, key, start])          | Get enumeration of groups during a groupby call.                                                                                                                        |
| [`expanded_args`](#lined.tools.expanded_args)(func)                               | Make's a func(\*args) function out of a func(args) one.                                                                                                                 |
| `extra_wraps`(func[, name, doc_prefix])                                                            |                                                                                                                                                                         |
| [`functioncaller`](#lined.tools.functioncaller)(\*args, \*\*kwargs)                | Call a function given positional and keyword arguments.                                                                                                                 |
| [`generator_version`](#lined.tools.generator_version)(func[, name])                   | From an Input->Ouput function, makes a Iterator[Input]->Itertor[Output] Some call this "vectorization", but it's not really a vector, but an iterable, thus the name.   |
| [`identity`](#lined.tools.identity)(x)                                       | Takes one argument, and returns it as is                                                                                                                                |
| [`if_then_else`](#lined.tools.if_then_else)(x[, if_func, then_func, else_func])  | Implement the if-then-else logic as a function.                                                                                                                         |
| `is_a_str`(x)                                                                                      |                                                                                                                                                                         |
| `is_not_iterable_or_is_a_str`(x)                                                                   |                                                                                                                                                                         |
| [`items`](#lined.tools.items)(mapping)                                    | Get an items generator from a mapping                                                                                                                                   |
| [`iterate`](#lined.tools.iterate)(iterable)                                 | Just iterate through a iterable Use this to "consume" or "run" an iterator automatically.                                                                               |
| [`iterize`](#lined.tools.iterize)(func[, name])                             | From an Input->Ouput function, makes a Iterator[Input]->Itertor[Output] Some call this "vectorization", but it's not really a vector, but an iterable, thus the name.   |
| [`keys_extractor`](#lined.tools.keys_extractor)(keys)                              | Deprecated: Use operator.itemgetter(\*keys) instead.                                                                                                                    |
| `make_it_a_tuple`(x)                                                                               |                                                                                                                                                                         |
| [`map_star`](#lined.tools.map_star)(func)                                    | Make a func(args) function out of a func(\*args) o.                                                                                                                     |
| `mk_filter`([filter_func])                                                                         |                                                                                                                                                                         |
| `mywraps`(func[, name, doc_prefix])                                                                |                                                                                                                                                                         |
| [`negate`](#lined.tools.negate)(func)                                      | Get a negated version of a function                                                                                                                                     |
| [`pairwise`](#lined.tools.pairwise)(iterable)                                | Yield sliding window pairs                                                                                                                                              |
| [`raise_`](#lined.tools.raise_)(exception)                                 | raises the given exception (instance or callable that returns one) Meant to be hooked to the out put of a function that returns an exception or a command to raise one. |
| [`raise_exception`](#lined.tools.raise_exception)(exception, \*args, \*\*kwargs)    | Raise an exception (from an exception instance, or a callable that makes one                                                                                            |
| [`return_instead_of_raising_exceptions`](#lined.tools.return_instead_of_raising_exceptions)([func, ...]) | Make a function return its exceptions instead of raising them.                                                                                                          |
| [`side_call`](#lined.tools.side_call)(x, callback)                            | Identity function that calls a callaback function before returning the input as is (unless the input is mutable and the callback changes it).                           |
| [`singularize_arg_input`](#lined.tools.singularize_arg_input)(func)                       | Make a func(args) function out of a func(\*args) o.                                                                                                                     |
| [`tail_io`](#lined.tools.tail_io)(func)                                     | Will apply function only to the tail of tuple inputs, still passing the header on.                                                                                      |
| [`valmap`](#lined.tools.valmap)(d, func[, copy_dict])                      | Apply func to the values of a shallow copy of d, unless copy_dict=False, in which case, it will be applied to the input dict itself.                                    |
| [`with_cursor`](#lined.tools.with_cursor)(func[, start, step])                  | Decorator a function so it enumerates the number of calls.                                                                                                              |
| [`wrap_first_arg_in_list`](#lined.tools.wrap_first_arg_in_list)(func)                      | Takes a func(X,...) function and returns a func([X],...) function.                                                                                                      |

### Classes

| [`Command`](#lined.tools.Command)(func, \*args, \*\*kwargs)     | Make a no-input callable that will execute a specific function call.          |
|----------------------------------------------------------------------------------------|-------------------------------------------------------------------------------|
| [`CommandIter`](#lined.tools.CommandIter)(func, \*args, \*\*kwargs) | An infinite iterator that returns the results of a Command called repeatedly. |
| [`Enumerate`](#lined.tools.Enumerate)(func[, start, step])        | Decorator a function so it enumerates the number of calls.                    |

### Exceptions

| [`ItemsNotSorted`](#lined.tools.ItemsNotSorted)   | Use to indicate that two consecutive items where not in the expected order   |
|-------------------------------------------------------------------|------------------------------------------------------------------------------|

### *class* lined.tools.Command(func, \*args, \*\*kwargs)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Make a no-input callable that will execute a specific function call.

```pycon
>>> command = Command(sum, [1, 2, 3])
>>> command()
6
>>> command = Command(print, 'hello', 'world', sep=', ')
>>> command()
hello, world
```

Note that the same can be achieved with
`partial(func, *args, **kwargs)`.

```pycon
>>> from operator import methodcaller
>>> from functools import partial
>>> def mk_command(func, *args, **kwargs):
...     return partial(methodcaller('__call__', *args, **kwargs), func)
>>> command = mk_command(print, 'hello', 'world', sep=', ')
>>> command()
hello, world
```

See: [https://en.wikipedia.org/wiki/Command_pattern](https://en.wikipedia.org/wiki/Command_pattern)

### *class* lined.tools.CommandIter(func, \*args, \*\*kwargs)

Bases: [`Command`](#lined.tools.Command)

An infinite iterator that returns the results of a Command called repeatedly.

Might become deprecated:
Use `iter(partial(func, *args, **kwargs), object())` instead.

```pycon
>>> from random import uniform
>>> from itertools import islice
>>> it = CommandIter(uniform, 0, 10)
>>> rand_nums = list(islice(it, 4))
>>> assert len(rand_nums) == 4
>>> rand_nums
[4.48171445690221, 7.466083642212892, 0.24120342781796422, 3.694956861724484]
```

### *class* lined.tools.Enumerate(func, start=0, step=1)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Decorator a function so it enumerates the number of calls.
Or in general, returns (cursor, func(x)) instead of just func(x),
where the start and step of the cursor can
be defined (default is start=0 and step=1)

```pycon
>>> def foo(x):
...    return x * 2
>>> new_foo = Enumerate(foo)
>>> new_foo('ha')
(0, 'haha')
>>> new_foo('ho')
(1, 'hoho')
>>> enum_foo_with_step = Enumerate(foo, start=3, step=7)
>>> enum_foo_with_step('z')
(3, 'zz')
>>> enum_foo_with_step(11)
(10, 22)
```

### *exception* lined.tools.ItemsNotSorted

Bases: [`RuntimeError`](https://docs.python.org/3/builtins/exceptions.html#RuntimeError)

Use to indicate that two consecutive items where not in the expected
order

### lined.tools.append_output_to_input(func, appender=<function <lambda>>)

Decorator that makes the function into a function returning its input with output

┌─────────────┐
│    input    │
└─────────────┘

> │
> ▼

┌─────────────┐
│    func     │
└─────────────┘

> │
> ▼

┌─────────────────┐
│ (input, output) │
└─────────────────┘

```pycon
>>> func = lambda x: f"hello {x}"
>>> func('world')
'hello world'
>>> new_func = append_output_to_input(func)
>>> new_func('world')
('world', 'hello world')
```

### lined.tools.apply_to_single_item(func, item_idx)

Get a version of func that applies itself to only the item_idx-th
element of the input,
leaving the rest untouched.

That is, apply_to_single_item(func, 2), for example, is a new_func such that

```text
new_func([a, b, c, d, e]) == [a, b, func(c), d, e]
```

* **Parameters:**
  * **func** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)) – A function to apply to a single element of an iterable (that
    has a […])
  * **item_idx** ([`int`](https://docs.python.org/3/builtins/functions.html#int)) – The particular item index to apply function to
* **Returns:**

```pycon
>>> apply_to_second_item = apply_to_single_item(
...     func=lambda x: x * 10, item_idx=1)
>>> apply_to_second_item([1, 2, 3, 4])
(1, 20, 2, 3, 4)
```

### lined.tools.blind(x, output)

Takes one argument, and returns it as is.
The output is meant to be bound by currying (functools.partial)

```pycon
>>> true_no_matter_what = partial(blind, output=True)
>>> false_no_matter_what = partial(blind, output=False)
>>> true_no_matter_what(42)
True
>>> false_no_matter_what(42)
False
```

### lined.tools.call(func)

Just call the input function with not arguments.

Equivalent to `functioncaller()`

```pycon
>>> from lined import Line
>>> from functools import partial
>>>
>>> line = Line(lambda x: partial(print, f"{x*3=}"), call)
>>> line(14)
x*3=42
```

### lined.tools.cast_to_tuple_if_non_iterable_or_a_string(x, \*, if_func=<function is_not_iterable_or_is_a_str>, then_func=<function make_it_a_tuple>, else_func=<function identity>)

Implement the if-then-else logic as a function.

```pycon
>>> if_then_else(
...     'world',
...     if_func=lambda x: x == 'world',
...     then_func="hello {}".format,
...     else_func=lambda x: x * 2)
'hello world'
>>> if_then_else('bora', if_func=lambda x: x == 'world',
... then_func="hello{}".format, else_func=lambda x: x * 2)
'borabora'
```

Really, it’s meant to be curried to make functional components.
For example, to make a function that ensures that a string is
encapsulated in a tuple, we could do this:

```pycon
>>> def is_a_str(x): return isinstance(x, str)
>>> def make_it_a_tuple(x): return tuple([x])
>>>
>>> ensure_tuple_if_string = partial(
...     if_then_else,
...     if_func=is_a_str,
...     then_func=make_it_a_tuple
... )
>>> ensure_tuple_if_string('a string')
('a string',)
>>> ensure_tuple_if_string(['a', 'list'])  # not a string so returned as is
['a', 'list']
```

### lined.tools.cast_to_tuple_if_string(x, \*, if_func=<function is_a_str>, then_func=<function make_it_a_tuple>, else_func=<function identity>)

Implement the if-then-else logic as a function.

```pycon
>>> if_then_else(
...     'world',
...     if_func=lambda x: x == 'world',
...     then_func="hello {}".format,
...     else_func=lambda x: x * 2)
'hello world'
>>> if_then_else('bora', if_func=lambda x: x == 'world',
... then_func="hello{}".format, else_func=lambda x: x * 2)
'borabora'
```

Really, it’s meant to be curried to make functional components.
For example, to make a function that ensures that a string is
encapsulated in a tuple, we could do this:

```pycon
>>> def is_a_str(x): return isinstance(x, str)
>>> def make_it_a_tuple(x): return tuple([x])
>>>
>>> ensure_tuple_if_string = partial(
...     if_then_else,
...     if_func=is_a_str,
...     then_func=make_it_a_tuple
... )
>>> ensure_tuple_if_string('a string')
('a string',)
>>> ensure_tuple_if_string(['a', 'list'])  # not a string so returned as is
['a', 'list']
```

### lined.tools.check_sorted_during_iteration(iterable, key=<built-in function le>, not_sorted_callback=<lined.tools.Command object>)

Wrap an iterable so that ordering of the elements is checked at runtime.

* **Parameters:**
  * **iterable** ([`Iterable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Iterable)) – Iterable to consume
  * **key** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable)[[[`Any`](https://docs.python.org/3/library/typing.html#typing.Any), [`Any`](https://docs.python.org/3/library/typing.html#typing.Any)], [`bool`](https://docs.python.org/3/builtins/functions.html#bool)]) – The function that defines what it means to be sorted.
    Could be a Any->bool function, which will act like the key argument
    of builtin sorted for example.
    Could also be an explicit (element, next_element)->bool function that
    returns True iff in the right order
  * **not_sorted_callback** ([`Callable`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Callable) | [`BaseException`](https://docs.python.org/3/builtins/exceptions.html#BaseException)) – 

    The function to call when two consecutive
    elements are not sorted. For example:
    - raising an error (the default)
    - logging the information, and skiping the offending element (or not)
* **Return type:**
  [`Generator`](https://docs.python.org/3/library/collections.abc.html#collections.abc.Generator)
* **Returns:**
  A generator consuming the input iterable

```pycon
>>> list(check_sorted_during_iteration(iter([1, 2, 3, 4])))
[1, 2, 3, 4]
```

```pycon
>>> try:
...     for i, x in enumerate(
...             check_sorted_during_iteration([2, 4, 3, 6]), 1):
...         print(x)
... except ItemsNotSorted:
...     print(
...         f"ItemsNotSorted after {i} element (whose value was {x})")
...     print(
...         "----> Normally, here, you'd put exception handling code")
...
2
4
ItemsNotSorted after 2 element (whose value was 4)
----> Normally, here, you'd put exception handling code
```

Now, mind you, you have total control over what sorted means.
For example, to define it as strict

```pycon
>>> comp = lambda x, y: x > y  # in real life, use operator.gt
>>> list(check_sorted_during_iteration(iter([4, 3, 2, 1]), key=comp))
[4, 3, 2, 1]
```

Now for a more complex example.
First we’ll define a function that will consume the iterable until an
error occurs, returning the elements consumed.

```pycon
>>> from lined.tools import consume_until_error
>>> from lined.simple import Pipe
>>> consume = Pipe(check_sorted_during_iteration, consume_until_error, list)
```

```pycon
>>> iterable = ['a', 'ba', 'cba', 'cba', 'back', 'bacca']
>>> consume(iterable, lambda x, y: x < y)  # compare with strict <
['a', 'ba', 'cba']
>>> consume(iterable, len)  # compare based on the length
['a', 'ba', 'cba', 'cba', 'back', 'bacca']
>>> consume(iterable, lambda x: x[0])  # compare based on first letter only
['a', 'ba', 'cba', 'cba']
>>> # compare based on whether the previous element is a subset of the next:
>>> consume(iterable, lambda x, y: set(x).issubset(y))
['a', 'ba', 'cba', 'cba', 'back']
```

### lined.tools.consume_until_error(iterable, caught_errors=(<class 'Exception'>, ))

Iterable that will simply exit with out error if one of the caught
errors occurs.

```pycon
>>> list(consume_until_error(map(lambda x: 1 / x, [4, 2, 1, 0, -1])))
[0.25, 0.5, 1.0]
```

### lined.tools.deiterize(func)

The inverse of iterize.
Takes an “iterized” (a.k.a. “vectorized”) function (i.e. a function that
works on iterables), and
That is, takes a func(X,…) function and returns a next(iter(func([X],
…))) function.

### lined.tools.del_fields(d, fields)

Returns the same mapping, but with specified fields removed.
Intended to be applied to a stream of Mappings, using partial to fix fields

```pycon
>>> d = [{'a': 1, 'b': 2}, {'a': 11, 'c': 3}]
>>> list(map(partial(del_fields, fields=['a']), d))
[{'b': 2}, {'c': 3}]
```

### lined.tools.dictify(func, copy_dict=True, name=None)

Makes a version of the input func that should be called on dictionaries and will
return dictionaries. The function will be applied to the values of a shallow
copy of the dict, unless copy_dict=False, in which case,
it will be applied to the input dict itself.

`dictify` is a partial of `valmap`

```pycon
>>> mult_by_10 = lambda x: x * 10
>>> f = dictify(mult_by_10)
>>> d = {'a': 2, 'b': 3}
>>> f(d)
{'a': 20, 'b': 30}
>>> d  # see that d unchanged
{'a': 2, 'b': 3}
>>> f = dictify(mult_by_10, copy_dict=False)  # but if we ask for it
>>> f(d)
{'a': 20, 'b': 30}
>>> # we still get a (transformed) dict in the output, but it's the same dict changed
>>> d  # now d itself changed
{'a': 20, 'b': 30}
```

### lined.tools.enumerate_groups(iterable, key=None, start=0)

Get enumeration of groups during a groupby call.

* **Parameters:**
  * **iterable** – An iterable
  * **key** – The key to use in the groupby logic
  * **start** – Where to start the enumeration (default is 0)
* **Returns:**
  A generator of (group_idx, group, item) triples

```pycon
>>> iterable = [0, 0, 0, 2, 5, 7, 8, 0, 0, 9, 3, 1]
>>> assert list(enumerate_groups(iterable, key=lambda x: x > 0)) == [
...  (0, False, 0),
...  (0, False, 0),
...  (0, False, 0),
...  (1, True, 2),
...  (1, True, 5),
...  (1, True, 7),
...  (1, True, 8),
...  (2, False, 0),
...  (2, False, 0),
...  (3, True, 9),
...  (3, True, 3),
...  (3, True, 1),
...  ]
```

### lined.tools.expanded_args(func)

Make’s a func(\*args) function out of a func(args) one.
In a way, the opposite of map_star.

```pycon
>>> sum([1,2,3,4])
10
>>> mysum = expanded_args(sum)
>>> mysum(1, 2, 3, 4)
10
```

### lined.tools.false_no_matter_what(x, , output=False)

Takes one argument, and returns it as is.
The output is meant to be bound by currying (functools.partial)

```pycon
>>> true_no_matter_what = partial(blind, output=True)
>>> false_no_matter_what = partial(blind, output=False)
>>> true_no_matter_what(42)
True
>>> false_no_matter_what(42)
False
```

### lined.tools.functioncaller(\*args, \*\*kwargs)

Call a function given positional and keyword arguments.

```pycon
>>> import operator
>>> import functools
>>> from lined import Pipe
>>> f = Pipe(
...     functools.partial(getattr, operator),  # get an operator func by name
...     functioncaller(49, 7)  # apply it to 49 and 7
... )
>>> f('add')
56
>>> f('sub')
42
```

#### NOTE
functioncaller just returns

`operator.methodcaller('__call__', *args, **kwargs)`.

### lined.tools.generator_version(func, name=None)

From an Input->Ouput function, makes a Iterator[Input]->Itertor[Output]
Some call this “vectorization”, but it’s not really a vector, but an
iterable, thus the name.

`iterize` is a partial of `map`.

```pycon
>>> f = lambda x: x * 10
>>> f(2)
20
>>> iterized_f = iterize(f)
>>> list(iterized_f(iter([1,2,3])))
[10, 20, 30]
```

Consider the following pipeline:

```pycon
>>> from lined import Pipe
>>>
>>> pipe = Pipe(lambda x: x * 2,
...                 lambda x: f"hello {x}")
>>> pipe(1)
'hello 2'
```

But what if you wanted to use the pipeline on a “stream” of data. The
following wouldn’t work:

```pycon
>>> try:
...     pipe(iter([1,2,3]))
... except TypeError as e:
...     print(f"{type(e).__name__}: {e}")
...
...
TypeError: unsupported operand type(s) for *: 'list_iterator' and 'int'
```

Remember that error: You’ll surely encounter it at some point.

The solution to it is (often): `iterize`,
which transforms a function that is meant to be applied to a single object,
into a function that is meant to be applied to an array, or any iterable
of such objects.
(You might be familiar (if you use `numpy` for example) with the related
concept of “vectorization”,
or [array programming](https://en.wikipedia.org/wiki/Array_programming).)

```pycon
>>> from lined import Pipe, iterize
>>> from typing import Iterable
>>>
>>> pipe = Pipe(iterize(lambda x: x * 2),
...                 iterize(lambda x: f"hello {x}"))
>>> iterable = pipe([1, 2, 3])
>>> # see that the result is an iterable
>>> assert isinstance(iterable, Iterable)
>>> list(iterable)  # consume the iterable and gather it's items
['hello 2', 'hello 4', 'hello 6']
```

### lined.tools.identity(x)

Takes one argument, and returns it as is

### lined.tools.if_then_else(x, if_func=functools.partial(<function blind>, output=True), then_func=<function identity>, else_func=<function identity>)

Implement the if-then-else logic as a function.

```pycon
>>> if_then_else(
...     'world',
...     if_func=lambda x: x == 'world',
...     then_func="hello {}".format,
...     else_func=lambda x: x * 2)
'hello world'
>>> if_then_else('bora', if_func=lambda x: x == 'world',
... then_func="hello{}".format, else_func=lambda x: x * 2)
'borabora'
```

Really, it’s meant to be curried to make functional components.
For example, to make a function that ensures that a string is
encapsulated in a tuple, we could do this:

```pycon
>>> def is_a_str(x): return isinstance(x, str)
>>> def make_it_a_tuple(x): return tuple([x])
>>>
>>> ensure_tuple_if_string = partial(
...     if_then_else,
...     if_func=is_a_str,
...     then_func=make_it_a_tuple
... )
>>> ensure_tuple_if_string('a string')
('a string',)
>>> ensure_tuple_if_string(['a', 'list'])  # not a string so returned as is
['a', 'list']
```

### lined.tools.items(mapping)

Get an items generator from a mapping

### lined.tools.iterate(iterable)

Just iterate through a iterable
Use this to “consume” or “run” an iterator automatically.

For example, consider the following:

```pycon
>>> from lined import Pipe, iterize, iterate
>>> pipe = Pipe(
...     iterize(lambda x: x * 2),
...     iterize(lambda x: print(f"hello {x}")),
... )
>>>
>>> for _ in pipe([1, 2, 3]):
...     pass
hello 2
hello 4
hello 6
```

It could be a bit awkward to have to “consume” the iterable to have it
take effect.
Just calling  `pipe([1, 2, 3])` to get those prints seems like a more
natural way.
This is where you can use `iterate`. It basically “launches” that
consuming loop for you.

```pycon
>>> pipe = Pipe(
...     iterize(lambda x: x * 2),
...     iterize(lambda x: print(f"hello {x}")),
...     iterate
... )
>>>
>>> pipe([1, 2, 3])
hello 2
hello 4
hello 6
```

### lined.tools.iterize(func, name=None)

From an Input->Ouput function, makes a Iterator[Input]->Itertor[Output]
Some call this “vectorization”, but it’s not really a vector, but an
iterable, thus the name.

`iterize` is a partial of `map`.

```pycon
>>> f = lambda x: x * 10
>>> f(2)
20
>>> iterized_f = iterize(f)
>>> list(iterized_f(iter([1,2,3])))
[10, 20, 30]
```

Consider the following pipeline:

```pycon
>>> from lined import Pipe
>>>
>>> pipe = Pipe(lambda x: x * 2,
...                 lambda x: f"hello {x}")
>>> pipe(1)
'hello 2'
```

But what if you wanted to use the pipeline on a “stream” of data. The
following wouldn’t work:

```pycon
>>> try:
...     pipe(iter([1,2,3]))
... except TypeError as e:
...     print(f"{type(e).__name__}: {e}")
...
...
TypeError: unsupported operand type(s) for *: 'list_iterator' and 'int'
```

Remember that error: You’ll surely encounter it at some point.

The solution to it is (often): `iterize`,
which transforms a function that is meant to be applied to a single object,
into a function that is meant to be applied to an array, or any iterable
of such objects.
(You might be familiar (if you use `numpy` for example) with the related
concept of “vectorization”,
or [array programming](https://en.wikipedia.org/wiki/Array_programming).)

```pycon
>>> from lined import Pipe, iterize
>>> from typing import Iterable
>>>
>>> pipe = Pipe(iterize(lambda x: x * 2),
...                 iterize(lambda x: f"hello {x}"))
>>> iterable = pipe([1, 2, 3])
>>> # see that the result is an iterable
>>> assert isinstance(iterable, Iterable)
>>> list(iterable)  # consume the iterable and gather it's items
['hello 2', 'hello 4', 'hello 6']
```

### lined.tools.keys_extractor(keys)

Deprecated: Use operator.itemgetter(\*keys) instead.

### lined.tools.map_star(func)

Make a func(args) function out of a func(\*args) o.
Also known as singularize_arg_input.
In a way, the opposite of map_starexpanded_args.

```pycon
>>> def foo(a, b):
...     return a + b
>>> singularized_foo = map_star(foo)
>>> singularized_foo((2, 3))
5
>>> assert singularized_foo([2, 3]) == singularized_foo({2, 3}) == foo(2, 3)
```

### lined.tools.negate(func)

Get a negated version of a function

Will return a function with
the same signature, but whose output is negated (that is, it calls the original
function getting the `output` but instead of returning it,
it returns `not output`.

```pycon
>>> sum([1, 2, 3])
6
>>> sum([-2, 2])
0
>>> sum_is_zero = negate(sum)
>>> sum_is_zero([1, 2, 3])
False
>>> sum_is_zero([-2, 2])
True
```

### lined.tools.pairwise(iterable)

Yield sliding window pairs

```pycon
>>> list(pairwise([1, 2, 3, 4]))
[(1, 2), (2, 3), (3, 4)]
```

### lined.tools.print_and_pass_on(x, \*, callback=<built-in function print>)

Passes input through to output, but prints before outputing

### lined.tools.raise_(exception)

raises the given exception (instance or callable that returns one)
Meant to be hooked to the out put of a function that returns an exception or a
command to raise one.

### lined.tools.raise_exception(exception, \*args, \*\*kwargs)

Raise an exception (from an exception instance, or a callable that
makes one

### lined.tools.return_instead_of_raising_exceptions(func=None, \*, exceptions=(<class 'Exception'>, ))

Make a function return its exceptions instead of raising them.

```pycon
>>> def foo(x, y):
...     return x / y
>>> f = return_instead_of_raising_exceptions(foo)
>>> f(6, 2)
3.0
>>> f(1,0)  # note that this doesn't raise, but returns the exception (instance)
ZeroDivisionError('division by zero')
```

* **Parameters:**
  * **func** – The function to transform.
  * **exceptions** – The exceptions to handle
    Default is Exception (letting other BaseException instances like
    KeyboardInterrupt still be raised). If you need more exceptions, or less
    exceptions to be handled, enter them here.
* **Returns:**

### lined.tools.side_call(x, callback)

Identity function that calls a callaback function before returning the
input as is (unless the input is mutable and the callback changes it).

```pycon
>>> from lined import Pipe
>>> add2 = lambda x: x + 2
>>> add2(40)
42
>>> from functools import partial
>>> logger = partial(side_call, callback=lambda x: print(f"input is {x}"))
>>> logged_add2 = Pipe(logger, add2)
>>> logged_add2(40)
input is 40
42
```

### lined.tools.singularize_arg_input(func)

Make a func(args) function out of a func(\*args) o.
Also known as singularize_arg_input.
In a way, the opposite of map_starexpanded_args.

```pycon
>>> def foo(a, b):
...     return a + b
>>> singularized_foo = map_star(foo)
>>> singularized_foo((2, 3))
5
>>> assert singularized_foo([2, 3]) == singularized_foo({2, 3}) == foo(2, 3)
```

### lined.tools.tail_io(func)

Will apply function only to the tail of tuple inputs, still passing
the header on.
That is, from a `x -> func(x)` function, you get a `(\*header, x) -> (
\*header, func(x))` function.

```pycon
>>> def foo(x):
...    return x * 2
>>>
>>> foo('boo')
'booboo'
>>> new_foo = tail_io(foo)
>>> new_foo((7, 'boo'))
(7, 'booboo')
>>> new_foo(('all', 'items', 'but', 'the', 'last', 'are', 'just',
...          'passed', 'on', 'boo'))
('all', 'items', 'but', 'the', 'last', 'are', 'just', 'passed', 'on', 'booboo')
```

### lined.tools.true_no_matter_what(x, , output=True)

Takes one argument, and returns it as is.
The output is meant to be bound by currying (functools.partial)

```pycon
>>> true_no_matter_what = partial(blind, output=True)
>>> false_no_matter_what = partial(blind, output=False)
>>> true_no_matter_what(42)
True
>>> false_no_matter_what(42)
False
```

### lined.tools.valmap(d, func, copy_dict=True)

Apply func to the values of a shallow copy of d, unless copy_dict=False,
in which case, it will be applied to the input dict itself.

```pycon
>>> d = {'a': 2, 'b': 3}
>>> valmap(d, lambda x: x * 10)
{'a': 20, 'b': 30}
>>> d  # see that d unchanged
{'a': 2, 'b': 3}
>>> valmap(d, lambda x: x * 10, copy_dict=False)  # but if we ask for it
{'a': 20, 'b': 30}
>>> # we still get a (transformed) dict in the output, but it's the same dict changed
>>> d  # now d itself changed
{'a': 20, 'b': 30}
```

### lined.tools.with_cursor(func, start=0, step=1)

Decorator a function so it enumerates the number of calls.
Or in general, returns (cursor, func(x)) instead of just func(x),
where the start and step of the cursor can
be defined (default is start=0 and step=1)

```pycon
>>> def foo(x):
...    return x * 2
>>> new_foo = with_cursor(foo)
>>> new_foo('ha')
(0, 'haha')
>>> new_foo('ho')
(1, 'hoho')
>>> enum_foo_with_step = with_cursor(foo, start=3, step=7)
>>> enum_foo_with_step('z')
(3, 'zz')
>>> enum_foo_with_step(11)
(10, 22)
```

### lined.tools.wrap_first_arg_in_list(func)

Takes a func(X,…) function and returns a func([X],…) function.
