# lined.simple

Simple lightweight utils

### Functions

| [`compose`](#lined.simple.compose)(\*funcs)   |    |
|---------------------------------------------------------------------|----|
| `signature_from_first_and_last_func`(...)                           |    |

### Classes

| [`Pipe`](#lined.simple.Pipe)(\*funcs, \*\*named_funcs)   | Simple function composition.   |
|-----------------------------------------------------------------------------------|--------------------------------|

### *class* lined.simple.Pipe(\*funcs, \*\*named_funcs)

Bases: [`object`](https://docs.python.org/3/builtins/functions.html#object)

Simple function composition. That is, gives you a callable that implements input -> f_1 -> … -> f_n -> output.

```pycon
>>> def foo(a, b=2):
...     return a + b
>>> f = Pipe(foo, lambda x: print(f"x: {x}"))
>>> f(3)
x: 5
```

You can name functions, but this would just be for documentation purposes.
The names are completely ignored.

```pycon
>>> g = Pipe(
...     add_numbers = lambda x, y: x + y,
...     multiply_by_2 = lambda x: x * 2,
...     stringify = str
... )
>>> g(2, 3)
'10'
```

### Notes

- Pipe instances don’t have a \_\_name_\_ etc. So some expectations of normal functions are not met.
- Pipe instance are pickalable (as long as the functions that compose them are)

### lined.simple.compose(\*funcs)

* **Parameters:**
  **funcs**
* **Returns:**

```pycon
>>> def foo(a, b=2):
...     return a + b
>>> f = compose(foo, lambda x: print(f"x: {x}"))
>>> f(3)
x: 5
```

### Notes

- composed functions are normal functions (have a \_\_name_\_ etc.) but are not pickalable. See Pipe for that.
