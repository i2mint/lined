# negating a function, methodcaller, and not_

Let's say we want to make a `negate` decorator that will return a function with 
the same signature, but whose output is negated (that is, it calls the original
function getting the `output` but instead of returning it, 
it returns `not output`. 

One of the uses is to avoid having to write `lambda x: not func(x)`, which is okay, but if we're going to be 
negating a lot, we might want to get a negated version of a function that can be assigned, named, 
and even pickled. 

The standard way to solve this would be:

```python
from functools import wraps

def negate(func):
    @wraps(func)
    def negated_func(*args, **kwargs):
        return not func(*args, **kwargs)
    return negated_func
```

In this example, we'll also show how to use partial and method caller to make 
a `startswith_caller` function that makes 
specific "this string starts with" functions.

```python
from functools import partial
from operator import methodcaller
startswith_caller = partial(methodcaller, 'startswith')
startswith_osdot = startswith_caller('os.')

assert startswith_osdot('os.path')
assert not startswith_osdot('ostentatious')
```

Let's test our `negate` function now...

```python
not_startswith_osdot = negate(startswith_osdot)  # the opposite of startswith_osdot

assert not not_startswith_osdot('os.path')
assert not_startswith_osdot('ostentatious')
```

Unfortunately, this `negate` doesn't product `picklalble` functions because 
it uses local objects (whatever that means!).
We'll profit of this situation to demo `return_instead_of_raising_exceptions`, 
which changes a callable to not raise exceptions, 
but return them instead.

```python
from lined import Pipe
from lined.tools import return_instead_of_raising_exceptions
import pickle

is_pickleable = Pipe(
    return_instead_of_raising_exceptions(
        Pipe(pickle.dumps, pickle.loads)
    ), 
    negate(lambda x: isinstance(x, BaseException))  
    # yeah, we could do lambda x: not isinstance(x,...), 
    # but hey, this is negate's show here!
)

assert is_pickleable(startswith_caller)
assert not is_pickleable(not_startswith_osdot) 
# negated functions are not pickalable, but we'd like them to...
```

`lined.Pipe` and `operator.not_` save you here. And the code looks even simpler


```python
from operator import not_

def negate(func):  # TODO: Do we want to use wraps(func) to get more than just signature?
    return Pipe(func, not_)

not_startswith_osdot = negate(startswith_osdot)  # the opposite of startswith_osdot
assert not not_startswith_osdot('os.path')
assert not_startswith_osdot('ostentatious')

assert is_pickleable(startswith_caller)
assert is_pickleable(not_startswith_osdot)  # yay, we can now pickle!!
```
