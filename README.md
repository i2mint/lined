
# lined

Building simple pipelines, simply.

And lightly too! No dependencies. All with pure builtin python.

A really simple example:

```pydocstring
>>> p = Pipeline(sum, str)
>>> p([2, 3])
'5'
```

A still quite simple example:

```pydocstring
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
```

Let's check out the signature of f:

```pydocstring
>>> from inspect import signature
>>>
>>> assert str(signature(f)) == '(a, b=1) -> float'
>>> assert signature(f).parameters == signature(first).parameters
>>> assert signature(f).return_annotation == signature(last).return_annotation == float
```

Border case: One function only

```pydocstring
>>> same_as_first = Pipeline(first)
>>> assert same_as_first(42) == first(42)
```



# More?

## string and dot digraph representations

Pipeline's string representation (`__repr__`) and how it deals with callables that don't have a `__name__` (hint: it makes one up):

```python
from lined.base import Pipeline
from functools import partial

pipe = Pipeline(sum, np.log, str, print, partial(map, str), name='some_name')
pipe
```
```
Pipeline(sum, log, str, print, unnamed_func_001, name='some_name')
```

If you have [graphviz](https://pypi.org/project/graphviz/) installed, you can also do this:
```python
pipe.dot_digraph()
```
![image](https://user-images.githubusercontent.com/1906276/96199560-ce0b8480-0f25-11eb-8b0a-5f0e537e48d6.png)

And if you don't, but have some other [dot language](https://www.graphviz.org/doc/info/lang.html) interpreter, you can just get the body (and fiddle with it):

```python
print('\n'.join(pipe.dot_digraph_body()))
```
```
rankdir="LR"
sum [shape="box"]
log [shape="box"]
str [shape="box"]
print [shape="box"]
unnamed_func_001 [shape="box"]
sum -> log
log -> str
str -> print
print -> unnamed_func_001
```

Optionally, a pipeline can have an `input_name` and/or an `output_name`. 
These will be used in the string representation and the dot digraph.

```python
pipe = Pipeline(sum, np.log, str, print, partial(map, str), input_name='x', output_name='y')
str(pipe)
```
```
"Pipeline(sum, log, str, print, unnamed_func_001, name='some_name')"
```

```python
pipe.dot_digraph()
```
![image](https://user-images.githubusercontent.com/1906276/96199887-86d1c380-0f26-11eb-9df6-642a3762787b.png)




# Tools


## iterize and iterate


```python
from lined import Pipeline

pipe = Pipeline(lambda x: x * 2, 
                lambda x: f"hello {x}")
pipe(1)
```




    'hello 2'



But what if you wanted to use the pipeline on a "stream" of data. The following wouldn't work:


```python
try:
    pipe(iter([1,2,3]))
except TypeError as e:
    print(f"{type(e).__name__}: {e}")
```

    TypeError: unsupported operand type(s) for *: 'list_iterator' and 'int'


Remember that error: You'll surely encounter it at some point. 

The solution to it is (often): `iterize`, which transforms a function that is meant to be applied to a single object, into a function that is meant to be applied to an array, or any iterable of such objects. 
(You might be familiar (if you use `numpy` for example) with the related concept of "vectorization", or [array programming](https://en.wikipedia.org/wiki/Array_programming).)



```python
from lined import Pipeline, iterize
from typing import Iterable

pipe = Pipeline(iterize(lambda x: x * 2), 
                iterize(lambda x: f"hello {x}"))
iterable = pipe([1, 2, 3])
assert isinstance(iterable, Iterable)  # see that the result is an iterable
list(iterable)  # consume the iterable and gather it's items
```




    ['hello 2', 'hello 4', 'hello 6']



Instead of just computing the string, say that the last step actually printed the string (called a "callback" function whose result was less important than it's effect -- like storing something, etc.).


```python
from lined import Pipeline, iterize, iterate

pipe = Pipeline(iterize(lambda x: x * 2), 
                iterize(lambda x: print(f"hello {x}")),
               )

for _ in pipe([1, 2, 3]):
    pass
```

    hello 2
    hello 4
    hello 6


It could be a bit awkward to have to "consume" the iterable to have it take effect. 

Just doing a 
```python
pipe([1, 2, 3])
```
to get those prints seems like a more natural way. 

This is where you can use `iterate`. It basically "launches" that consuming loop for you.


```python
from lined import Pipeline, iterize, iterate

pipe = Pipeline(iterize(lambda x: x * 2), 
                iterize(lambda x: print(f"hello {x}")),
                iterate
               )

pipe([1, 2, 3])
```

    hello 2
    hello 4
    hello 6


# Ramblings

## Decorating

Toddlers write lines of code. 
Grown-ups write functions. Plenty of them. 

Why break lines of code into small functions? Where to start...
- It's called modularity, and that's good
- You can reuse functions (and no, copy/paste isn't D.R.Y. -- 
and if you don't know what D.R.Y. is, 
[grow up](https://en.wikipedia.org/wiki/Don%27t_repeat_yourself!)).
- Because [7+-2](https://en.wikipedia.org/wiki/The_Magical_Number_Seven,_Plus_or_Minus_Two), 
a.k.a [chunking](https://en.wikipedia.org/wiki/Chunking_(psychology)) or Miller's Law.
- You can [decorate](https://en.wikipedia.org/wiki/Python_syntax_and_semantics#Decorators)
functions, not lines of code.

`lined` sets you up to take advantage of these goodies. 

Note this line (currently 117) of lined/base.py , in the init of Line:

    self.funcs = tuple(map(fnode, self.funcs))

That is, every function is cast to with `fnode`.

`fnode` is:

    def fnode(func, name=None):
        return Fnode(func, name)
        
and `Fnode` is just a class that "transparently" wraps the function. 
This is so that we can then use `Fnode` to do all kinds of things to the function 
(without actually touching the function itself).

    @dataclass
    class Fnode:
        func: Callable
        __name__: Optional[str] = None

    def __post_init__(self):
        wraps(self.func)(self)
        self.__name__ = self.__name__ or func_name(self.func)

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)