
# lined

Building simple pipelines, simply.


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




