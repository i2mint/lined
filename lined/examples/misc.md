# Getting a DAG from a module

```python
from lined import Pipe, iterize, mk_filter
from lined.tools import negate, iterize, true_no_matter_what


def module_callables(module, filt=true_no_matter_what):
    f = Pipe(
        dir,
        mk_filter(lambda x: not x.startswith('__')),
        mk_filter(filt),
        iterize(lambda a: getattr(module, a)),
        mk_filter(callable),
        mk_filter(
            lambda obj: getattr(obj, '__module__', None) == module.__name__),
        list
    )
    return f(module)


#     list


import wealth.aligned_umap_analysis as module

funcs = module_callables(module, filt=lambda x: not x.startswith('assert'))

from meshed import DAG
from meshed.util import conservative_parameter_merge
from functools import partial

dag = DAG(funcs,
          parameter_merge=partial(conservative_parameter_merge,
                                  same_default=False,
                                  same_annotation=False)
          )
dag.dot_digraph(start_lines=['rankdir=LR;'])
```
