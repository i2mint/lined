"""
Making pipelines easily, and being able to visualize and diagnose them.
"""
from lined.base import (
    Line,
    Pipeline,
    LayeredPipeline,
    LineSentineled,
    LineParametrized,
    ParallelFuncs,
    log_calls,
)

from lined.simple import Pipe
from lined.tools import (
    Command,
    items,
    iterize,
    iterate,
    tail_io,
    map_star,
    singularize_arg_input,  # old alias of map_star,
    with_cursor,
    BufferStats,
    mk_filter,
    side_call,
    print_and_pass_on,
)
from lined.util import partial_plus, n_required_args, dot_to_ascii
