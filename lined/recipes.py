from functools import partial
from lined import Line
from lined.tools import map_star, iterize

transposer = map_star(zip)
transposer.__name__ = 'transposer'


def mk_transposer_to_array(dtype=None):
    """Make a transposer that ransposes an iterable of n iterables of size k into an iterable of k arrays of size n.
    """
    from numpy import array
    return Line(map_star(zip), iterize(partial(array, dtype=dtype), name='transpose_to_array'))

