from functools import partial, wraps


def keys_extractor(keys):
    def extract(x):
        return tuple((x[i] for i in keys))

    return extract


def items(mapping):
    """Get an items generator from a mapping"""
    return mapping.items()


# Function transformers ###################################################################

def func_name(func, dflt_name='no_name'):
    return getattr(func, '__name__', dflt_name)


def extra_wraps(func, name=None, doc_prefix=""):
    func.__name__ = name or func_name(func)
    func.__doc__ = doc_prefix + getattr(func, '__name__', '')
    return func


def mywraps(func, name=None, doc_prefix=""):
    def wrapper(wrapped):
        return extra_wraps(wraps(func)(wrapped), name=name, doc_prefix=doc_prefix)

    return wrapper


def generator_version(func, name=None):
    """"""
    wrapper = mywraps(func, name=name,
                      doc_prefix=f"generator version of {func_name(func)}:\n")
    return wrapper(partial(map, func))


def singularize_arg_input(func):
    """Make a func(args) function out of a func(*args) one"""

    @mywraps(func, doc_prefix=f"singularize_arg_input version of {func_name(func)}")
    def func_with_single_arg_input(args):
        return func(*args)

    return func_with_single_arg_input
