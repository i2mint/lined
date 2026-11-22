# lined.recipes

Recipes to demo uses of lined functionalities

### Functions

| [`mk_transposer_to_array`](#lined.recipes.mk_transposer_to_array)([dtype])   | Make a transposer that transposes an iterable of `n` iterables of size `k` into an iterable of `k` arrays of size `n`.   |
|------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------|

### lined.recipes.mk_transposer_to_array(dtype=None)

Make a transposer that transposes an iterable of `n` iterables of size `k`
into an iterable of `k` arrays of size `n`.

## Commented out to avoid CI choking on the absence of numpy
# >>> from numpy import array
# >>> transpose = mk_transposer_to_array(dtype=int)
# >>> transpose(iter([range(1,4), range(4, 7)]))
# # array([[1, 4],
# #        [2, 5],
# #        [3, 6]])
