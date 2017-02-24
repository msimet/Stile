Using the binning module
========================

Some systematics tests will benefit from being performed on subsamples of the data binned in some 
way. For example, PSF models may be more accurate in regions of high stellar density, so producing
PSF characterization tests as a function of galactic latitude might be useful.  Shape measurement
may be more likely to fail for low signal-to-noise objects or objects which are small relative to
the PSF, so checking shape measurement accuracy in signal-to-noise or size bins may reveal problems
that cannot be seen when most of the signal comes from larger or more well-resolved objects.  Some
tests may also benefit from selection cuts, which you can also think of as a single broad bin with
a defined lower edge and an infinite upper edge.

To make these tests easier, Stile contains some simple functions to bin your data.  Two of
them--:class:`stile.BinStep <stile.binning.BinStep>` and 
:class:`stile.BinList <stile.binning.BinList>`--have simple, predefined ways of acting on your
data set. The third, :class:`stile.BinFunction <stile.binning.BinFunction>`, uses user-defined
functions to split your data.

Basic interface
---------------

To start binning your data, you create a :class:`Bin*` object that will contain binning
definitions.  For instance, if you wanted to bin the ``ra`` column in 10 bins:

>>> bin_object = stile.BinStep(field='ra', n_bins=10, low=0, high=360)

To use it, call the object; it returns a list of objects which you can apply to your data to
produce properly binned subsets.

>>> for single_bin in bin_object():
>>>     binned_data = single_bin(data)

``binned_data`` has the same format as ``data``, just with fewer rows if any of the rows were
outside the boundaries of the `single_bin` object.

The ``single_bin`` above is actually another class called a :class:`stile.binning.SingleBin`.  It
knows its boundaries and it also contains a string you can use in program outputs.

>>> single_bin = bin_object()[0]
>>> print single_bin.low
0.0
>>> print single_bin.hi
36.0
>>> print single_bin.short_name
'0'

Types of binning schemes
------------------------

The :class:`stile.BinList <stile.binning.BinList>` is the simplest class.  To create it, call it
with a list of bin edges and a field name (see the :doc:`data` documentation for more information
on field names).

>>> bin_list_object = stile.BinList(field='g1', 
      bin_list=[-10, -1, -0.5, -0.3, -0.1, -0.05, 0, 0.05, 0.1, 0.3, 0.5, 1, 10])

:class:`stile.BinStep <stile.binning.BinStep>` is also fairly simple.  It generates bins that are
equally spaced in linear or log space based on the provided arguments.  It is created using at
least three of the arguments ``low`` (the low edge of the lowest bin), ``high`` (the high edge of
the highest bin), ``n_bins`` (the number of bins to create), and ``step`` (the step size for the
bin).  All four arguments may be passed, but will be checked for consistency if so.

>>> bin_step_object = stile.BinStep(field='g2', low=-2, high=2, step=0.1)
>>> bin_step_object = stile.BinStep(field='g2', low=-2, high=2, n_bins=40)

will create identical binning schemes.

Finally, :class:`stile.BinFunction <stile.binning.BinFunction>` is available for more complex
binning schemes, especially those that rely on more than one field of data.  To use it, you will
need a function that either 1) accepts an entire data array (with fields defined as described
in :doc:`data`) and returns a vector of integers corresponding to the bin number for each row in
the data array, or 2) accepts an entire data array plus an integer bin number and returns a Boolean
mask.  You will also need to specify the maximum expected number of bins, either as an argument
passed to the constructor or as an attribute of the function.  Then, you define the bin object as

>>> bin_function_object = stile.BinFunction(func, n_bins=n_bins)

if the function returns a vector of bin indices, or

>>> bin_function_object = stile.BinFunction(func, n_bins=n_bins, returns_bools=True)

if it returns Boolean masks.  This object can be called like any other :class:`Bin*` object to
create a list of callable objects, and it will work with :func:`stile.ExpandBinList
<stile.binning.ExpandBinList>` as well.  However, the child objects it creates when you call it
don't have ``.low`` or ``.high`` attributes, so any automatic processing or looping that assumes
these attributes exist (such as for naming files) will fail.

Combining binning schemes
-------------------------

Maybe you have two binning schemes you'd like to use at once: a binning in magnitude and a binning
in galaxy weight ``'w'``.  There is a function, :func:`stile.ExpandBinList
<stile.binning.ExpandBinList>`, to automatically loop through all the possible pairs of those
binning schemes.

.. note::
  The interface for :func:`ExpandBinList` may be changing in the near future--see `Stile issue 82
  <https://github.com/msimet/Stile/issues/82>`_.

:func:`stile.ExpandBinList <stile.binning.ExpandBinList>` returns a `list of lists`.  The inner
lists are all possible pairs (tuples) of the binning schemes passed to the function.  So, for
example, given the magnitude binning object ``magnitude_bin_object`` and the galaxy weight binning
object ``weight_bin_object``, the data would be binned like this:

>>> for bin_set in stile.ExpandBinList(magnitude_bin_object, weight_bin_object):
>>>     binned_data = data
>>>     for bin in bin_set:
>>>         binned_data = bin(binned_data)

:func:`stile.ExpandBinList <stile.binning.ExpandBinList>` can accept any number of bin objects as
arguments (including none).  In the lists it returns, the first object passed as an argument
changes most slowly, followed by the second, etc (so the first item in the list it returns will be
``[magnitude_bin_object_0, weight_bin_object_0]``, the second will be
``[magnitude_bin_object_0, weight_bin_object_1]``, etc).


