"""
stile_utils.py: Various utilities for the Stile pipeline.  Includes input parsing and some numerical helper
functions.
"""

import numpy


def Parser():
    """
    Returns an argparse Parser object with input args used by Stile and TreeCorr.
    """
    import treecorr_utils
    import argparse
    p = argparse.Parser(parent=treecorr_utils.Parser())
    #TODO: add, obviously, EVERYTHING ELSE
    return p


def FormatArray(d, fields=None):
    """
    Turn a regular NumPy array of arbitrary types into a formatted array, with optional field name
    description.

    This function uses the existing dtype of the array ``d``.  This means that arrays of
    heterogeneous objects may not return the dtype you expect (for example, ``int``\s will be
    converted to ``float``\s if there are floats in the array, or all numbers will be converted to
    strings if there are any strings in the array).  Predefining the format or using a function like
    ``numpy.genfromtxt()`` will prevent these issues, as will reading from a FITS file.

    :param d:      A NumPy array.
    :param fields: A dictionary whose keys are the names of the fields you'd like for the output
                   array, and whose values are field numbers (starting with 0) whose names those
                   keys should replace (or, if the array is already formatted, the existing field
                   names the keys should replace); alternately, a list with the same length as the
                   rows of ``d``. [default: None]
    :returns:      A formatted numpy array with the same shape as ``d`` except that the innermost
                   dimension has turned into a record field if it was not already one, optionally
                   with field names appropriately replaced.
    """
    # We want arrays to be numpy.arrays with field access (so we can say d['ra'] or something like
    # that).  In order for these to be created correctly, two conditions have to be met:
    # - The array needs to be initialized with the innermost dimension as a tuple rather than
    #   a list or other array, so NumPy knows that it's a record field;
    # - The array has to be created with a dtype that indicates there are multiple fields.  For
    #   convenience I'm going to do this as a single string of the form '?,?,?[...]' where each
    #   question mark is a single character (plus optional width for strings/voids) denoting what
    #   kind of data to expect. (This is the "array-protocol type string", see
    #   http://docs.scipy.org/doc/numpy/reference/arrays.dtypes.html)
    if not hasattr(d, 'dtype'):
        # If it's not an array, make it one.
        d = numpy.array(d)
    if not d.dtype.names:
        # If it is an array, but doesn't have a "names" attribute, that means it doesn't have
        # records/fields.  So we need to reformat the array.  Given the difficulty of generating
        # an individual dtype for each field, we'll just use the dtype of the overall array for
        # every entry, which involves no casting of types.
        d_shape = d.shape
        if len(d_shape) == 1:  # Assume this was a single row (not a set of 1-column rows)
            d = numpy.array([d])
            d_shape = d.shape
        # Cast this into a 2-d array
        new_d = d.reshape(-1, d_shape[-1])
        # Generate the dtype string
        if isinstance(d.dtype, str):
            dtype = ','.join([d.dtype]*len(d[0]))
        else:
            dtype_char = d.dtype.char
            if dtype_char == 'S' or dtype_char == 'O' or dtype_char == 'V' or dtype_char == 'U':
                dtype = ','.join([d.dtype.str]*len(new_d[0]))  # need the width as well as the char
            else:
                dtype = ','.join([dtype_char]*len(new_d[0]))
        # Make a new array with each row turned into a tuple and the correct dtype
        d = numpy.array([tuple(nd) for nd in new_d], dtype=dtype)
        if len(d_shape) > 1:
            # If this was a more-than-2d array, reshape it back to that original form, minus the
            # dimension we turned into a record (which will no longer appear in the shape).
            d = d.reshape(d_shape[:-1])
    if fields:
        # If the "fields" parameter was set, rewrite the numpy.dtype.names attribute to be the
        # field specification we want.
        if isinstance(fields, dict):
            names = list(d.dtype.names)
            for key in fields:
                names[fields[key]] = key
            d.dtype.names = names
        elif len(fields) == len(d.dtype.names):
            d.dtype.names = fields
        else:
            raise RuntimeError('Cannot use given fields: '+str(fields))
    return d


class Stats:
    """A Stats object can carry around and output the statistics of some array.

    Currently it can carry around two types of statistics:

    (1) Basic array statistics: typically one would use length (N), min, max, median, mean, standard
        deviation (stddev), variance, median absolute deviation ('mad') as defined using the
        ``simple_stats`` option at initialization.

    (2) Percentiles: the value at a given percentile level.

    The :class:`StatSysTest <stile.sys_tests.StatSysTest>` class can be used to create and populate values for one of
    these objects.  If you want to change the list of simple statistics, it's only necessary to
    change the code there, not here.
    """

    def __init__(self, simple_stats):
        self.simple_stats = simple_stats
        for stat in self.simple_stats:
            init_str = 'self.' + stat + '=None'
            exec init_str

        self.percentiles = None
        self.values = None

    def __str__(self):
        """This routine will print the contents of the ``Stats`` object in a nice format.

        We assume that the ``Stats`` object was created by a :class:`StatSysTest`, so that certain
        sanity checks have already been done (e.g., self.percentiles, if not None, is iterable)."""
        # Preamble:
        ret_str = 'Summary statistics:\n'

        # Loop over simple statistics and print them, if not None.  Generically if one is None then
        # all will be, so just check one.
        test_str = "test_val = self."+("%s"%self.simple_stats[0])
        exec test_str
        if test_val is not None:
            for stat in self.simple_stats:
                this_string = 'this_val = self.'+stat
                exec this_string
                ret_str += '\t%s: %f\n'%(stat, this_val)
            ret_str += '\n'

        # Loop over combinations of percentiles and values, and print them.
        if self.percentiles is not None:
            ret_str += 'Below are lists of (percentile, value) combinations:\n'
            for index in range(len(self.percentiles)):
                ret_str += '\t%f %f\n'%(self.percentiles[index], self.values[index])

        return ret_str

fieldNames = {
    'g1': 'g1, a shear component in the ra direction',
    'g2': 'g2, a shear component 45 degrees from the ra direction',
    'sigma': 'a size parameter for objects with dimension [length] in arbitrary units',
    'psf_g1': 'the g1 of the psf at the location of this object',
    'psf_g2': 'the g2 of the psf at the location of this object',
    'psf_sigma': 'the sigma of the psf at the location of this object',
    'w': 'the weight to apply per object',
    'z': 'the redshift of the object'}

objectNames = {
    'galaxy': 'galaxy data',
    'star': 'star data',
    'galaxy lens': 'galaxies to be used as lenses in galaxy-galaxy lensing',
    'star PSF': 'stars used in PSF determination',
    'star bright': 'especially bright stars',
    'galaxy random': 'random catalog corresponding to the "galaxy" sample',
    'star random': 'random catalog corresponding to the "star" sample'
}

