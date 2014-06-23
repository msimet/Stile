"""@file stile_utils.py
Various utilities for the Stile pipeline.  Includes input parsing and some numerical helper 
functions.
"""

import numpy
import weakref
import os

def Parser():
    """
    Returns an argparse Parser object with input args used by Stile and corr2.
    """
    import corr2_utils
    import argparse
    p = argparse.Parser(parent=corr2_utils.Parser())
    #TODO: add, obviously, EVERYTHING ELSE
    return p

def ExpandBinList(bin_list):
    """
    Take a list of Bin* objects, and expand them to return a list of SimpleBins to step through.  
    E.g., if the user passes a list :
        >>> bin_list = [ListBin0, ListBin1]
    where ListBin1.n_bins = 2 and ListBin2.n_bins = 3, then calling this function will return
        >>> [[SimpleBinObject_0_0, SimpleBinObject_1_0],
             [SimpleBinObject_0_0, SimpleBinObject_1_1],
             [SimpleBinObject_0_0, SimpleBinObject_1_2],
             [SimpleBinObject_0_1, SimpleBinObject_1_0],
             [SimpleBinObject_0_1, SimpleBinObject_1_1],
             [SimpleBinObject_0_1, SimpleBinObject_1_2]].
    The first item in the list changes most slowly, then the second item, etc.
             
    @param bin_list  A list of Bin-type objects, such as the ones in binning.py, which when called
                     return a list of items which behave like SimpleBins.  (No checks are made in
                     this function that these are valid items.)
    @returns         A list of lists of SimpleBins (or other items returned by calling the input
                     items in bin_list), properly nested.
    """
    if not bin_list:
        return []
    # A copy of bin_list that we can alter without altering the parent list, but which doesn't 
    # duplicate the objects in bin_list. (I think.)
    bl = bin_list[:]
    data_bins = [[]]
    while bl:
        this_bin = bl.pop()
        data_bins = [[bin]+d for bin in this_bin() for d in data_bins]
    return data_bins


def FormatArray(d,fields=None,only_floats=False):
    """
    Turn a regular NumPy array of arbitrary types into a formatted array, with optional field name 
    description.

    @param d      A NumPy array
    @param fields A dictionary whose keys are the names of the fields you'd like for the output 
                  array, and whose values are field numbers (starting with 0) whose names those keys should replace; alternately, a list with the same length as the rows of d. 
                  (default: None)
    @returns      A formatted numpy array with the same shape as d except that the innermost 
                  dimension has turned into a record field, optionally with field names 
                  appropriately replaced.
    """
    if hasattr(d,'dtype') and hasattr(d.dtype,'names') and d.dtype.names:
        pass
    else:
        d_shape = d.shape
        new_d = d.reshape(-1,d_shape[-1])
        new_d = numpy.array(d)
        if isinstance(d.dtype,str):
            dtype = ','.join([d.dtype]*len(d[0]))
        else:
            dtype = ','.join([d.dtype.char]*len(d[0]))
        d = numpy.array([tuple(nd) for nd in new_d],dtype=dtype)
        if len(d_shape)>1:
            d = d.reshape(d_shape[:-1])
    if fields:
        if isinstance(fields,dict):
            names = list(d.dtype.names)
            for key in fields:
                names[fields[key]]=key
            d.dtype.names = names
        elif len(fields)==len(d.dtype.names):
            d.dtype.names = fields
        else:
            raise RuntimeError('Cannot use given fields: '+str(fields))
    return d

class Stats:
    """A Stats object can carry around and output the statistics of some array.

    Currently it can carry around two types of statistics:

    (1) Basic array statistics: typically one would use length (N), min, max, median, mean, standard
        deviation (stddev), variance, median absolute deviation ('mad') as defined using the
        `simple_stats` option at initialization.

    (2) Percentiles: the value at a given percentile level.

    The StatSysTest class in `sys_tests.py` can be used to create and populate values for one of
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
        """This routine will print the contents of the Stats object in a nice format.

        We assume that the Stats object was created by a StatSysTest, so that certain sanity checks
        have already been done (e.g., self.percentiles, if not None, is iterable)."""
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
                ret_str += '\t%f %f\n'%(self.percentiles[index],self.values[index])

        return ret_str
