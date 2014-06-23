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

