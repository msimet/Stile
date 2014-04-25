"""@file stile_utils.py
Various utilities for the Stile pipeline.  Includes input parsing and some numerical helper 
functions.
"""

import numpy

def parser():
    import corr2_utils
    import argparse
    p = argparse.Parser(parent=corr2_utils.parser)
    #TODO: add, obviously, EVERYTHING ELSE
    return p

def get_vector_type(x):
    """
    Figure out the least-precise type that can represent an entire vector of data.

    @param x An iterable of data (a single scalar value will likely result in errors)
    @returns a string usable in a NumPy dtype declaration for x
    """
    if not hasattr(x,'astype'):
        x = numpy.array(x)
    #TODO: better data types here
    for var_type, type_name in [(int,'l'),(float,'d')]:
        try:
            x.astype(var_type)
            return type_name
        except:
            pass
    return 'S'+str(max([len(xx) for xx in x]))

def make_recarray(d,fields=None):
    """
    Turn a regular NumPy array into a numpy.recarray, with optional field name description.
    @param d      A NumPy array (or other iterable which satisfies hasattr(d,'shape')).
    @param fields A dictionary whose keys are the names of the fields you'd like for the output 
                  array, and whose values are field numbers (starting with 0) whose names those keys 
                  should replace. (default: None)
    @returns      A numpy.recarray with the same shape as d except that the innermost dimension 
                  has turned into a record field, optionally with field names appropriately replaced
    """
    if hasattr(d,'dtype') and hasattr(d,'dtype.names'):
        pass
    else:
        d_shape = d.shape
        new_d = d.reshape(-1,d_shape[-1])
        new_d = numpy.array(d)
        types = ','.join([get_vector_type(new_d[:,i]) for i in range(len(new_d[0]))])
        d = numpy.array([tuple(nd) for nd in new_d],dtype=types).reshape(d_shape[:-1])
    if fields:
        names = d.dtype.names
        for key in fields:
            names[fields[key]]=key
        d.dtype.names=names
    return d
    
        
    
