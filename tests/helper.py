import numpy

def FormatSame(arr1,arr2):
    """
    Do some NumPy data type tricks to make sure that two formatted arrays can be compared to each
    other.

    In short:
        - Convert both arrays to numpy.arrays if they were something else (mostly for the case where
          you in fact have a FITSrec)
        - Check that both arrays have the same number of columns and the same shape otherwise
        - Ensure that both have the same byteorder (so you can compare FITS files, which have fixed
          byteorder, to Python arrays, which use whatever the installation says to use)
        - Change the field names to be the same for both arrays

    Then return a tuple of the two arrays, which can then be passed to the numpy.testing functions.
    """
    if not arr1.shape==arr2.shape:
        return (arr1, arr2)
    arr1 = numpy.array(arr1) # to protect against FITSrecs
    if not len(arr1.dtype.names)==len(arr2.dtype.names) or not arr1.shape==arr2.shape:
        return (arr1, arr2)
    arr1 = arr1.astype(arr1.dtype.newbyteorder('='))
    arr2 = arr2.astype(arr2.dtype.newbyteorder('='))
    arr2.dtype.names = arr1.dtype.names
    return (arr1, arr2)

