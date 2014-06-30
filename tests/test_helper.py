import numpy

def format_same(arr1,arr2):
    if not arr1.shape==arr2.shape:
        return False
    arr1 = numpy.array(arr1) # to protect against FITSrecs
    if not len(arr1.dtype.names)==len(arr2.dtype.names):
        return False
    arr1 = arr1.astype(arr1.dtype.newbyteorder('='))
    arr2 = arr2.astype(arr2.dtype.newbyteorder('='))
    arr2.dtype.names = arr1.dtype.names
    return (arr1,arr2)
    
