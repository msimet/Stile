import numpy
import time
try:
    import stile
except ImportError:
    import sys
    sys.path.append('..')
    import stile

def test_FormatArray():
    t0 = time.time()
    data_raw = [(1,'hello',2.0),(3,'boo',5.0)]
    data0 = numpy.array(data_raw)
    data1 = numpy.array(data_raw,dtype='l,S5,d')
    old_dnames = data1.dtype.names
    result = stile.FormatArray(data0)
    numpy.testing.assert_equal(result,data1.astype(result.dtype))
    result = stile.FormatArray(data1)
    numpy.testing.assert_equal(result,data1)
    result = stile.FormatArray(data0,fields=['one','two','three'])
    data1.dtype.names = result.dtype.names
    numpy.testing.assert_equal(result,data1.astype(result.dtype))
    numpy.testing.assert_equal(result.dtype.names,['one','two','three'])
    result2 = stile.FormatArray(data0,fields={'one': 0, 'two': 1, 'three': 2})
    numpy.testing.assert_equal(result,result2)
    data1.dtype.names = old_dnames
    result = stile.FormatArray(data1,fields=['one','two','three'])
    data1.dtype.names = result.dtype.names
    numpy.testing.assert_equal(result,data1.astype(result.dtype))
    numpy.testing.assert_equal(result.dtype.names,['one','two','three'])
    data1.dtype.names = old_dnames
    result2 = stile.FormatArray(data1,fields={'one': 0, 'two': 1, 'three': 2})
    numpy.testing.assert_equal(result,result2)
    try:
        numpy.testing.assert_raises(TypeError,[1,2])
    except ImportError:
        print "nose is required for numpy.testing.assert_raises tests"
    t1 = time.time()
    print "Time to test FormatArray: ", 1000*(t1-t0), "ms"
    
def main():
    test_ExpandBinList()
    test_FormatArray()

if __name__=='__main__':
    main()
