import numpy
import time
try:
    import stile
except ImportError:
    import sys
    sys.path.append('..')
    import stile

def test_ExpandBinList():
    t0 = time.time()
    # Needs a callable object
    def return_objs(x,n):
        def func():
            return [str(nn)+x for nn in range(n)]
        return func
    results = stile.ExpandBinList([return_objs('a',3),return_objs('b',2),return_objs('c',4)])
    numpy.testing.assert_equal(results,[('0a','0b','0c'),('0a','0b','1c'),('0a','0b','2c'),
                                        ('0a','0b','3c'),('0a','1b','0c'),('0a','1b','1c'),
                                        ('0a','1b','2c'),('0a','1b','3c'),('1a','0b','0c'),
                                        ('1a','0b','1c'),('1a','0b','2c'),('1a','0b','3c'),
                                        ('1a','1b','0c'),('1a','1b','1c'),('1a','1b','2c'),
                                        ('1a','1b','3c'),('2a','0b','0c'),('2a','0b','1c'),
                                        ('2a','0b','2c'),('2a','0b','3c'),('2a','1b','0c'),
                                        ('2a','1b','1c'),('2a','1b','2c'),('2a','1b','3c')])
    numpy.testing.assert_equal(stile.ExpandBinList(None),[])
    numpy.testing.assert_equal(stile.ExpandBinList([]),[])
    bin_obj0 = stile.BinStep('column_0',low=0,high=6,n_bins=2)
    bin_obj1 = stile.BinList('column_1',[0,2,4])
    results = stile.ExpandBinList([bin_obj0,bin_obj1])
    expected_results = [(stile.binning.SingleBin('column_0',low=0,high=3,short_name='b'),
                         stile.binning.SingleBin('column_1',low=0,high=2,short_name='b')),
                        (stile.binning.SingleBin('column_0',low=0,high=3,short_name='b'),
                         stile.binning.SingleBin('column_1',low=2,high=4,short_name='b')),
                        (stile.binning.SingleBin('column_0',low=3,high=6,short_name='b'),
                         stile.binning.SingleBin('column_1',low=0,high=2,short_name='b')),
                        (stile.binning.SingleBin('column_0',low=3,high=6,short_name='b'),
                         stile.binning.SingleBin('column_1',low=2,high=4,short_name='b'))]
    numpy.testing.assert_equal(len(results),len(expected_results))
    from test_binning import compare_single_bin
    [(compare_single_bin(rpair[0],epair[0]), compare_single_bin(rpair[1],epair[1]))
                                                  for rpair, epair in zip(results,expected_results)]
    try: 
        numpy.testing.assert_raises(TypeError,stile.ExpandBinList,bin_obj0,bin_obj1)
    except ImportError:
        print "nose is required for numpy.testing.assert_raises tests"
    t1 = time.time()
    print "Time to test ExpandBinList: ", 1000*(t1-t0), "ms"

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
