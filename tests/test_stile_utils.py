import numpy
try:
    import stile
except ImportError:
    import sys
    sys.path.append('..')
    import stile


def test_Parser():
    pass

def test_ExpandBinList():
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
    #numpy.testing.assert_raises(ValueError,stile.ExpandBinList,bin_obj0,bin_obj1)

def test_FormatArray():
    #FormatArray(d,fields=None,only_floats=False)
    pass

def test_OSFile():
    pass
    
def test_MakeFiles():
    #MakeFiles(dh, data, data2=None, random=None, random2=None):
    pass

def main():
    test_Parser()
    test_ExpandBinList()
    test_GetVectorType()
    test_FormatArray()
    test_OSFile()
    test_MakeFiles()

if __name__=='__main__':
    main()
