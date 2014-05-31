def test_Parser():
    pass

def test_ExpandBinList():
    # Needs a callable object
    def return_objs(x,n):
        def func():
            return [str(nn)+x for nn in range(n)]
        return func
    results = stile.ExpandBinList(return_objs('a',3),return_objs('b',2),return_objs('c',4))
    numpy.testing.assert_equal(results,[('1a','1b','1c'),('1a','1b','2c'),('1a','1b','3c'),
                                        ('1a','1b','4c'),('1a','2b','1c'),('1a','2b','2c'),
                                        ('1a','2b','3c'),('1a','2b','4c'),('2a','2b','1c'),
                                        ('2a','1b','2c'),('2a','1b','3c'),('2a','1b','4c'),
                                        ('2a','2b','1c'),('2a','2b','2c'),('2a','2b','3c'),
                                        ('2a','2b','4c'),('3a','1b','1c'),('3a','1b','2c'),
                                        ('3a','1b','3c'),('3a','1b','4c'),('3a','2b','1c'),
                                        ('3a','2b','2c'),('3a','2b','3c'),('3a','2b','4c')]
    numpy.testing.assert_equal(stile.ExpandBinList(None),[])
    numpy.testing.assert_equal(stile.ExpandBinList([]),[])
    bin_obj0 = stile.BinStep('column_0',low=0,high=6,n_bins=2)
    bin_obj1 = stile.BinList('column_1',[0,2,4])
    results = stile.ExpandBinList([bin_obj0,bin_obj1])
    expected_results = [(stile.binning.SingleBin('column_0',low=0,high=3,short_name='b'),
                         stile.binning.SingleBin('column_1',low=0,high=2,short_name='b')),
                        (stile.binning.SingleBin('column_0',low=0,high=3,short_name='b'),
                         stile.binning.SingleBin('column_1',low=2,high=4,short_name='b')),
                        (stile.binning.SingleBin('column_0',low=0,high=3,short_name='b'),
                         stile.binning.SingleBin('column_1',low=4,high=6,short_name='b')),
                        (stile.binning.SingleBin('column_0',low=3,high=6,short_name='b'),
                         stile.binning.SingleBin('column_1',low=0,high=2,short_name='b')),
                        (stile.binning.SingleBin('column_0',low=3,high=6,short_name='b'),
                         stile.binning.SingleBin('column_1',low=2,high=4,short_name='b')),
                        (stile.binning.SingleBin('column_0',low=3,high=6,short_name='b'),
                         stile.binning.SingleBin('column_1',low=4,high=6,short_name='b'))]
    numpy.testing.assert_equal(len(results),len(expected_results))
    from test_binning import compare_single_bin
    compare_results = [(compare_single_bin(rpair[0],epair[0]),
                        compare_single_bin(rpair[1],epair[1]))
                                                for rpair, epair in zip(results,expected_results)]
    numpy.testing.assert_equal(compare_results,[(True,True)]*len(results))
    numpy.testing.assert_raises(ValueError,stile.ExpandBinList,bin_obj0,bin_obj1)

def test_GetVectorType(x):
    vec0 = ['hello','goodbye'] 
    vec1 = [0,1,2]
    vec2 = [1.,2.,3.]
    vec3 = [2.5,3.5,4.5]
    vec4 = ['nan', 6, 8]
    vec5 = ['inf', 6, 8]
    vec6 = [complex(1,0),complex(2,0)]
    vec7 = [complex(1,1),complex(2,2)]
    vec8 = ['"Hello"',u'\u00abgoodbye\u00bb'] # Unicode symbols for French-style quotation marks
    vec9 = [stile.binning.SingleBin('column_0',low=0,high=1,short_name='b')
    vec10 = 3
    vec11 = 5.5
    # TODO: single- vs double-precision, once that's better implemented in GetVectorType() itself
    numpy.testing.assert_equal(stile.GetVectorType(vec0),'S7')
    numpy.testing.assert_equal(stile.GetVectorType(vec1),'l')
    numpy.testing.assert_equal(stile.GetVectorType(vec2),'l')
    numpy.testing.assert_equal(stile.GetVectorType(vec3),'d')
    numpy.testing.assert_equal(stile.GetVectorType(vec4),'d')
    numpy.testing.assert_equal(stile.GetVectorType(vec5),'d')
    numpy.testing.assert_equal(stile.GetVectorType(vec6),'l')
    numpy.testing.assert_raises(RuntimeWarning,stile.GetVectorType,vec7)
    numpy.testing.assert_equal(stile.GetVectorType(vec8),'S9')
    numpy.testing.assert_raises(RuntimeWarning,stile.GetVectorType,vec9)
    numpy.testing.assert_equal(stile.GetVectorType(vec10),'l')
    numpy.testing.assert_equal(stile.GetVectorType(vec11),'d')
    
def test_MakeRecarray(d,fields=None,only_floats=False):
    pass
        
def test_MakeFiles(dh, data, data2=None, random=None, random2=None):
    pass

def main():
    test_Parser()
    test_ExpandBinList()
    test_GetVectorType()
    test_MakeRecarray()
    test_MakeFiles()
