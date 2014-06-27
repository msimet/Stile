import numpy
import sys

try:
    import stile
except ImportError:
    sys.path.append('..')
    import stile

import time

# Test values for later tests
bin_array_1 = [[0.5],[1.5],[2.5],[3.5],[4.5]]
bin_array_1 = numpy.array([tuple(b) for b in bin_array_1],dtype=[('field_0',float)] )
bin_array_2 = [[1],[2],[3],[4],[5]]
bin_array_2 = numpy.array([tuple(b) for b in bin_array_2],dtype=[('field_0',float)] )
bin_array_3 = [[0.5],[0.5],[5.5],[4.5],[3.5]]
bin_array_3 = numpy.array([tuple(b) for b in bin_array_3],dtype=[('field_0',float)] )
bin_array_4 = [0.5,1.5,2.5,3.5,4.5]
bin_array_4 = numpy.array([tuple(bin_array_4)],
                          dtype=[('field_0',float), ('field_1',float), ('field_2',float),    
                                 ('field_3',float), ('field_4',float)] )
bin_array_5 = [[1.,2.],[3.,4.],[5.,6.]]
bin_array_5 = numpy.array([tuple(b) for b in bin_array_5],
                          dtype=[('field_0',float),('field_1',float)] )
bin_array_6 = [[-1],[1]]
bin_array_6 = numpy.array([tuple(b) for b in bin_array_6],dtype=[('field_0',float)] )

def binfunction(x):
    return numpy.ceil(x)

def compare_single_bin(b1,b2):
    assert b1.field==b2.field
    numpy.testing.assert_almost_equal([b1.low,b1.high],[b2.low,b2.high])

def test_BinStep_SingleBin_creation():
    t0 = time.time()
    lhs = stile.BinStep('field_0',low=0,high=6,step=1)
    lhn = stile.BinStep('field_0',low=0,high=6,n_bins=6)
    lsn = stile.BinStep('field_0',low=0,step=1,n_bins=6)
    hsn = stile.BinStep('field_0',high=6,step=1,n_bins=6)
    reverse_lhs = stile.BinStep('field_0',low=6,high=0,step=-1)
    
    expected_obj_list = [stile.binning.SingleBin('field_0',low=0,high=1,short_name='b'), 
                         stile.binning.SingleBin('field_0',low=1,high=2,short_name='b'),
                         stile.binning.SingleBin('field_0',low=2,high=3,short_name='b'),
                         stile.binning.SingleBin('field_0',low=3,high=4,short_name='b'),
                         stile.binning.SingleBin('field_0',low=4,high=5,short_name='b'),
                         stile.binning.SingleBin('field_0',low=5,high=6,short_name='b')]

    names = ["passed low, high, and step",
             "passed low, high, and n_bins",
             "passed low, step, and n_bins",
             "passed high, step, and n_bins",
             "passed low, high, and step with low and high reversed"]
    objs = [lhs,lhn,lsn,hsn,reverse_lhs]
    for obj in objs:
        obj_list = obj()
        if obj==reverse_lhs:
            obj_list.reverse()
        if not len(obj_list)==6:
            raise AssertionError('BinStep ('+name+') created wrong number of SingleBins!')
        try:
            [compare_single_bin(obj_list[i],expected_obj_list[i]) for i in range(len(obj_list))]
        except AssertionError:
            raise AssertionError('BinStep ('+name+') created incorrect SingleBins!')


    lhs = stile.BinStep('field_0',low=0.25,high=8,step=numpy.log(2.),use_log=True)
    lhn = stile.BinStep('field_0',low=0.25,high=8,n_bins=5,use_log=True)
    lsn = stile.BinStep('field_0',low=0.25,step=numpy.log(2.),n_bins=5,use_log=True)
    hsn = stile.BinStep('field_0',high=8,step=numpy.log(2.),n_bins=5,use_log=True)
    reverse_lhs = stile.BinStep('field_0',low=8,high=0.25,step=-numpy.log(2.),use_log=True)        
    
    expected_obj_list = [stile.binning.SingleBin('field_0',low=0.25,high=0.5,short_name='b'), 
                         stile.binning.SingleBin('field_0',low=0.5,high=1.,short_name='b'),
                         stile.binning.SingleBin('field_0',low=1.,high=2.,short_name='b'),
                         stile.binning.SingleBin('field_0',low=2.,high=4.,short_name='b'),
                         stile.binning.SingleBin('field_0',low=4.,high=8.,short_name='b')]

    names = ["passed low, high, and step",
             "passed low, high, and n_bins",
             "passed low, step, and n_bins",
             "passed high, step, and n_bins",
             "passed low, high, and step with low and high reversed"]
    objs = [lhs,lhn,lsn,hsn,reverse_lhs]
    for obj,name in zip(objs,names):
        obj_list = obj()
        if obj==reverse_lhs:
            obj_list.reverse()
        if not len(obj_list)==5:
            raise AssertionError('Log BinStep ('+name+') created wrong number of SingleBins!')
        try:
            [compare_single_bin(obj_list[i],expected_obj_list[i]) for i in range(len(obj_list))]
        except AssertionError:
            raise AssertionError('Log BinStep ('+name+') created incorrect SingleBins!')

def test_BinList_SingleBin_creation():
    t0 = time.time()
    obj = stile.BinList('field_0',[0,1.1,1.9,3.0,4.0,5.0,6.5])
    
    expected_obj_list = [stile.binning.SingleBin('field_0',low=0,high=1.1,short_name='b'),
                         stile.binning.SingleBin('field_0',low=1.1,high=1.9,short_name='b'),
                         stile.binning.SingleBin('field_0',low=1.9,high=3,short_name='b'),
                         stile.binning.SingleBin('field_0',low=3,high=4,short_name='b'),
                         stile.binning.SingleBin('field_0',low=4,high=5,short_name='b'),
                         stile.binning.SingleBin('field_0',low=5,high=6.5,short_name='b')]

    obj_list = obj()

    if not len(obj_list)==6:
        raise AssertionError('BinList created wrong number of SingleBins!')
    try:
        [compare_single_bin(obj_list[i],expected_obj_list[i]) for i in range(len(obj_list))]
    except AssertionError:
        raise AssertionError('BinList created incorrect SingleBins!')

    obj = stile.BinList('field_0',[6.5,5.0,4.0,3.0,1.9,1.1,0])
    obj_list = obj()
    obj_list.reverse()
    if not len(obj_list)==6:
        raise AssertionError('Reversed BinList created wrong number of SingleBins!')
    try:
        [compare_single_bin(obj_list[i],expected_obj_list[i]) for i in range(len(obj_list))]
    except AssertionError:
        raise AssertionError('Reversed BinList created incorrect SingleBins!')
    

def test_BinStep_linear():
    t0 = time.time()
    lhs = stile.BinStep('field_0',low=0,high=6,step=1)
    lhn = stile.BinStep('field_0',low=0,high=6,n_bins=6)
    lsn = stile.BinStep('field_0',low=0,step=1,n_bins=6)
    hsn = stile.BinStep('field_0',high=6,step=1,n_bins=6)
    reverse_lhs = stile.BinStep('field_0',low=6,high=0,step=-1)
    
    names = ["passed low, high, and step",
             "passed low, high, and n_bins",
             "passed low, step, and n_bins",
             "passed high, step, and n_bins",
             "passed low, high, and step with low and high reversed"]
    objs = [lhs,lhn,lsn,hsn,reverse_lhs]

    # Expected results; each item of the list is the result of the n-th SingleBin.
    # Formatted arrays don't compare properly to non-formatted arrays, so we use slices of the
    # original array to ensure the formatting matches properly.
    expected_bin_array_1 = [bin_array_1[0],bin_array_1[1],bin_array_1[2],bin_array_1[3],
                            bin_array_1[4],bin_array_1[:0]]
    expected_bin_array_2 = [bin_array_2[:0],bin_array_2[0],bin_array_2[1],bin_array_2[2],
                            bin_array_2[3],bin_array_2[4]]
    expected_bin_array_3 = [bin_array_3[0:2],bin_array_3[:0],bin_array_3[:0],bin_array_3[4],
                            bin_array_3[3],bin_array_3[2]]
    expected_bin_array_4 = [bin_array_4[0],bin_array_4[:0],bin_array_4[:0],
                            bin_array_4[:0],bin_array_4[:0],bin_array_4[:0]]
    expected_bin_array_5 = [bin_array_5[:0],bin_array_5[0],bin_array_5[:0],
                            bin_array_5[1],bin_array_5[:0],bin_array_5[2]]
    expected_bin_array_6 = [bin_array_6[:0],bin_array_6[1],bin_array_6[:0],
                            bin_array_6[:0],bin_array_6[:0],bin_array_6[:0]]

    for obj, name in zip(objs,names):
        err_msg = ("BinStep test ("+name+
                   ") failed to produce correct binning for array %s in bin # %i")
        obj_list = obj()
        if len(obj_list)!=6:
            raise RuntimeError('Wrong number of bins created from BinStep with '+
                                name+': '+str(len(obj_list)))
        if obj==reverse_lhs:
            obj_list.reverse()
        for i,singlebin in enumerate(obj_list):
            results = bin_array_1[singlebin(bin_array_1)]
            numpy.testing.assert_equal(results,expected_bin_array_1[i],
                                       err_msg=err_msg%(bin_array_1,i))
            results = bin_array_2[singlebin(bin_array_2)]
            numpy.testing.assert_equal(results,expected_bin_array_2[i],
                                       err_msg=err_msg%(bin_array_2,i))
            results = bin_array_3[singlebin(bin_array_3)]
            numpy.testing.assert_equal(results,expected_bin_array_3[i],
                                       err_msg=err_msg%(bin_array_3,i))
            results = bin_array_4[singlebin(bin_array_4)]
            numpy.testing.assert_equal(results,expected_bin_array_4[i],
                                       err_msg=err_msg%(bin_array_4,i))
            results = bin_array_5[singlebin(bin_array_5)]
            numpy.testing.assert_equal(results,expected_bin_array_5[i],
                                       err_msg=err_msg%(bin_array_5,i))
            results = bin_array_6[singlebin(bin_array_6)]
            numpy.testing.assert_equal(results,expected_bin_array_6[i],
                                       err_msg=err_msg%(bin_array_6,i))
    t1 = time.time()
    print "Time to test linear BinStep binning: ", 1000*(t1-t0), "ms"

def test_BinStep_log():
    t0 = time.time()
    lhs = stile.BinStep('field_0',low=0.25,high=8,step=numpy.log(2.),use_log=True)
    lhn = stile.BinStep('field_0',low=0.25,high=8,n_bins=5,use_log=True)
    lsn = stile.BinStep('field_0',low=0.25,step=numpy.log(2.),n_bins=5,use_log=True)
    hsn = stile.BinStep('field_0',high=8,step=numpy.log(2.),n_bins=5,use_log=True)
    reverse_lhs = stile.BinStep('field_0',low=8,high=0.25,step=-numpy.log(2.),use_log=True)        
    names = ["passed low, high, and step",
             "passed low, high, and n_bins",
             "passed low, step, and n_bins",
             "passed high, step, and n_bins",
             "passed low, high, and step with low and high reversed"]
    
    objs = [lhs,lhn,lsn,hsn,reverse_lhs]

    expected_bin_array_1 = [bin_array_1[:0],bin_array_1[0],bin_array_1[1],
                            bin_array_1[2:4],bin_array_1[4]]
    expected_bin_array_2 = [bin_array_2[:0],bin_array_2[:0],bin_array_2[0],
                            bin_array_2[1:3],bin_array_2[3:]]
    expected_bin_array_3 = [bin_array_3[:0],bin_array_3[:2],bin_array_3[:0],
                            bin_array_3[4],bin_array_3[2:4]]
    expected_bin_array_4 = [bin_array_4[:0],bin_array_4[0],bin_array_4[:0],
                            bin_array_4[:0],bin_array_4[:0]]
    expected_bin_array_5 = [bin_array_5[:0],bin_array_5[:0],bin_array_5[0],
                            bin_array_5[1],bin_array_5[2]]
    expected_bin_array_6 = [bin_array_6[:0],bin_array_6[:0],bin_array_6[1],
                            bin_array_6[:0],bin_array_6[:0]]

    for obj, name in zip(objs,names):
        err_msg = ("Logarithmic BinStep test ("+name+
                   ") failed to produce correct binning for array %s in bin # %i")
        obj_list = obj()
        if len(obj_list)!=5:
            raise RuntimeError('Wrong number of bins created from logarithmic BinStep with '+
                                name+': '+str(len(obj_list)))
        if obj==reverse_lhs:
            obj_list.reverse()
        for i,singlebin in enumerate(obj_list):
            results = bin_array_1[singlebin(bin_array_1)]
            numpy.testing.assert_equal(results,expected_bin_array_1[i],
                                       err_msg=err_msg%(bin_array_1,i))
            results = bin_array_2[singlebin(bin_array_2)]
            numpy.testing.assert_equal(results,expected_bin_array_2[i],
                                       err_msg=err_msg%(bin_array_2,i))
            results = bin_array_3[singlebin(bin_array_3)]
            numpy.testing.assert_equal(results,expected_bin_array_3[i],
                                       err_msg=err_msg%(bin_array_3,i))
            results = bin_array_4[singlebin(bin_array_4)]
            numpy.testing.assert_equal(results,expected_bin_array_4[i],
                                       err_msg=err_msg%(bin_array_4,i))
            results = bin_array_5[singlebin(bin_array_5)]
            numpy.testing.assert_equal(results,expected_bin_array_5[i],
                                       err_msg=err_msg%(bin_array_5,i))
            results = bin_array_6[singlebin(bin_array_6)]
            numpy.testing.assert_equal(results,expected_bin_array_6[i],
                                       err_msg=err_msg%(bin_array_6,i))
    t1 = time.time()
    print "Time to test log BinStep binning: ", 1000*(t1-t0), "ms"

def test_BinList():
    t0 = time.time()
    obj_forward = stile.BinList('field_0',[0,1.,1.9,3.0,4.0,5.0,6.5])
    obj_reverse = stile.BinList('field_0',[6.5,5.0,4.0,3.0,1.9,1.,0])
    
    names = [" ", " (reversed) "]
    objs = [obj_forward,obj_reverse]

    # Expected results; each item of the list is the result of the n-th SingleBin
    expected_bin_array_1 = [bin_array_1[0],bin_array_1[1],bin_array_1[2],bin_array_1[3],
                            bin_array_1[4],bin_array_1[:0]]
    expected_bin_array_2 = [bin_array_2[:0],bin_array_2[0],bin_array_2[1],bin_array_2[2],
                            bin_array_2[3],bin_array_2[4]]
    expected_bin_array_3 = [bin_array_3[0:2],bin_array_3[:0],bin_array_3[:0],bin_array_3[4],
                            bin_array_3[3],bin_array_3[2]]
    expected_bin_array_4 = [bin_array_4[0],bin_array_4[:0],bin_array_4[:0],
                            bin_array_4[:0],bin_array_4[:0],bin_array_4[:0]]
    expected_bin_array_5 = [bin_array_5[:0],bin_array_5[0],bin_array_5[:0],
                            bin_array_5[1],bin_array_5[:0],bin_array_5[2]]
    expected_bin_array_6 = [bin_array_6[:0],bin_array_6[1],bin_array_6[:0],
                            bin_array_6[:0],bin_array_6[:0],bin_array_6[:0]]

    for obj, name in zip(objs,names):
        err_msg = ("BinList"+name+"failed to produce correct binning for array %s in bin # %i")
        obj_list = obj()
        if len(obj_list)!=6:
            raise RuntimeError('Wrong number of bins created from BinList'+name+': '+
                                 str(len(obj_list)))
        if obj==obj_reverse:
            obj_list.reverse()
        for i,singlebin in enumerate(obj_list):
            results = bin_array_1[singlebin(bin_array_1)]
            numpy.testing.assert_equal(results,expected_bin_array_1[i],
                                       err_msg=err_msg%(bin_array_1,i))
            results = bin_array_2[singlebin(bin_array_2)]
            numpy.testing.assert_equal(results,expected_bin_array_2[i],
                                       err_msg=err_msg%(bin_array_2,i))
            results = bin_array_3[singlebin(bin_array_3)]
            numpy.testing.assert_equal(results,expected_bin_array_3[i],
                                       err_msg=err_msg%(bin_array_3,i))
            results = bin_array_4[singlebin(bin_array_4)]
            numpy.testing.assert_equal(results,expected_bin_array_4[i],
                                       err_msg=err_msg%(bin_array_4,i))
            results = bin_array_5[singlebin(bin_array_5)]
            numpy.testing.assert_equal(results,expected_bin_array_5[i],
                                       err_msg=err_msg%(bin_array_5,i))
            results = bin_array_6[singlebin(bin_array_6)]
            numpy.testing.assert_equal(results,expected_bin_array_6[i],
                                       err_msg=err_msg%(bin_array_6,i))
    t1 = time.time()
    print "Time to test BinList binning: ", 1000*(t1-t0), "ms"

def test_bin_creation_errors():
    try:
        t0 = time.time()
        # Invalid bounds in logarithmic BinStep
        numpy.testing.assert_raises(ValueError,stile.BinStep,'c',
                                    low=0,high=10,step=1,use_log=True)
        numpy.testing.assert_raises(ValueError,stile.BinStep,'c',
                                    low=10,high=-1,step=-1,use_log=True)
        # Various not-enough-arguments errors to BinStep (probably overkill)
        numpy.testing.assert_raises(TypeError,stile.BinStep)
        numpy.testing.assert_raises(TypeError,stile.BinStep,'c')
        numpy.testing.assert_raises(TypeError,stile.BinStep,'c',low=1)
        numpy.testing.assert_raises(TypeError,stile.BinStep,'c',low=1,high=2)
        numpy.testing.assert_raises(TypeError,stile.BinStep,'c',low=1,step=2)
        numpy.testing.assert_raises(TypeError,stile.BinStep,'c',low=1,n_bins=2)
        numpy.testing.assert_raises(TypeError,stile.BinStep,'c',step=1)
        numpy.testing.assert_raises(TypeError,stile.BinStep,'c',step=1,n_bins=2)
        numpy.testing.assert_raises(TypeError,stile.BinStep,'c',step=1,high=2)
        numpy.testing.assert_raises(TypeError,stile.BinStep,'c',n_bins=1)
        numpy.testing.assert_raises(TypeError,stile.BinStep,'c',n_bins=1,high=2)
        numpy.testing.assert_raises(TypeError,stile.BinStep,'c',high=2)
        # Inconsistent and nonsense arguments to BinStep
        numpy.testing.assert_raises(ValueError,stile.BinStep,'c',low=1,high=0,step=0.5)
        numpy.testing.assert_raises(ValueError,stile.BinStep,'c',low=0,high=1,step=-0.5)
        numpy.testing.assert_raises(ValueError,stile.BinStep,'c',low=0,high=5,step=1,n_bins=7)
        stile.BinStep('c',low=0,high=-1,step=-0.5) # actually consistent
        numpy.testing.assert_raises(ValueError,stile.BinStep,'c',low=1,high=1,step=0.5)
        numpy.testing.assert_raises(ValueError,stile.BinStep,'c',low=1,high=2,n_bins=-1)
        numpy.testing.assert_raises(TypeError,stile.BinStep,0,low=0,high=5,step=1)
        # Wrong arguments to BinList
        numpy.testing.assert_raises(TypeError,stile.BinList,'c',[1,2,3],n_bins=1)
        numpy.testing.assert_raises(ValueError,stile.BinList,'c',[1,3,2])
        numpy.testing.assert_raises(TypeError,stile.BinList,0,[1,3])
        numpy.testing.assert_raises(TypeError,stile.BinList,'c',[])
        numpy.testing.assert_raises(TypeError,stile.BinList,[1,3,2])
        numpy.testing.assert_raises(TypeError,stile.BinList,'c')
        t1 = time.time()
        print "Time to test bin object construction errors: ", 1000*(t1-t0), "ms"
    except ImportError:
        print 'The assert_raises tests require nose'

def test_singlebin_input_errors():
    t0 = time.time()
    sb = stile.binning.SingleBin('field_0',low=0,high=10,short_name='boo')
    sfb = stile.binning.SingleFunctionBin(binfunction,1)
    sb.long_name # check that this was made properly
    try:
        numpy.testing.assert_raises(TypeError,sb,[1,2,3,4])
        numpy.testing.assert_raises(ValueError,sb,numpy.array([1,2,3,4]))
        numpy.testing.assert_raises(ValueError,sb,
                                    numpy.array([(1,),(2,),(3,),(4,)],dtype=[('field_1',int)]))
        numpy.testing.assert_raises(TypeError,sb,3)
    except ImportError:
        print 'The assert_raises tests require nose'
    t1 = time.time()
    print "Time to test SingleBin data input errors: ", 1000*(t1-t0), "ms"

if __name__=='__main__':
    test_singlebin_input_errors() #Do this first, so SingleBin errors don't propagate to later tests
    test_BinStep_SingleBin_creation()
    test_BinList_SingleBin_creation()
    test_BinStep_linear()
    test_BinStep_log()
    test_BinList()
    test_bin_creation_errors()

