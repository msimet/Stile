import numpy
import time
import test_helper
try:
    import stile
except:
    import sys
    sys.path.append('..')
    import stile
    
# Some sample corr2 dicts to play with
# Nonsense key/value pair
dict1 = {'flarp': 'fwep'} 
# OK if check_status=False, fails if check_status = True
dict2 = {'file_name': 'f1.dat', 'do_auto_corr': True, 'do_cross_corr': False, 
         'file_name2': 'f2.dat', 'rand_file_name': 'r1.dat', 'rand_file_name2': 'r2.dat',
         'file_list': 'fl1.dat', 'file_list2': 'fl2.dat', 'rand_file_list': 'rl1.dat',
         'rand_file_list2': 'rl2.dat', 'file_type': 'ASCII', 'delimiter': ' ', 
         'comment_marker': '#', 'first_row': 0, 'last_row': 10, 'x_col': 3, 'y_col': 4,
         'ra_col': 5, 'dec_col': 6, 'x_units': 'arcsec', 'y_units': 'arcsec', 
         'ra_units': 'degrees', 'dec_units': 'degrees', 'g1_col': 7, 'g2_col': 8, 'k_col': 9,
         'w_col': 10, 'flip_g1': False, 'flip_g2': False, 'pairwise': False, 'project': False,
         'project_ra': 0, 'project_dec': 0, 'min_sep': 0.1, 'max_sep': 10, 'nbins': 10, 
         'bin_size': 0.99, 'sep_units': 'degrees', 'bin_slop': 0.8, 'smooth_scale': 1, 
         'n2_file_name': 'o1.dat', 'n2_statistic': 'compensated', 'ng_file_name': 'o2.dat',
         'ng_statistic': 'compensated', 'g2_file_name': 'o3.dat', 'nk_file_name': 'o4.dat',
         'nk_statistic': 'simple', 'k2_file_name': 'o5.dat', 'kg_file_name': 'o6.dat',
         'precision': 3, 'm2_file_name': 'o7.dat', 'm2_uform': 'Crittenden', 
         'nm_file_name': 'o8.dat', 'norm_file_name': 'o9.dat', 'verbose': 3, 'num_threads': 16,
         'split_method': 'median'} 
# OK if check_status=False, fails if check_status = True; this one also checks the type-checking
# by passing some things as strings
dict3 = {'file_name': 'f1.dat', 'do_auto_corr': 'True', 'do_cross_corr': 'False', 
         'file_name2': 'f2.dat', 'rand_file_name': 'r1.dat', 'rand_file_name2': 'r2.dat',
         'file_list': 'fl1.dat', 'file_list2': 'fl2.dat', 'rand_file_list': 'rl1.dat',
         'rand_file_list2': 'rl2.dat', 'file_type': 'ASCII', 'delimiter': ' ', 
         'comment_marker': '#', 'first_row': '0', 'last_row': '10', 'x_col': '3', 'y_col': '4',
         'ra_col': '5', 'dec_col': '6', 'x_units': 'arcsec', 'y_units': 'arcsec', 
         'ra_units': 'degrees', 'dec_units': 'degrees', 'g1_col': '7', 'g2_col': '8', 
         'k_col': '9', 'w_col': '10', 'flip_g1': 'False', 'flip_g2': 'False', 
         'pairwise': 'False', 'project': 'False', 'project_ra': '0', 'project_dec': '0', 
         'min_sep': '0.1', 'max_sep': '10', 'nbins': '10', 'bin_size': '0.99', 
         'sep_units': 'degrees', 'bin_slop': '0.8', 'smooth_scale': '1', 
         'n2_file_name': 'o1.dat', 'n2_statistic': 'compensated', 'ng_file_name': 'o2.dat',
         'ng_statistic': 'compensated', 'g2_file_name': 'o3.dat', 'nk_file_name': 'o4.dat',
         'nk_statistic': 'simple', 'k2_file_name': 'o5.dat', 'kg_file_name': 'o6.dat',
         'precision': 3, 'm2_file_name': 'o7.dat', 'm2_uform': 'Crittenden', 
         'nm_file_name': 'o8.dat', 'norm_file_name': 'o9.dat', 'verbose': '3', 
         'num_threads': '16', 'split_method': 'median'} 
# Should fail some of the type checks (do_auto_corr, precision)
dict4 = {'file_name': 'f1.dat', 'do_auto_corr': '3', 'do_cross_corr': 'False', 
         'file_name2': 'f2.dat', 'rand_file_name': 'r1.dat', 'rand_file_name2': 'r2.dat',
         'file_list': 'fl1.dat', 'file_list2': 'fl2.dat', 'rand_file_list': 'rl1.dat',
         'rand_file_list2': 'rl2.dat', 'file_type': 'ASCII', 'delimiter': ' ', 
         'comment_marker': '#', 'first_row': '0', 'last_row': '10', 'x_col': '3', 'y_col': '4',
         'ra_col': '5', 'dec_col': '6', 'x_units': 'arcsec', 'y_units': 'arcsec', 
         'ra_units': 'degrees', 'dec_units': 'degrees', 'g1_col': '7', 'g2_col': '8', 
         'k_col': '9', 'w_col': '10', 'flip_g1': 'False', 'flip_g2': 'False', 
         'pairwise': 'False', 'project': 'False', 'project_ra': '0', 'project_dec': '0', 
         'min_sep': '0.1', 'max_sep': '10', 'nbins': '10', 'bin_size': '0.99', 
         'sep_units': 'degrees', 'bin_slop': '0.8', 'smooth_scale': '1', 
         'n2_file_name': 'o1.dat', 'n2_statistic': 'compensated', 'ng_file_name': 'o2.dat',
         'ng_statistic': 'compensated', 'g2_file_name': 'o3.dat', 'nk_file_name': 'o4.dat',
         'nk_statistic': 'simple', 'k2_file_name': 'o5.dat', 'kg_file_name': 'o6.dat',
         'precision': 'yellow', 'm2_file_name': 'o7.dat', 'm2_uform': 'Crittenden', 
         'nm_file_name': 'o8.dat', 'norm_file_name': 'o9.dat', 'verbose': '3', 
         'num_threads': '16', 'split_method': 'median'} 
# Should fail some of the value checks (file_type and m2_uform)
dict5 = {'file_name': 'f1.dat', 'do_auto_corr': True, 'do_cross_corr': False, 
         'file_name2': 'f2.dat', 'rand_file_name': 'r1.dat', 'rand_file_name2': 'r2.dat',
         'file_list': 'fl1.dat', 'file_list2': 'fl2.dat', 'rand_file_list': 'rl1.dat',
         'rand_file_list2': 'rl2.dat', 'file_type': 'Unicode', 'delimiter': ' ', 
         'comment_marker': '#', 'first_row': 0, 'last_row': 10, 'x_col': 3, 'y_col': 4,
         'ra_col': 5, 'dec_col': 6, 'x_units': 'arcsec', 'y_units': 'arcsec', 
         'ra_units': 'degrees', 'dec_units': 'degrees', 'g1_col': 7, 'g2_col': 8, 'k_col': 9,
         'w_col': 10, 'flip_g1': False, 'flip_g2': False, 'pairwise': False, 'project': False,
         'project_ra': 0, 'project_dec': 0, 'min_sep': 0.1, 'max_sep': 10, 'nbins': 10, 
         'bin_size': 0.99, 'sep_units': 'degrees', 'bin_slop': 0.8, 'smooth_scale': 1, 
         'n2_file_name': 'o1.dat', 'n2_statistic': 'compensated', 'ng_file_name': 'o2.dat',
         'ng_statistic': 'compensated', 'g2_file_name': 'o3.dat', 'nk_file_name': 'o4.dat',
         'nk_statistic': 'simple', 'k2_file_name': 'o5.dat', 'kg_file_name': 'o6.dat',
         'precision': 3, 'm2_file_name': 'o7.dat', 'm2_uform': 'Swindon', 
         'nm_file_name': 'o8.dat', 'norm_file_name': 'o9.dat', 'verbose': 3, 'num_threads': 16,
         'split_method': 'median'} 
# Should pass even with check_status = True
dict6 = {'flip_g1': False, 'flip_g2': False, 'project': False, 'project_ra': 0, 
         'project_dec': 0, 'min_sep': 0.1, 'max_sep': 10, 'nbins': 10, 
         'bin_size': 0.99, 'sep_units': 'degrees', 'bin_slop': 0.8, 
         'precision': 3, 'm2_uform': 'Crittenden', 'split_method': 'median'} 
# The output of a run of corr2.
corr2_output = numpy.array(
    [(5.389e-02, 5.443e-02, 2.206e-02, -4.259e-02, 2.578e-02, 1.820e+02, 1.820e+02),
     (6.260e-02, 6.205e-02, 3.738e-03, 2.995e-02, 2.079e-02, 2.800e+02, 2.800e+02),
     (7.271e-02, 7.241e-02, 1.857e-02, -4.192e-02, 1.755e-02, 3.930e+02, 3.930e+02), 
     (8.446e-02, 8.489e-02, -9.208e-03, -6.122e-03, 1.556e-02, 5.000e+02, 5.000e+02), 
     (9.811e-02, 9.817e-02, 1.986e-02, 9.554e-03, 1.447e-02, 5.780e+02, 5.780e+02), 
     (1.140e-01, 1.140e-01, 7.698e-03, 4.370e-03, 1.068e-02, 1.061e+03, 1.061e+03), 
     (1.324e-01, 1.342e-01, 9.733e-03, -4.264e-03, 1.037e-02, 1.125e+03, 1.125e+03), 
     (1.538e-01, 1.520e-01, 2.400e-03, -7.149e-03, 9.048e-03, 1.478e+03, 1.478e+03), 
     (1.786e-01, 1.772e-01, 2.964e-03, -1.999e-03, 7.230e-03, 2.315e+03, 2.315e+03), 
     (2.075e-01, 2.076e-01, 1.652e-03, -2.616e-03, 6.580e-03, 2.795e+03, 2.795e+03), 
     (2.410e-01, 2.411e-01, -1.277e-02, 2.982e-04, 5.274e-03, 4.350e+03, 4.350e+03), 
     (2.799e-01, 2.855e-01, 1.326e-03, -1.140e-03, 4.651e-03, 5.593e+03, 5.593e+03), 
     (3.252e-01, 3.270e-01, -2.517e-03, 3.061e-04, 4.554e-03, 5.834e+03, 5.834e+03), 
     (3.777e-01, 3.752e-01, -3.427e-03, -5.551e-03, 3.459e-03, 1.011e+04, 1.011e+04), 
     (4.387e-01, 4.393e-01, 2.000e-04, -2.414e-03, 3.095e-03, 1.263e+04, 1.263e+04), 
     (5.096e-01, 4.973e-01, -3.784e-03, 5.500e-04, 3.025e-03, 1.322e+04, 1.322e+04), 
     (5.920e-01, 5.873e-01, 2.131e-03, 9.000e-05, 3.673e-03, 8.971e+03, 8.971e+03), 
     (6.877e-01, 6.877e-01, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00), 
     (7.988e-01, 7.988e-01, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00),
     (9.278e-01, 9.278e-01, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00)],
     dtype=[("R_nominal",float),("<R>",float),("<gamT>",float),("<gamX>",float),
            ("sig",float),("weight",float),("npairs",float)])

def compare_text_files(f1,f2, reorder=True):
    """
    Compare the contents of two text files. Any whitespace on a line is converted to a single space,
    and beginning/ending whitespace is stripped along with blank lines.  The files are then sorted
    (if reorder=True, the default), and the resulting arrays compared line-by-line.
    
    @param f1      The name of a file which exists in the filesystem
    @param f2      The name of a file which exists in the filesystem
    @param reorder Whether to sort the lines of the files.  This is appropriate if the order of 
                   information is unimportant, eg a config file with no repeated keys.
    @returns       True if the files are the same (modulo whitespace/blank lines/order as described 
                   above), else False
    """
    with open(f1) as f:
        f1_data = f.readlines()
    with open(f2) as f:
        f2_data = f.readlines()
    f1_data = [ff1.split() for ff1 in f1_data]
    f2_data = [ff2.split() for ff2 in f2_data]
    f1_data = [ff1 for ff1 in f1_data if ff1]
    f2_data = [ff2 for ff2 in f2_data if ff2]
    f1_data = [' '.join(ff1) for ff1 in f1_data]
    f2_data = [' '.join(ff2) for ff2 in f2_data]
    if reorder:
        f1_data.sort()
        f2_data.sort()
    return f1_data==f2_data

def test_CheckArguments():
    t0 = time.time()
             
    stile.corr2_utils.CheckArguments(dict2,check_status=False)
    stile.corr2_utils.CheckArguments(dict3,check_status=False)
    stile.corr2_utils.CheckArguments(dict6,check_status=True)
    try:
        numpy.testing.assert_raises(ValueError,stile.corr2_utils.CheckArguments,dict1,
                                    check_status=True)
        numpy.testing.assert_raises(ValueError,stile.corr2_utils.CheckArguments,dict2,
                                    check_status=True)
        numpy.testing.assert_raises(ValueError,stile.corr2_utils.CheckArguments,dict3,
                                    check_status=True)
        numpy.testing.assert_raises(ValueError,stile.corr2_utils.CheckArguments,dict4,
                                    check_status=True)
        numpy.testing.assert_raises(ValueError,stile.corr2_utils.CheckArguments,dict4,
                                    check_status=False)
        numpy.testing.assert_raises(ValueError,stile.corr2_utils.CheckArguments,dict5,
                                    check_status=True)
        numpy.testing.assert_raises(ValueError,stile.corr2_utils.CheckArguments,dict5,
                                    check_status=False)
    except ImportError:
        pass
    t1 = time.time()
    print "Time to test corr2_utils.CheckArguments: ", 1000*(t1-t0), "ms"
    
def test_WriteCorr2ConfigurationFile():
    import tempfile
    import os
    t0 = time.time()
    handle, f2 = tempfile.mkstemp(dir='.')
    stile.WriteCorr2ConfigurationFile(f2,dict2)
    if not compare_text_files(f2,'test_data/corr2_dict2_config_file.dat'):
        os.close(handle)
        raise AssertionError('WriteCorr2ConfigurationFile() produced incorrect output for test '+
                             'args dict dict2; check file %s for problems'%f2)
    else:
        os.close(handle)
        os.remove(f2)
    t1 = time.time()
    print "Time to test WriteCorr2ConfigurationFile: ", 1000*(t1-t0), "ms"
    
def test_ReadCorr2ResultsFile():
    t0 = time.time()
    arr = stile.ReadCorr2ResultsFile('test_data/corr2_output.dat')
    numpy.testing.assert_equal(arr,corr2_output)
    try:
        numpy.testing.assert_raises(StopIteration,stile.ReadCorr2ResultsFile,
                                'test_data/empty_file.dat')
    except ImportError:
        pass
    t1 = time.time()
    print "Time to test ReadCorr2ResultsFile: ", 1000*(t1-t0), "ms"

def test_AddCorr2Dict():
    t0 = time.time()
    new_dict = stile.corr2_utils.AddCorr2Dict(dict1) # nonsense dict
    if new_dict['corr2_kwargs']:
        raise TypeError('The "corr2_kwargs" key of the new dict should have no entries')
    new_dict = stile.corr2_utils.AddCorr2Dict(dict2)
    if not new_dict['corr2_kwargs']==dict2:
        raise TypeError('All entries from the dict should have been copied to the "corr2_options" '+
                        'key of the new dict')
    t1 = time.time()
    print "Time to test AddCorr2Dict: ", 1000*(t1-t0), "ms"

def test_MakeCorr2Cols():
    t0 = time.time()
    if stile.corr2_utils.MakeCorr2Cols(dict1):
        raise TypeError('Returned dict should have no entries')
    if stile.corr2_utils.MakeCorr2Cols(dict2):
        raise TypeError('Returned dict should have no entries')
    
    list1 = ['x','y','k','w']
    list2 = ['ra','dec','id']
    listdict1 = {'y': 1, 'x': 0, 'k': 2, 'w': 3}
    listdict2 = {'ra': 0, 'dec': 1, 'id': 2}
    listdict3 = {'ra': 'RA_J2000', 'dec': 'DEC_J2000'}
    
    list1_results = {'x_col': 1, 'y_col': 2, 'k_col': 3, 'w_col': 4}
    list2_results = {'ra_col': 1, 'dec_col': 2}
    list2_withk_results = {'ra_col': 1, 'dec_col': 2, 'k_col': 3} 
    list3_results = {'ra_col': 'RA_J2000', 'dec_col': 'DEC_J2000'}
    
    numpy.testing.assert_equal(stile.corr2_utils.MakeCorr2Cols(list1),list1_results)
    numpy.testing.assert_equal(stile.corr2_utils.MakeCorr2Cols(list2),list2_results)
    numpy.testing.assert_equal(stile.corr2_utils.MakeCorr2Cols(list2,use_as_k='id'),
                               list2_withk_results)
    numpy.testing.assert_equal(stile.corr2_utils.MakeCorr2Cols(listdict1),list1_results)
    numpy.testing.assert_equal(stile.corr2_utils.MakeCorr2Cols(listdict2),list2_results)
    numpy.testing.assert_equal(stile.corr2_utils.MakeCorr2Cols(listdict2,use_as_k='id'),
                               list2_withk_results)
    numpy.testing.assert_equal(stile.corr2_utils.MakeCorr2Cols(listdict3),list3_results)
    
    t1 = time.time()
    print "Time to test MakeCorr2Cols: ", 1000*(t1-t0), "ms"

def test_OSFile():
    t0 = time.time()
    arr0 = [1,2,3,4,5]
    arr1 = numpy.array([[1,2,3],[4.,5.,6.]])
    arr2 = numpy.array([(1.5,2,3.5),(4,5,6)],dtype='d,l,d')
    arr2.dtype.names = ['one','two','three']
    OSFile0 = stile.corr2_utils.OSFile(arr0)
    OSFile1 = stile.corr2_utils.OSFile(arr1)
    OSFile2 = stile.corr2_utils.OSFile(arr2)
    OSFile3 = stile.corr2_utils.OSFile([OSFile0,OSFile1,OSFile2])
    OSFile4 = stile.corr2_utils.OSFile(arr2,fields=['two','three','one'])
    OSFile5 = stile.corr2_utils.OSFile(arr2,fields={'two':0,'three':1,'one':2})
    OSFile6 = stile.corr2_utils.OSFile(OSFile2)
    OSFile7 = stile.corr2_utils.OSFile(OSFile2,fields=['two','three','one'])
    numpy.testing.assert_equal(stile.ReadTable(OSFile0.file_name),
                               numpy.array([tuple(arr0)],dtype='l,l,l,l,l'))
    numpy.testing.assert_equal(stile.ReadTable(OSFile1.file_name),
                               numpy.array([tuple(a) for a in arr1],dtype='l,l,l'))
    result = numpy.array(stile.ReadTable(OSFile2.file_name))
    numpy.testing.assert_equal(*test_helper.FormatSame(result,arr2))
    str_len = max([len(OSFile0.file_name),len(OSFile1.file_name),len(OSFile2.file_name)])
    str_dtype = ','.join(['S'+str(str_len)]*3)
    numpy.testing.assert_equal(stile.ReadTable(OSFile3.file_name),
                               numpy.array(
                                    [(OSFile0.file_name,OSFile1.file_name,OSFile2.file_name)],
                                    dtype=str_dtype))
    result = stile.ReadTable(OSFile4.file_name,fields=['two','three','one'])
    numpy.testing.assert_equal(*test_helper.FormatSame(result,arr2[['two','three','one']]))
    result = stile.ReadTable(OSFile5.file_name,fields=['two','three','one'])
    numpy.testing.assert_equal(*test_helper.FormatSame(result,arr2[['two','three','one']]))
    assert OSFile6==OSFile2 # Fun fact: numpy.testing.assert_equal of objects ignores __eq__
    numpy.testing.assert_equal(stile.ReadASCIITable(OSFile4.file_name),
                               stile.ReadASCIITable(OSFile7.file_name))
    del OSFile7
    del OSFile6
    del OSFile5
    del OSFile4
    del OSFile3
    del OSFile2
    del OSFile1
    del OSFile0
    t1 = time.time()
    print "Time to test OSFile: ", 1000*(t1-t0), "ms"
    
def test_MakeCorr2FileKwargs():
    t0 = time.time()
    data = [(1.0, 2.54, 0.25, -0.16),
            (3.1, 2.36, 0.0, 0.8)]
    data = numpy.array(data,dtype=[('ra', float),('dec',float),('g1',float),('g2',float)])
    #    data as file lists
    result = stile.MakeCorr2FileKwargs(data)
    assert len(result.keys())==5
    assert 'file_name' in result
    assert 'ra_col' in result
    assert 'dec_col' in result
    assert 'g1_col' in result
    assert 'g2_col' in result
    assert isinstance(result['file_name'],stile.corr2_utils.OSFile)
    numpy.testing.assert_equal(*test_helper.FormatSame(
                                    stile.ReadASCIITable('test_data/data_table.dat'),
                                    stile.ReadTable(result['file_name'].file_name)))
    numpy.testing.assert_equal([result['ra_col'],result['dec_col'],result['g1_col'],
                                result['g2_col']],[1,2,3,4]) # corr2 cols start from 1!

    result = stile.MakeCorr2FileKwargs(('test_data/data_table.dat',['ra','dec','g1','g2']))
    assert len(result.keys())==5
    assert 'file_name' in result
    assert 'ra_col' in result
    assert 'dec_col' in result
    assert 'g1_col' in result
    assert 'g2_col' in result
    assert result['file_name']=='test_data/data_table.dat'
    numpy.testing.assert_equal([result['ra_col'],result['dec_col'],result['g1_col'],
                                result['g2_col']],[1,2,3,4])
    
    result2 = stile.MakeCorr2FileKwargs(('test_data/data_table.dat',
                                        {'ra': 0, 'dec': 1, 'g1': 2, 'g2': 3}))
    assert result==result2

    result = stile.MakeCorr2FileKwargs(('test_data/data_table.dat',['ra','dec','g1','g2']),
                                       data2 = data)
    assert len(result.keys())==6
    assert 'file_name' in result
    assert 'file_name2' in result
    assert 'ra_col' in result
    assert 'dec_col' in result
    assert 'g1_col' in result
    assert 'g2_col' in result
    assert isinstance(result['file_name2'],stile.corr2_utils.OSFile)
    numpy.testing.assert_equal(*test_helper.FormatSame(
                                    stile.ReadASCIITable('test_data/data_table.dat'),
                                    stile.ReadTable(result['file_name2'].file_name)))
    numpy.testing.assert_equal([result['ra_col'],result['dec_col'],result['g1_col'],
                                result['g2_col']],[1,2,3,4]) # corr2 cols start from 1!
    
    result = stile.MakeCorr2FileKwargs(('test_data/data_table.dat',['dec','ra','g1','g2']),
                                       data2 = data)
    assert len(result.keys())==6
    assert 'file_name' in result
    assert 'file_name2' in result
    assert 'ra_col' in result
    assert 'dec_col' in result
    assert 'g1_col' in result
    assert 'g2_col' in result
    assert isinstance(result['file_name2'],stile.corr2_utils.OSFile)
    result2 = numpy.array(stile.ReadTable(result['file_name2'].file_name))
    names = result2.dtype.names
    names = [names[1], names[0], names[2], names[3]]
    result2 = result2[names]
    numpy.testing.assert_equal(*test_helper.FormatSame(result2,
                               stile.ReadASCIITable('test_data/data_table.dat')))
    numpy.testing.assert_equal([result['ra_col'],result['dec_col'],result['g1_col'],
                                result['g2_col']],[2,1,3,4]) # corr2 cols start from 1!
    
    result = stile.MakeCorr2FileKwargs(data,data[['dec','ra','g1','g2']])
    assert len(result.keys())==6
    assert 'file_name' in result
    assert 'file_name2' in result
    assert 'ra_col' in result
    assert 'dec_col' in result
    assert 'g1_col' in result
    assert 'g2_col' in result
    assert isinstance(result['file_name'],stile.corr2_utils.OSFile)
    assert isinstance(result['file_name2'],stile.corr2_utils.OSFile)
    numpy.testing.assert_equal(stile.ReadTable(result['file_name'].file_name),
                               stile.ReadTable(result['file_name2'].file_name))
    
    result = stile.MakeCorr2FileKwargs(('test_data/data_table.dat',['ra','dec','g1','g2']),
                                       ('test_data/data_table.dat',['dec','ra','g1','g2']))
    assert len(result.keys())==6
    assert 'file_name' in result
    assert 'file_name2' in result
    assert 'ra_col' in result
    assert 'dec_col' in result
    assert 'g1_col' in result
    assert 'g2_col' in result
    assert (isinstance(result['file_name'],stile.corr2_utils.OSFile) or 
            isinstance(result['file_name2'],stile.corr2_utils.OSFile))
    assert not (isinstance(result['file_name'],stile.corr2_utils.OSFile) and
                isinstance(result['file_name2'],stile.corr2_utils.OSFile))
    result2 = stile.ReadTable(str(result['file_name']))
    result3 = stile.ReadTable(str(result['file_name2']))
    numpy.testing.assert_equal(result2[result2.dtype.names[result['ra_col']-1]],
                               result3[result3.dtype.names[result['dec_col']-1]])
    numpy.testing.assert_equal(result2[result2.dtype.names[result['dec_col']-1]],
                               result3[result3.dtype.names[result['ra_col']-1]])
    numpy.testing.assert_equal(result2[result2.dtype.names[result['g1_col']-1]],
                               result3[result3.dtype.names[result['g1_col']-1]])
    numpy.testing.assert_equal(result2[result2.dtype.names[result['g2_col']-1]],
                               result3[result3.dtype.names[result['g2_col']-1]])
    
    result = stile.MakeCorr2FileKwargs([data,data])
    assert len(result.keys())==5
    assert 'file_list' in result
    assert 'ra_col' in result
    assert 'dec_col' in result
    assert 'g1_col' in result
    assert 'g2_col' in result
    assert isinstance(result['file_list'],stile.corr2_utils.OSFile)
    result_file_names = stile.ReadTable(str(result['file_list']))
    assert len(result_file_names.dtype.names)==2
    numpy.testing.assert_equal(stile.ReadTable(result_file_names['f0'][0]),
                               stile.ReadTable(result_file_names['f1'][0]))
    numpy.testing.assert_equal(*test_helper.FormatSame(stile.ReadTable(result_file_names['f0'][0]),
                               stile.ReadASCIITable('test_data/data_table.dat')))
    
    result = stile.MakeCorr2FileKwargs([('test_data/data_table.dat',['ra','dec','g1','g2']),
                                        ('test_data/data_table.dat',['ra','dec','g1','g2'])])
    assert len(result.keys())==5
    assert 'file_list' in result
    assert 'ra_col' in result
    assert 'dec_col' in result
    assert 'g1_col' in result
    assert 'g2_col' in result
    assert isinstance(result['file_list'],stile.corr2_utils.OSFile)
    result_file_names = stile.ReadTable(str(result['file_list']))
    assert len(result_file_names.dtype.names)==2
    numpy.testing.assert_equal(result_file_names['f0'][0],'test_data/data_table.dat')
    numpy.testing.assert_equal(result_file_names['f1'][0],'test_data/data_table.dat')
    
    result = stile.MakeCorr2FileKwargs([('test_data/data_table.dat',['ra','dec','g1','g2']),
                                        ('test_data/data_table.dat',['dec','ra','g1','g2'])])
    
    assert len(result.keys())==5
    assert 'file_list' in result
    assert 'ra_col' in result
    assert 'dec_col' in result
    assert 'g1_col' in result
    assert 'g2_col' in result
    assert isinstance(result['file_list'],stile.corr2_utils.OSFile)
    result_file_names = stile.ReadTable(str(result['file_list']))
    assert len(result_file_names.dtype.names)==2
    if (result_file_names['f0'][0]=='test_data/data_table.dat' or 
        result_file_names['f1'][0]=='test_data/data_table.dat'):
        if result_file_names['f0'][0]==result_file_names['f1'][0]:
            raise AssertionError('Filenames should be different, but are the same')
        result2 = stile.ReadTable(result_file_names['f0'][0])
        result3 = stile.ReadTable(result_file_names['f1'][0])[['f1','f0','f2','f3']]
        result3.dtype.names = ['f0','f1','f2','f3']
        numpy.testing.assert_equal(*test_helper.FormatSame(result2,result3))
    else:
        raise AssertionError('Both files rewritten (should have been one)')
    osfile = stile.corr2_utils.OSFile(data)
    result = stile.MakeCorr2FileKwargs(osfile)
    assert len(result.keys())==5
    assert 'file_name' in result
    assert 'ra_col' in result
    assert 'dec_col' in result
    assert 'g1_col' in result
    assert 'g2_col' in result
    assert osfile == result['file_name']
    t1 = time.time()
    print "Time to test MakeCorr2FileKwargs: ", 1000*(t1-t0), "ms"
    
if __name__=='__main__':
    try:
        import nose
    except:
        print "assert_raises tests require nose"
    test_CheckArguments()
    test_WriteCorr2ConfigurationFile()
    test_ReadCorr2ResultsFile()
    test_AddCorr2Dict()
    test_MakeCorr2Cols()
    test_OSFile()
    test_MakeCorr2FileKwargs()

