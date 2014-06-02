import numpy
import time
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
    f1_data = [''.join(ff1) for ff1 in f1_data]
    f2_data = [''.join(ff2) for ff2 in f2_data]
    if reorder:
        f1_data.sort()
        f2_data.sort()
    return f1_data==f2_data

def test_CheckOptions():
    t0 = time.time()
             
    numpy.testing.assert_raises(ValueError,stile.corr2_utils.CheckOptions,dict1,check_status=True)
    numpy.testing.assert_raises(ValueError,stile.corr2_utils.CheckOptions,dict2,check_status=True)
    stile.corr2_utils.CheckOptions(dict2,check_status=False)
    numpy.testing.assert_raises(ValueError,stile.corr2_utils.CheckOptions,dict3,check_status=True)
    stile.corr2_utils.CheckOptions(dict3,check_status=False)
    numpy.testing.assert_raises(ValueError,stile.corr2_utils.CheckOptions,dict4,check_status=True)
    numpy.testing.assert_raises(ValueError,stile.corr2_utils.CheckOptions,dict4,check_status=False)
    numpy.testing.assert_raises(ValueError,stile.corr2_utils.CheckOptions,dict5,check_status=True)
    numpy.testing.assert_raises(ValueError,stile.corr2_utils.CheckOptions,dict5,check_status=False)
    stile.corr2_utils.CheckOptions(dict6,check_status=True)
    t1 = time.time()
    print "Time to test corr2_utils.check_options: ", 1000*(t1-t0), "ms"
    
def test_WriteCorr2ConfigurationFile():
    import tempfile
    import os
    t0 = time.time()
    handle, f2 = tempfile.mkstemp(dir='.')
    stile.WriteCorr2ConfigurationFile(f2,dict2)
    if not compare_text_files(f2,'test_data/corr2_dict2_config_file.dat'):
        os.close(handle)
        #os.remove(f2)
        raise AssertionError('WriteCorr2ConfigurationFile() produced incorrect output for test '+
                             'args dict dict2')
    else:
        os.close(handle)
        os.remove(f2)
    t1 = time.time()
    print "Time to test WriteCorr2ConfigurationFile: ", 1000*(t1-t0), "ms"
    
def test_ReadCorr2ResultsFile():
    t0 = time.time()
    arr = stile.ReadCorr2ResultsFile('test_data/corr2_output.dat')
    numpy.testing.assert_equal(arr,corr2_output)
    numpy.testing.assert_raises(RuntimeError,stile.ReadCorr2ResultsFile,
                                'test_data/empty_file.dat')
    t1 = time.time()
    print "Time to test read_corr2_results_file: ", 1000*(t1-t0), "ms"

def test_AddCorr2Dict():
    t0 = time.time()
    new_dict = stile.corr2_utils.AddCorr2Dict(dict1)
    if new_dict['corr2_options']:
        raise TypeError('The "corr2_options" key of the new dict should have no entries')
    new_dict = stile.corr2_utils.AddCorr2Dict(dict2)
    if not new_dict['corr2_options']==dict2:
        raise TypeError('All entries from the dict should have been copied to the "corr2_options" '+
                        'key of the new dict')
    t1 = time.time()
    print "Time to test add_corr2_dict: ", 1000*(t1-t0), "ms"

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
    
    list1_results = {'x_col': 1, 'y_col': 2, 'k_col': 3, 'w_col': 4}
    list2_results = {'ra_col': 1, 'dec_col': 2}
    
    numpy.testing.assert_equal(stile.corr2_utils.MakeCorr2Cols(list1),list1_results)
    numpy.testing.assert_equal(stile.corr2_utils.MakeCorr2Cols(list2),list2_results)
    numpy.testing.assert_equal(stile.corr2_utils.MakeCorr2Cols(listdict1),list1_results)
    numpy.testing.assert_equal(stile.corr2_utils.MakeCorr2Cols(listdict2),list2_results)
    
    t1 = time.time()
    print "Time to test make_corr2_cols: ", 1000*(t1-t0), "ms"
    
if __name__=='__main__':
    test_CheckOptions()
    test_WriteCorr2ConfigurationFile()
    test_ReadCorr2ResultsFile()
    test_AddCorr2Dict()
    test_MakeCorr2Cols()

