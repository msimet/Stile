import numpy
import helper
import unittest
try:
    import stile
except:
    import sys
    sys.path.append('..')
    import stile

class TestTreeCorrUtils(unittest.TestCase):
    def setUp(self):
        # Some sample TreeCorr dicts to play with
        # Nonsense key/value pair
        self.dict1 = {'flarp': 'fwep'} 
        # OK if check_status=False, fails if check_status = True
        self.dict2 = {'file_name': 'f1.dat', 'file_name2': 'f2.dat', 'rand_file_name': 'r1.dat', 
                      'rand_file_name2': 'r2.dat','file_list': 'fl1.dat', 'file_list2': 'fl2.dat', 
                      'rand_file_list': 'rl1.dat','rand_file_list2': 'rl2.dat', 
                      'file_type': 'ASCII', 'delimiter': ' ', 'comment_marker': '#', 
                      'first_row': 0, 'last_row': 10, 'x_col': 3, 'y_col': 4,'ra_col': 5, 
                      'dec_col': 6, 'x_units': 'arcsec', 'y_units': 'arcsec', 
                      'ra_units': 'degrees', 'dec_units': 'degrees', 'g1_col': 7, 'g2_col': 8, 
                      'k_col': 9,'w_col': 10, 'flip_g1': False, 'flip_g2': False, 
                      'pairwise': False, 'min_sep': 0.1, 'max_sep': 10, 'nbins': 10, 
		      'bin_size': 0.99, 'sep_units': 'degrees', 'bin_slop': 0.8,  
                      'n2_file_name': 'o1.dat', 'n2_statistic': 'compensated', 
                      'ng_file_name': 'o2.dat', 'ng_statistic': 'compensated', 
                      'g2_file_name': 'o3.dat', 'nk_file_name': 'o4.dat', 'nk_statistic': 'simple', 
                      'k2_file_name': 'o5.dat', 'kg_file_name': 'o6.dat', 'precision': 3, 
                      'm2_file_name': 'o7.dat', 'm2_uform': 'Crittenden', 'nm_file_name': 'o8.dat', 
                      'norm_file_name': 'o9.dat', 'verbose': 3, 'num_threads': 16, 
                      'split_method': 'median'} 
        # The output of a run of TreeCorr.
        self.treecorr_output = numpy.array(
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

    def test_ReadTreeCorrResultsFile(self):
        """Test the routine that reads in TreeCorr results files."""
        arr = stile.ReadTreeCorrResultsFile('test_data/TreeCorr_output.dat')
        numpy.testing.assert_equal(arr,self.treecorr_output)
	# Seems to depend on which version of NumPy which error is raised
        self.assertRaises((TypeError,StopIteration),stile.ReadTreeCorrResultsFile,
                                    'test_data/empty_file.dat')

    def test_PickTreeCorrKeys(self):
        """Test the routine that adds to a dict to be given to TreeCorr."""
        new_dict = stile.treecorr_utils.PickTreeCorrKeys(self.dict1) # nonsense dict
        self.assertEqual(len(new_dict),0,
                          msg='The new dict should have no entries')
        new_dict = stile.treecorr_utils.PickTreeCorrKeys(self.dict2)
        self.assertEqual(new_dict,self.dict2,
                         msg='All entries from the dict should have been copied to the '+
                             'new dict')

    
if __name__=='__main__':
    unittest.main()

