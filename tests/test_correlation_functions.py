import sys
import os
import numpy
import subprocess
import helper
import unittest

try:
    import stile
except ImportError:
    sys.path.append('..')
    import stile

class temp_data_handler():
    def __init__(self):
        self.temp_dir = None
    def getOutputPath(self,p):
        return p

class TestCorrelationFunctions(unittest.TestCase):
    def setUp(self):
        # The output of our known data set from the example directory
        self.expected_result = numpy.array(
                   [(5.389e-02, 5.443e-02, 2.206e-02,-4.259e-02, 2.578e-02, 1.820e+02, 1.820e+02),
                    (6.260e-02, 6.205e-02, 3.738e-03, 2.995e-02, 2.079e-02, 2.800e+02, 2.800e+02),
                    (7.271e-02, 7.241e-02, 1.857e-02,-4.192e-02, 1.755e-02, 3.930e+02, 3.930e+02),
                    (8.446e-02, 8.489e-02,-9.208e-03,-6.122e-03, 1.556e-02, 5.000e+02, 5.000e+02),
                    (9.811e-02, 9.817e-02, 1.986e-02, 9.554e-03, 1.447e-02, 5.780e+02, 5.780e+02),
                    (1.140e-01, 1.140e-01, 7.698e-03, 4.370e-03, 1.068e-02, 1.061e+03, 1.061e+03),
                    (1.324e-01, 1.342e-01, 9.733e-03,-4.264e-03, 1.037e-02, 1.125e+03, 1.125e+03),
                    (1.538e-01, 1.520e-01, 2.400e-03,-7.149e-03, 9.048e-03, 1.478e+03, 1.478e+03),
                    (1.786e-01, 1.772e-01, 2.964e-03,-1.999e-03, 7.230e-03, 2.315e+03, 2.315e+03),
                    (2.075e-01, 2.076e-01, 1.652e-03,-2.616e-03, 6.580e-03, 2.795e+03, 2.795e+03),
                    (2.410e-01, 2.411e-01,-1.277e-02, 2.982e-04, 5.274e-03, 4.350e+03, 4.350e+03),
                    (2.799e-01, 2.855e-01, 1.326e-03,-1.140e-03, 4.651e-03, 5.593e+03, 5.593e+03),
                    (3.252e-01, 3.270e-01,-2.517e-03, 3.061e-04, 4.554e-03, 5.834e+03, 5.834e+03),
                    (3.777e-01, 3.752e-01,-3.427e-03,-5.551e-03, 3.459e-03, 1.011e+04, 1.011e+04),
                    (4.387e-01, 4.393e-01, 2.000e-04,-2.414e-03, 3.095e-03, 1.263e+04, 1.263e+04),
                    (5.096e-01, 4.973e-01,-3.784e-03, 5.500e-04, 3.025e-03, 1.322e+04, 1.322e+04),
                    (5.920e-01, 5.873e-01, 2.131e-03, 9.000e-05, 3.673e-03, 8.971e+03, 8.971e+03),
                    (6.877e-01, 6.877e-01, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00),
                    (7.988e-01, 7.988e-01, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00),
                    (9.278e-01, 9.278e-01, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00)],
                    dtype=[("R_nominal",float),("<R>",float),("<gamT>",float),("<gamX>",float),
                           ("sig",float),("weight",float),("npairs",float)])
            
    def test_getCF(self):
        """Test getCF() directly, and the two kinds of file specification."""
        stile_args = {'corr2_kwargs': { 'ra_units': 'degrees', 
                                        'dec_units': 'degrees',
                                        'min_sep': 0.05,
                                        'max_sep': 1,
                                        'sep_units': 'degrees',
                                        'nbins': 20
                                        } }
        col_kwargs = {'ra_col': 2, 'dec_col': 3, 'g1_col': 5, 'g2_col': 6}
        cf = stile.sys_tests.CorrelationFunctionSysTest()
        dh = temp_data_handler()
        results = cf.getCF(stile_args,'ng',None,None,
                                            file_name='../examples/example_lens_catalog.dat',
                                            file_name2='../examples/example_source_catalog.dat',
                                            **col_kwargs)
        self.assertEqual(self.expected_result.dtype.names,results.dtype.names)
        numpy.testing.assert_array_equal(*helper.FormatSame(results,self.expected_result))
        kwargs = stile_args['corr2_kwargs']
        stile_args['corr2_kwargs'] = {}
        kwargs.update(col_kwargs)
        results2 = cf.getCF(stile_args,'ng',
                                            file_name='../examples/example_lens_catalog.dat',
                                            file_name2='../examples/example_source_catalog.dat',
                                             **kwargs)
        numpy.testing.assert_equal(results,results2)
        results2 = cf.getCF(stile_args,'ng',
                                             data=('../examples/example_lens_catalog.dat',
                                                   {'ra': 1, 'dec': 2, 'g1': 4, 'g2': 5}),
                                             data2=('../examples/example_source_catalog.dat',
                                                   {'ra': 1, 'dec': 2, 'g1': 4, 'g2': 5}),
                                             **kwargs)
        numpy.testing.assert_equal(results,results2)

        # Then, test the tests that use .getCF().
        realshear = stile.RealShearSysTest()
        results3 = realshear(stile_args,file_name='../examples/example_lens_catalog.dat',
                                        file_name2='../examples/example_source_catalog.dat',
                                        **kwargs)
        numpy.testing.assert_equal(results,results3)
        results3 = realshear(stile_args,data=('../examples/example_lens_catalog.dat',
                                              {'ra': 1, 'dec': 2, 'g1': 4, 'g2': 5}),
                                        data2=('../examples/example_source_catalog.dat',
                                              {'ra': 1, 'dec': 2, 'g1': 4, 'g2': 5}),
                                           **kwargs)
        numpy.testing.assert_equal(results,results3)
        self.assertRaises(TypeError,cf.getCF)
        self.assertRaises(subprocess.CalledProcessError,
                          cf.getCF,stile_args,'ng',
                          file_name='../examples/example_lens_catalog.dat', **col_kwargs)
        self.assertRaises(ValueError,cf.getCF,stile_args,'hello',
                          file_name='../examples/example_lens_catalog.dat',
                          file_name2='../examples/example_source_catalog.dat',
                          **col_kwargs)
    
if __name__=='__main__':
    print "Note: You may see some errors printed as we deliberately send wrong arguments to corr2.",
    print "You do not need to worry about these as long as the Python script completes correctly."
    unittest.main()

