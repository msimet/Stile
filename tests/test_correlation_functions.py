import sys
import os
import numpy
import subprocess
import helper
import unittest
import treecorr

try:
    import stile
except ImportError:
    sys.path.append('..')
    import stile


class temp_data_handler():
    def __init__(self):
        self.temp_dir = None

    def getOutputPath(self, p):
        return p


class TestCorrelationFunctions(unittest.TestCase):
    def setUp(self):
        # The output of our known data set from the example directory
        self.expected_result = numpy.array(
                   [(0.053888, 0.054426, 0.022059, -0.042588, 0.025785, 182.0, 182.0),
                    (0.062596, 0.062048, 0.0037377, 0.02995, 0.020788, 280.0, 280.0),
                    (0.072711, 0.072412, 0.018572, -0.041915, 0.017547, 393.0, 393.0),
                    (0.08446, 0.084892, -0.0092076, -0.006122, 0.015557, 500.0, 500.0),
                    (0.098107, 0.098174, 0.019855, 0.0095542, 0.014469, 578.0, 578.0),
                    (0.11396, 0.11403, 0.0076978, 0.0043697, 0.010679, 1061.0, 1061.0),
                    (0.13237, 0.13423, 0.0097326, -0.0042636, 0.010371, 1125.0, 1125.0),
                    (0.15376, 0.15199, 0.0024, -0.0071487, 0.0090482, 1478.0, 1478.0),
                    (0.17861, 0.17718, 0.0029644, -0.0019988, 0.0072297, 2315.0, 2315.0),
                    (0.20747, 0.20757, 0.0016518, -0.0026162, 0.0065797, 2795.0, 2795.0),
                    (0.241, 0.24111, -0.012766, 0.00029823, 0.0052742, 4350.0, 4350.0),
                    (0.27994, 0.28555, 0.0013257, -0.0011404, 0.0046513, 5593.0, 5593.0),
                    (0.32517, 0.32698, -0.0025167, 0.00030612, 0.0045542, 5834.0, 5834.0),
                    (0.37772, 0.37523, -0.0034272, -0.0055511, 0.0034589, 10114.0, 10114.0),
                    (0.43875, 0.43928, 0.00019999, -0.0024145, 0.0030951, 12631.0, 12631.0),
                    (0.50965, 0.49734, -0.0037836, 0.00055003, 0.0030254, 13220.0, 13220.0),
                    (0.592, 0.58727, 0.0021309, 9.0001e-05, 0.0036726, 8971.0, 8971.0),
                    (0.68766, 0.68766, 0.0, 0.0, 0.0, 0.0, 0.0),
                    (0.79877, 0.79877, 0.0, 0.0, 0.0, 0.0, 0.0),
                    (0.92784, 0.92784, 0.0, 0.0, 0.0, 0.0, 0.0)],
                    dtype=[("R_nom [deg]", float), ("<R> [deg]", float), ("<gamT>", float), ("<gamX>", float),
                           ("sigma", float), ("weight", float), ("npairs", float)])

        self.expected_result_meanlogr = numpy.array(
                   [(0.053888, 0.054459, -2.9109, 0.022059, -4.2588e-02, 0.025785, 182., 182.),
                    (0.062596, 0.062105, -2.7799, 0.0037377, 2.9950e-02, 0.020788, 280., 280.),
                    (0.072711, 0.072468, -2.6254, 0.018572, -4.1915e-02, 0.017547, 393., 393.),
                    (0.08446, 0.084946, -2.4664, -0.0092076, -6.1220e-03, 0.015557, 500., 500.),
                    (0.098107, 0.098256, -2.321, 0.019855, 9.5542e-03, 0.014469, 578., 578.),
                    (0.11396, 0.11411, -2.1713, 0.0076978, 4.3697e-03, 0.010679, 1061., 1061.),
                    (0.13237, 0.13435, -2.0082, 0.0097326, -4.2636e-03, 0.010371, 1125., 1125.),
                    (0.15376, 0.15209, -1.8839, 0.0024, -7.1487e-03, 0.0090482, 1478., 1478.),
                    (0.17861, 0.17727, -1.7306, 0.0029644, -1.9988e-03, 0.0072297, 2315., 2315.),
                    (0.20747, 0.20774, -1.5723, 0.0016518, -2.6162e-03, 0.0065797, 2795., 2795.),
                    (0.241, 0.24133, -1.4225, -0.012766, 2.9823e-04, 0.0052742, 4350., 4350.),
                    (0.27994, 0.28572, -1.2533, 0.0013257, -1.1404e-03, 0.0046513, 5593., 5593.),
                    (0.32517, 0.3273, -1.1179, -0.0025167, 3.0612e-04, 0.0045542, 5834., 5834.),
                    (0.37772, 0.37581, -0.98021, -0.0034272, -5.5511e-03, 0.0034589, 10114., 10114.),
                    (0.43875, 0.43984, -0.82263, 0.00019999, -2.4145e-03, 0.0030951, 12631., 12631.),
                    (0.50965, 0.49764, -0.69848, -0.0037836, 5.5003e-04, 0.0030254, 13220., 13220.),
                    (0.592, 0.58802, -0.53226, 0.0021309, 9.0001e-05, 0.0036726, 8971., 8971.),
                    (0.68766, 0.68766, -0.37447, 0., 0.0000e+00, 0., 0., 0.),
                    (0.79877, 0.79877, -0.22468, 0., 0.0000e+00, 0., 0., 0.),
                    (0.92784, 0.92784, -0.074893, 0., 0.0000e+00, 0., 0., 0.)],
                    dtype=[("R_nom [deg]", float), ("meanR [deg]", float), 
                           ("meanlogR [deg]", float), ("gamT", float), ("gamX", float),
                           ("sigma", float), ("weight", float), ("npairs", float)])

    def test_getCF(self):
        """Test getCF() directly, without first processing by child classes."""
        # Note: bin_slop 1 is not optimal for bins this small, but an earlier version of TreeCorr
        # did this by default with the settings here; we define bin_slop 1 for backwards 
        # compatibility
        stile_args = {'ra_units': 'degrees', 'dec_units': 'degrees', 'min_sep': 0.05, 'max_sep': 1,
                      'sep_units': 'degrees', 'nbins': 20, 'bin_slop': 1}
        cf = stile.sys_tests.CorrelationFunctionSysTest()
        dh = temp_data_handler()
        lens_data = stile.ReadASCIITable('../examples/example_lens_catalog.dat',
                    fields={'id': 0, 'ra': 1, 'dec': 2, 'z': 3, 'g1': 4, 'g2': 5})
        source_data = stile.ReadASCIITable('../examples/example_source_catalog.dat',
                    fields={'id': 0, 'ra': 1, 'dec': 2, 'z': 3, 'g1': 4, 'g2': 5})
        lens_catalog = treecorr.Catalog(ra=numpy.array([lens_data['ra']]),
                                        dec=numpy.array([lens_data['dec']]),
                                        ra_units='degrees', dec_units='degrees')
        source_catalog = treecorr.Catalog(ra=source_data['ra'], dec=source_data['dec'],
                                          g1=source_data['g1'], g2=source_data['g2'],
                                          ra_units='degrees', dec_units='degrees')
        results = cf.getCF('ng', lens_catalog, source_catalog, **stile_args)
        if "meanlogR [deg]" in results.dtype.names:
            expected = self.expected_result_meanlogr
        else:
            expected = self.expected_result
        numpy.testing.assert_array_equal(*helper.FormatSame(results, expected))
        results2 = cf.getCF('ng', lens_data, source_data, config=stile_args)
        self.assertEqual(expected.dtype.names, results.dtype.names)
        # Missing necessary data file
        numpy.testing.assert_equal(results, results2)
        self.assertRaises(TypeError,
                          cf.getCF, {}, 'ng',
                          file_name='../examples/example_lens_catalog.dat')
        # Nonsensical correlation type
        self.assertRaises(ValueError, cf.getCF, 'hello', lens_data, source_data, config=stile_args)

        # Then, test a test that uses .getCF().
        realshear = stile.sys_tests.GalaxyShearSysTest()
        results3 = realshear(lens_data, source_data, config=stile_args)
        numpy.testing.assert_equal(results, results3)
        
        
    def test_generator(self):
        """Make sure the CorrelationFunctionSysTest() generator returns the right objects"""
        object_list = ['GalaxyShear', 'BrightStarShear', 'StarXGalaxyDensity', 'StarXGalaxyShear', 
                       'StarXStarShear', 'GalaxyDensityCorrelation', 'StarDensityCorrelation']
        for object_type in object_list:
            object_1 = stile.CorrelationFunctionSysTest(object_type)
            object_2 = eval('stile.sys_tests.'+object_type+'SysTest()')
            self.assertEqual(type(object_1),type(object_2))
        
        self.assertRaises(ValueError,stile.CorrelationFunctionSysTest,'hello')
        self.assertEqual(type(stile.sys_tests.BaseCorrelationFunctionSysTest()), 
                         type(stile.CorrelationFunctionSysTest()))
        
    
if __name__=='__main__':
    unittest.main()

