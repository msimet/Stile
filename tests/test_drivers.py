try:
    import stile
except:
    import sys
    sys.path.append('..')
    import stile
import numpy
import unittest
import types
import os
import glob

class TestDrivers(unittest.TestCase):
    def setUp(self):
        self.filename = '__stile_test_file'
        def getOutputPath(other_self, *args):
            n = len(glob.glob(os.path.join(self.filename)+'*'+args[-1]))
            if n:
                return self.filename+str(n+1)+args[-1]
            else:
                return self.filename+args[-1]

        self.driver = stile.ConfigDriver()
        
        self.config = stile.ConfigDataHandler({})
        setattr(self.config, 'getOutputPath', types.MethodType(getOutputPath, self.config))
        
        self.finished_tests = []
        def runTest(original_self, config, data, sys_test_list, name):
            self.finished_tests.append((data[0][0], sys_test_list, "single"))
        def runMultiTest(original_self, config, data, sys_test_list, name):
            self.finished_tests.append(([d[0][0] for d in data], sys_test_list, "multi"))
        self.driver_captured = stile.ConfigDriver()
        setattr(self.driver_captured, '_runMultiSysTestHelper', types.MethodType(runMultiTest, self.driver_captured))
        setattr(self.driver_captured, '_runSysTestHelper', types.MethodType(runTest, self.driver_captured))
        
        self.config_1 = {'single':
                            {'CCD': 
                                {'catalog': {
                                    'galaxy lens': '../examples/example_lens_catalog.dat', 
                                    'galaxy': '../examples/example_source_catalog.dat'}}},
                         'fields': {'id': 0, 'ra': 1, 'dec': 2, 'z': 3, 'g1': 4, 'g2': 5}}
        self.tests_1 = [{'epoch': 'single', 'extent': 'CCD', 'format': 'catalog', 'name': 'CorrelationFunctionSysTest', 'type': 'GalaxyShear', 'extra_args': {'ra_units': 'degrees', 'dec_units': 'degrees', 'min_sep': 0.05, 'max_sep': 1, 'sep_units': 'degrees', 'nbins': 20}}]
        self.tests_dict_1 = {'sys_test': stile.GalaxyShearSysTest(), 'extra_args': self.tests_1[0]['extra_args'], 'bins': [], 'bin_list': []}
        self.expected_result_1 = numpy.array(
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
                    dtype=[("R_nom",float),("<R>",float),("<gamT>",float),("<gamX>",float),
                           ("sigma",float),("weight",int),("npairs",int)])
        self.config_1_flags = {'single':
                            {'CCD': 
                                {'catalog': {
                                    'galaxy lens': {'name': 'test_data/combo_example_catalog.dat', 'flag_field': {'obj': 0}}, 
                                    'galaxy': {'name': 'test_data/combo_example_catalog.dat', 'flag_field': {'obj': 1}}}}},
                         'fields': {'id': 0, 'ra': 1, 'dec': 2, 'z': 3, 'g1': 4, 'g2': 5, 'obj': 6}}
        
        self.tests_2 = [{'epoch': 'single', 'extent': 'CCD', 'format': 'catalog', 'name': 'StatSysTest', 'field': 'g1'},
                        {'epoch': 'single', 'extent': 'CCD', 'format': 'catalog', 'name': 'StatSysTest', 'field': 'ra', 'bins': {'name': 'BinStep', 'low': -100, 'high': 100, 'n_bins': 2, 'field': 'ra'}}]
        self.bins = stile.BinStep(low=-100, high=100, n_bins=2, field='ra')
        self.tests_dict_2a = {'sys_test': stile.StatSysTest(field='g1'), 'extra_args': {}, 'bins': [], 'bin_list': []}
        self.tests_dict_2b = {'sys_test': stile.StatSysTest(field='ra'), 'extra_args': {}, 'bins': [self.bins], 'bin_list': [self.bins]}
        
        self.data = stile.ReadTable('../examples/example_source_catalog.dat', fields={'id': 0, 'ra': 1, 'dec': 2, 'z': 3, 'g1': 4, 'g2': 5})
        self.lens_data = stile.ReadTable('../examples/example_lens_catalog.dat', fields={'id': 0, 'ra': 1, 'dec': 2, 'z': 3, 'g1': 4, 'g2': 5})
        self.expected_result_2a = stile.StatSysTest()(self.data['g1'])
        # Call to get the SingleBin objects
        self.expected_result_2b = [stile.StatSysTest()(bin(self.data)['ra']) for bin in self.bins()]
        
    def tearDown(self):
        files = glob.glob(self.filename+'*')
        for file in files:
            os.remove(file)
        for letter in ['a', 'b', 'c', 'd', 'e', 'f']:
            if os.path.isfile(letter):
                os.remove(letter)

    def test_sys_test_helper(self):
        self.driver._runSysTestHelper(config=self.config, data=self.data, sys_test=self.tests_dict_2a, name='name')
        with open(self.filename+'.txt') as f:
            results = f.read()
        self.assertEqual(results, str(self.expected_result_2a)+'\n')
        for i, (bin, expected) in enumerate(zip(self.bins(), self.expected_result_2b)):
            self.driver._runSysTestHelper(config=self.config, data=bin(self.data), sys_test=self.tests_dict_2b, name='name')
            with open(self.filename+str(i+2)+'.txt') as f:
                results = f.read()
            self.assertEqual(results, str(expected)+'\n')
    
    def test_multi_sys_test_helper(self):
        self.driver._runMultiSysTestHelper(config=self.config, data=[self.lens_data, self.data], sys_test=self.tests_dict_1, name='name')
        results = stile.ReadTable(self.filename+'.txt', read_header=True)
        numpy.testing.assert_equal(results, self.expected_result_1)
    
    def test_run_sys_tests(self):
        self.driver.RunSysTests(config=self.config, data=self.data, sys_test_list = [self.tests_dict_2a, self.tests_dict_2b], name='name')
        with open(self.filename+'.txt') as f:
            results = f.read()
        self.assertEqual(results, str(self.expected_result_2a)+'\n')
        with open(self.filename+'2.txt') as f:
            results = f.read()
        self.assertEqual(results, str(self.expected_result_2b[0])+'\n')
        with open(self.filename+'3.txt') as f:
            results = f.read()
        self.assertEqual(results, str(self.expected_result_2b[1])+'\n')

    def test_run_multi_sys_tests(self):
        self.driver.RunMultiSysTests(config=self.config, data=[self.lens_data, self.data], sys_test_list=[self.tests_dict_1], name='name')
        results = stile.ReadTable(self.filename+'.txt', read_header=True)
        numpy.testing.assert_equal(results, self.expected_result_1)
    
    def test_run_all_tests(self):
        # make some little temporary files
        for letter in ['a', 'b', 'c', 'd', 'e', 'f']:
            with open(letter, 'w') as f:
                f.write(letter+' '+letter+'\n')  # Two so it makes an array, not a scalar, when read in
        complicated_dict = {
            'single': {
                'field': {
                    'catalog': {
                        'galaxy': ['a', 'b', 'c'],
                        'galaxy lens': ['d', 'e', 'f']} },
                'CCD': {
                    'catalog': {
                        'galaxy': ['a', 'a', 'b'],
                        'galaxy lens': ['d', 'e', 'f']} } } }
        complicated_test_dict = [{'name': 'CorrelationFunction', 'type': 'GalaxyShear'}, {'name': 'Stat', 'field': 'ra', 'object_type': 'galaxy'}, {'name': 'Stat', 'field': 'g1', 'extent': 'CCD', 'object_type': 'galaxy lens'}]
        config = stile.ConfigDataHandler({'files': complicated_dict, 'sys_tests': complicated_test_dict})
        self.driver_captured(config)
        # What's happening here?
        # - First we hit d,a from the groupings in single-field-catalog
        # - d has no tests *here* so we move on to a, which we also loaded in
        # - ditto for next three
        # - then we get to 'd' again.  Here it has a single-data-set test, so we do it
        # - then d,a shear test
        # - then the a single test, then the e,a multi test since we already had a, then single-e
        # - then on to f as normal
        expected_file_letters = [['d', 'a'], 'a', ['e', 'b'], 'b', ['f', 'c'], 'c', 'd', ['d', 'a'], 'a', ['e', 'a'], 'e', 'f', ['f', 'b'], 'b']
        gs = stile.GalaxyShearSysTest
        st = stile.StatSysTest
        expected_tests = [gs, st, gs, st, gs, st, st, gs, st, gs, st, st, gs, st]
        for res, letters, test_type in zip(self.finished_tests, expected_file_letters, expected_tests):
            self.assertEqual(res[0], letters)
            self.assertIsInstance(res[1]['sys_test'], test_type)
            self.assertEqual(res[1]['bin_list'], [])
            self.assertEqual(res[1]['extra_args'] , {})
    
    
    
if __name__=='__main__':
    unittest.main()
    