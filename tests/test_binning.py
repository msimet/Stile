import numpy
import sys
import unittest

try:
    import stile
except ImportError:
    sys.path.append('..')
    import stile


def binfunction(x):
    return numpy.ceil(x)

def compare_single_bin(b1, b2):
    if b1.field != b2.field:
        return False
    return numpy.allclose([b1.low, b1.high], [b2.low, b2.high])

class TestBinning(unittest.TestCase):
    def setUp(self):
        bin_array_1 = [[0.5], [1.5], [2.5], [3.5], [4.5]]
        self.bin_array_1 = numpy.array([tuple(b) for b in bin_array_1], dtype=[('field_0', float)])
        bin_array_2 = [[1], [2], [3], [4], [5]]
        self.bin_array_2 = numpy.array([tuple(b) for b in bin_array_2], dtype=[('field_0', float)])
        bin_array_3 = [[0.5], [0.5], [5.5], [4.5], [3.5]]
        self.bin_array_3 = numpy.array([tuple(b) for b in bin_array_3], dtype=[('field_0', float)])
        bin_array_4 = [0.5, 1.5, 2.5, 3.5, 4.5]
        self.bin_array_4 = numpy.array([tuple(bin_array_4)],
                                  dtype=[('field_0', float), ('field_1', float), ('field_2', float),
                                         ('field_3', float), ('field_4', float)])
        bin_array_5 = [[1., 2.], [3., 4.], [5., 6.]]
        self.bin_array_5 = numpy.array([tuple(b) for b in bin_array_5],
                                  dtype=[('field_0', float), ('field_1', float)])
        bin_array_6 = [[-1], [1]]
        self.bin_array_6 = numpy.array([tuple(b) for b in bin_array_6], dtype=[('field_0', float)])

    def test_BinStep_SingleBin_creation(self):
        """Test that the constructor for SingleBin and BinStep objects behaves appropriately given
        various inputs."""
        # All of these should return the same objects (the expected_obj_list), except the final one,
        # which should return them in the reverse order.
        lhs = stile.BinStep(field='field_0', low=0, high=6, step=1)
        lhn = stile.BinStep(field='field_0', low=0, high=6, n_bins=6)
        lsn = stile.BinStep(field='field_0', low=0, step=1, n_bins=6)
        hsn = stile.BinStep(field='field_0', high=6, step=1, n_bins=6)
        reverse_lhs = stile.BinStep(field='field_0', low=6, high=0, step=-1)

        expected_obj_list = [stile.binning.SingleBin(field='field_0', low=0, high=1, short_name='b'),
                             stile.binning.SingleBin(field='field_0', low=1, high=2, short_name='b'),
                             stile.binning.SingleBin(field='field_0', low=2, high=3, short_name='b'),
                             stile.binning.SingleBin(field='field_0', low=3, high=4, short_name='b'),
                             stile.binning.SingleBin(field='field_0', low=4, high=5, short_name='b'),
                             stile.binning.SingleBin(field='field_0', low=5, high=6, short_name='b')]

        names = ["passed low, high, and step",
                 "passed low, high, and n_bins",
                 "passed low, step, and n_bins",
                 "passed high, step, and n_bins",
                 "passed low, high, and step with low and high reversed"]
        objs = [lhs, lhn, lsn, hsn, reverse_lhs]
        for obj, name in zip(objs, names):
            obj_list = obj()
            if obj == reverse_lhs:
                obj_list.reverse()
            self.assertEqual(len(obj_list), 6,
                             msg='BinStep ('+name+') created wrong number of SingleBins!')
            for i in range(len(obj_list)):
                self.assertTrue(compare_single_bin(obj_list[i], expected_obj_list[i]),
                             msg='BinStep ('+name+') created incorrect SingleBins!')

        # As above, but using logarithmic bins.
        lhs = stile.BinStep(field='field_0', low=0.25, high=8, step=numpy.log(2.), use_log=True)
        lhn = stile.BinStep(field='field_0', low=0.25, high=8, n_bins=5, use_log=True)
        lsn = stile.BinStep(field='field_0', low=0.25, step=numpy.log(2.), n_bins=5, use_log=True)
        hsn = stile.BinStep(field='field_0', high=8, step=numpy.log(2.), n_bins=5, use_log=True)
        reverse_lhs = stile.BinStep(field='field_0', low=8, high=0.25, step=-numpy.log(2.),
                                    use_log=True)

        expected_obj_list = [stile.binning.SingleBin(field='field_0', low=0.25, high=0.5, short_name='b'),
                             stile.binning.SingleBin(field='field_0', low=0.5, high=1., short_name='b'),
                             stile.binning.SingleBin(field='field_0', low=1., high=2., short_name='b'),
                             stile.binning.SingleBin(field='field_0', low=2., high=4., short_name='b'),
                             stile.binning.SingleBin(field='field_0', low=4., high=8., short_name='b')]

        names = ["passed low, high, and step",
                 "passed low, high, and n_bins",
                 "passed low, step, and n_bins",
                 "passed high, step, and n_bins",
                 "passed low, high, and step with low and high reversed"]
        objs = [lhs, lhn, lsn, hsn, reverse_lhs]
        for obj, name in zip(objs, names):
            obj_list = obj()
            if obj == reverse_lhs:
                obj_list.reverse()
            self.assertEqual(len(obj_list), 5,
                             msg='Log BinStep ('+name+') created wrong number of SingleBins!')
            for i in range(len(obj_list)):
                self.assertTrue(compare_single_bin(obj_list[i], expected_obj_list[i]),
                             msg='BinStep ('+name+') created incorrect SingleBins!')

    def test_BinList_SingleBin_creation(self):
        """Test that the creation of a SingleBin exhibits appropriate behavior."""
        obj = stile.BinList([0, 1.1, 1.9, 3.0, 4.0, 5.0, 6.5], 'field_0')

        expected_obj_list = [stile.binning.SingleBin('field_0', low=0, high=1.1, short_name='b'),
                             stile.binning.SingleBin('field_0', low=1.1, high=1.9, short_name='b'),
                             stile.binning.SingleBin('field_0', low=1.9, high=3, short_name='b'),
                             stile.binning.SingleBin('field_0', low=3, high=4, short_name='b'),
                             stile.binning.SingleBin('field_0', low=4, high=5, short_name='b'),
                             stile.binning.SingleBin('field_0', low=5, high=6.5, short_name='b')]

        obj_list = obj()

        self.assertEqual(len(obj_list), 6,
                         msg='BinList created wrong number of SingleBins!')
        for i in range(len(obj_list)):
            self.assertTrue(compare_single_bin(obj_list[i], expected_obj_list[i]),
                         msg='BinList created incorrect SingleBins!')

        obj = stile.BinList([6.5, 5.0, 4.0, 3.0, 1.9, 1.1, 0], field='field_0')
        obj_list = obj()
        obj_list.reverse()
        self.assertEqual(len(obj_list), 6,
                         msg='Reversed BinList created wrong number of SingleBins!')
        for i in range(len(obj_list)):
            self.assertTrue(compare_single_bin(obj_list[i], expected_obj_list[i]),
                         msg='Reversed BinList created incorrect SingleBins!')
        self.assertRaises(ValueError, stile.BinList, [0.5, 1.5, 1.0])

    def test_BinStep_linear(self):
        """Test that BinStep objects with linear spacing behave appropriately."""
        lhs = stile.BinStep('field_0', low=0, high=6, step=1)
        lhn = stile.BinStep('field_0', low=0, high=6, n_bins=6)
        lsn = stile.BinStep('field_0', low=0, step=1, n_bins=6)
        hsn = stile.BinStep('field_0', high=6, step=1, n_bins=6)
        reverse_lhs = stile.BinStep('field_0', low=6, high=0, step=-1)

        names = ["passed low, high, and step",
                 "passed low, high, and n_bins",
                 "passed low, step, and n_bins",
                 "passed high, step, and n_bins",
                 "passed low, high, and step with low and high reversed"]
        objs = [lhs, lhn, lsn, hsn, reverse_lhs]

        # Expected results; each item of the list is the result of the n-th SingleBin.
        # Formatted arrays don't compare properly to non-formatted arrays, so we use slices of the
        # original array to ensure the formatting matches properly even for empty (formatted)
        # arrays.
        expected_bin_array_1 = [self.bin_array_1['field_0'][0], self.bin_array_1['field_0'][1],
                                self.bin_array_1['field_0'][2], self.bin_array_1['field_0'][3],
                                self.bin_array_1['field_0'][4], self.bin_array_1['field_0'][:0]]
        expected_bin_array_2 = [self.bin_array_2['field_0'][:0], self.bin_array_2['field_0'][0],
                                self.bin_array_2['field_0'][1], self.bin_array_2['field_0'][2],
                                self.bin_array_2['field_0'][3], self.bin_array_2['field_0'][4]]
        expected_bin_array_3 = [self.bin_array_3['field_0'][0:2], self.bin_array_3['field_0'][:0],
                                self.bin_array_3['field_0'][:0], self.bin_array_3['field_0'][4],
                                self.bin_array_3['field_0'][3], self.bin_array_3['field_0'][2]]
        expected_bin_array_4 = [self.bin_array_4['field_0'][0], self.bin_array_4['field_0'][:0],
                                self.bin_array_4['field_0'][:0], self.bin_array_4['field_0'][:0],
                                self.bin_array_4['field_0'][:0], self.bin_array_4['field_0'][:0]]
        expected_bin_array_5 = [self.bin_array_5['field_0'][:0], self.bin_array_5['field_0'][0],
                                self.bin_array_5['field_0'][:0], self.bin_array_5['field_0'][1],
                                self.bin_array_5['field_0'][:0], self.bin_array_5['field_0'][2]]
        expected_bin_array_6 = [self.bin_array_6['field_0'][:0], self.bin_array_6['field_0'][1],
                                self.bin_array_6['field_0'][:0], self.bin_array_6['field_0'][:0],
                                self.bin_array_6['field_0'][:0], self.bin_array_6['field_0'][:0]]

        for obj, name in zip(objs, names):
            err_msg = ("BinStep test ("+name+
                       ") failed to produce correct binning for array %s in bin # %i")
            obj_list = obj()
            self.assertEqual(len(obj_list), 6,
                             msg=('Wrong number of bins created from BinStep with '+
                                  name+': '+str(len(obj_list))))

            if obj == reverse_lhs:
                obj_list.reverse()
            for i, singlebin in enumerate(obj_list):
                results = singlebin(self.bin_array_1)
                numpy.testing.assert_equal(results, expected_bin_array_1[i],
                                           err_msg=err_msg%(self.bin_array_1, i))
                results = singlebin(self.bin_array_2)
                numpy.testing.assert_equal(results, expected_bin_array_2[i],
                                           err_msg=err_msg%(self.bin_array_2, i))
                results = singlebin(self.bin_array_3)
                numpy.testing.assert_equal(results, expected_bin_array_3[i],
                                           err_msg=err_msg%(self.bin_array_3, i))
                results = singlebin(self.bin_array_4)
                numpy.testing.assert_equal(results, expected_bin_array_4[i],
                                           err_msg=err_msg%(self.bin_array_4, i))
                results = singlebin(self.bin_array_5)
                numpy.testing.assert_equal(results, expected_bin_array_5[i],
                                           err_msg=err_msg%(self.bin_array_5, i))
                results = singlebin(self.bin_array_6)
                numpy.testing.assert_equal(results, expected_bin_array_6[i],
                                           err_msg=err_msg%(self.bin_array_6, i))

    def test_BinStep_log(self):
        """Test that BinStep objects with logarithmic spacing behave appropriately."""
        lhs = stile.BinStep(field='field_0', low=0.25, high=8, step=numpy.log(2.), use_log=True)
        lhn = stile.BinStep(field='field_0', low=0.25, high=8, n_bins=5, use_log=True)
        lsn = stile.BinStep(field='field_0', low=0.25, step=numpy.log(2.), n_bins=5, use_log=True)
        hsn = stile.BinStep(field='field_0', high=8, step=numpy.log(2.), n_bins=5, use_log=True)
        reverse_lhs = stile.BinStep(field='field_0', low=8, high=0.25, step=-numpy.log(2.),
                                    use_log=True)
        names = ["passed low, high, and step",
                 "passed low, high, and n_bins",
                 "passed low, step, and n_bins",
                 "passed high, step, and n_bins",
                 "passed low, high, and step with low and high reversed"]

        objs = [lhs, lhn, lsn, hsn, reverse_lhs]

        expected_bin_array_1 = [self.bin_array_1['field_0'][:0], self.bin_array_1['field_0'][0],
                                self.bin_array_1['field_0'][1], self.bin_array_1['field_0'][2:4],
                                self.bin_array_1['field_0'][4]]
        expected_bin_array_2 = [self.bin_array_2['field_0'][:0], self.bin_array_2['field_0'][:0],
                                self.bin_array_2['field_0'][0], self.bin_array_2['field_0'][1:3],
                                self.bin_array_2['field_0'][3:]]
        expected_bin_array_3 = [self.bin_array_3['field_0'][:0], self.bin_array_3['field_0'][:2],
                                self.bin_array_3['field_0'][:0], self.bin_array_3['field_0'][4],
                                self.bin_array_3['field_0'][2:4]]
        expected_bin_array_4 = [self.bin_array_4['field_0'][:0], self.bin_array_4['field_0'][0],
                                self.bin_array_4['field_0'][:0], self.bin_array_4['field_0'][:0],
                                self.bin_array_4['field_0'][:0]]
        expected_bin_array_5 = [self.bin_array_5['field_0'][:0], self.bin_array_5['field_0'][:0],
                                self.bin_array_5['field_0'][0], self.bin_array_5['field_0'][1],
                                self.bin_array_5['field_0'][2]]
        expected_bin_array_6 = [self.bin_array_6['field_0'][:0], self.bin_array_6['field_0'][:0],
                                self.bin_array_6['field_0'][1], self.bin_array_6['field_0'][:0],
                                self.bin_array_6['field_0'][:0]]

        for obj, name in zip(objs, names):
            err_msg = ("Logarithmic BinStep test ("+name+
                       ") failed to produce correct binning for array %s in bin # %i")
            obj_list = obj()
            self.assertEqual(len(obj_list), 5,
                             msg=('Wrong number of bins created from logarithmic BinStep with '+
                                  name+': '+str(len(obj_list))))
            if obj == reverse_lhs:
                obj_list.reverse()
            for i, singlebin in enumerate(obj_list):
                results = singlebin(self.bin_array_1)
                numpy.testing.assert_equal(results, expected_bin_array_1[i],
                                           err_msg=err_msg%(self.bin_array_1, i))
                results = singlebin(self.bin_array_2)
                numpy.testing.assert_equal(results, expected_bin_array_2[i],
                                           err_msg=err_msg%(self.bin_array_2, i))
                results = singlebin(self.bin_array_3)
                numpy.testing.assert_equal(results, expected_bin_array_3[i],
                                           err_msg=err_msg%(self.bin_array_3, i))
                results = singlebin(self.bin_array_4)
                numpy.testing.assert_equal(results, expected_bin_array_4[i],
                                           err_msg=err_msg%(self.bin_array_4, i))
                results = singlebin(self.bin_array_5)
                numpy.testing.assert_equal(results, expected_bin_array_5[i],
                                           err_msg=err_msg%(self.bin_array_5, i))
                results = singlebin(self.bin_array_6)
                numpy.testing.assert_equal(results, expected_bin_array_6[i],
                                           err_msg=err_msg%(self.bin_array_6, i))

    def test_BinList(self):
        """Test that BinList objects behave appropriately with respect to SingleBin behavior."""
        obj_forward = stile.BinList([0, 1., 1.9, 3.0, 4.0, 5.0, 6.5], field='field_0')
        obj_reverse = stile.BinList([6.5, 5.0, 4.0, 3.0, 1.9, 1., 0], field='field_0')

        names = [" ", " (reversed) "]
        objs = [obj_forward, obj_reverse]

        # Expected results; each item of the list is the result of the n-th SingleBin
        expected_bin_array_1 = [self.bin_array_1['field_0'][0], self.bin_array_1['field_0'][1],
                                self.bin_array_1['field_0'][2], self.bin_array_1['field_0'][3],
                                self.bin_array_1['field_0'][4], self.bin_array_1['field_0'][:0]]
        expected_bin_array_2 = [self.bin_array_2['field_0'][:0], self.bin_array_2['field_0'][0], 
                                self.bin_array_2['field_0'][1], self.bin_array_2['field_0'][2],
                                self.bin_array_2['field_0'][3], self.bin_array_2['field_0'][4]]
        expected_bin_array_3 = [self.bin_array_3['field_0'][0:2], self.bin_array_3['field_0'][:0],
                                self.bin_array_3['field_0'][:0], self.bin_array_3['field_0'][4],
                                self.bin_array_3['field_0'][3], self.bin_array_3['field_0'][2]]
        expected_bin_array_4 = [self.bin_array_4['field_0'][0], self.bin_array_4['field_0'][:0],
                                self.bin_array_4['field_0'][:0], self.bin_array_4['field_0'][:0],
                                self.bin_array_4['field_0'][:0], self.bin_array_4['field_0'][:0]]
        expected_bin_array_5 = [self.bin_array_5['field_0'][:0], self.bin_array_5['field_0'][0],
                                self.bin_array_5['field_0'][:0], self.bin_array_5['field_0'][1],
                                self.bin_array_5['field_0'][:0], self.bin_array_5['field_0'][2]]
        expected_bin_array_6 = [self.bin_array_6['field_0'][:0], self.bin_array_6['field_0'][1],
                                self.bin_array_6['field_0'][:0], self.bin_array_6['field_0'][:0],
                                self.bin_array_6['field_0'][:0], self.bin_array_6['field_0'][:0]]

        for obj, name in zip(objs, names):
            err_msg = ("BinList"+name+"failed to produce correct binning for array %s in bin # %i")
            obj_list = obj()
            self.assertEqual(len(obj_list), 6,
                             msg=('Wrong number of bins created from BinList'+name+': '+
                                     str(len(obj_list))))
            if obj == obj_reverse:
                obj_list.reverse()
            for i, singlebin in enumerate(obj_list):
                results = singlebin(self.bin_array_1)
                numpy.testing.assert_equal(results, expected_bin_array_1[i],
                                           err_msg=err_msg%(self.bin_array_1, i))
                results = singlebin(self.bin_array_2)
                numpy.testing.assert_equal(results, expected_bin_array_2[i],
                                           err_msg=err_msg%(self.bin_array_2, i))
                results = singlebin(self.bin_array_3)
                numpy.testing.assert_equal(results, expected_bin_array_3[i],
                                           err_msg=err_msg%(self.bin_array_3, i))
                results = singlebin(self.bin_array_4)
                numpy.testing.assert_equal(results, expected_bin_array_4[i],
                                           err_msg=err_msg%(self.bin_array_4, i))
                results = singlebin(self.bin_array_5)
                numpy.testing.assert_equal(results, expected_bin_array_5[i],
                                           err_msg=err_msg%(self.bin_array_5, i))
                results = singlebin(self.bin_array_6)
                numpy.testing.assert_equal(results, expected_bin_array_6[i],
                                           err_msg=err_msg%(self.bin_array_6, i))

    def test_bin_creation_errors(self):
        """Test for initialization errors and proper treatment of weird arguments."""
        # Invalid bounds in logarithmic BinStep
        self.assertRaises(ValueError, stile.BinStep, 'c', low=0, high=10, step=1, use_log=True)
        self.assertRaises(ValueError, stile.BinStep, 'c', low=10, high=-1, step=-1, use_log=True)
        # Various not-enough-arguments errors to BinStep (probably overkill)
        self.assertRaises(TypeError, stile.BinStep)
        self.assertRaises(TypeError, stile.BinStep, 'c')
        self.assertRaises(TypeError, stile.BinStep, 'c', low=1)
        self.assertRaises(TypeError, stile.BinStep, 'c', low=1, high=2)
        self.assertRaises(TypeError, stile.BinStep, 'c', low=1, step=2)
        self.assertRaises(TypeError, stile.BinStep, 'c', low=1, n_bins=2)
        self.assertRaises(TypeError, stile.BinStep, 'c', step=1)
        self.assertRaises(TypeError, stile.BinStep, 'c', step=1, n_bins=2)
        self.assertRaises(TypeError, stile.BinStep, 'c', step=1, high=2)
        self.assertRaises(TypeError, stile.BinStep, 'c', n_bins=1)
        self.assertRaises(TypeError, stile.BinStep, 'c', n_bins=1, high=2)
        self.assertRaises(TypeError, stile.BinStep, 'c', high=2)
        # Inconsistent and nonsense arguments to BinStep
        self.assertRaises(ValueError, stile.BinStep, 'c', low=1, high=0, step=0.5)
        self.assertRaises(ValueError, stile.BinStep, 'c', low=0, high=1, step=-0.5)
        self.assertRaises(ValueError, stile.BinStep, 'c', low=0, high=5, step=1, n_bins=7)
        stile.BinStep('c', low=0, high=-1, step=-0.5)  # actually consistent
        self.assertRaises(ValueError, stile.BinStep, 'c', low=1, high=1, step=0.5)
        self.assertRaises(ValueError, stile.BinStep, 'c', low=1, high=2, n_bins=-1)
        self.assertRaises(TypeError, stile.BinStep, 0, low=0, high=5, step=1)
        # Wrong arguments to BinList
        self.assertRaises(TypeError, stile.BinList, [1, 2, 3], 'c', n_bins=1)
        self.assertRaises(ValueError, stile.BinList, [1, 3, 2], 'c')
        self.assertRaises(TypeError, stile.BinList, [1, 3], 0)
        self.assertRaises(TypeError, stile.BinList, [], 'c')
        self.assertRaises(ValueError, stile.BinList, [1, 3, 2])
        self.assertRaises(TypeError, stile.BinList, 'c')

    def test_singlebin_input_errors(self):
        """Test that SingleBin objects appropriately object to strange input."""
        sb = stile.binning.SingleBin(field='field_0', low=0, high=10, short_name='boo')
        sfb = stile.binning.SingleFunctionBin(binfunction, 1)
        self.assertIsNotNone(sb.long_name)  # check that this was made properly
        self.assertRaises(TypeError, sb, [1, 2, 3, 4])
        self.assertRaises((IndexError, ValueError), sb, numpy.array([1, 2, 3, 4]))
        self.assertRaises(ValueError, sb,
                          numpy.array([(1, ), (2, ), (3, ), (4, )], dtype=[('field_1', int)]))
        self.assertRaises(TypeError, sb, 3)

    def test_ExpandBinList(self):
        """Test the function that takes a set of objects which each generate a list and returns all
        possible sets of one object from each list, in the order we expect it to do that."""
        # ExpandBinList needs something that returns callable object
        def return_objs(x, n):
            def func():
                return [str(nn)+x for nn in range(n)]
            return func
        results = stile.ExpandBinList([return_objs('a', 3),
                                       return_objs('b', 2),
                                       return_objs('c', 4)])
        numpy.testing.assert_equal(results,
                                       [('0a', '0b', '0c'), ('0a', '0b', '1c'), ('0a', '0b', '2c'),
                                        ('0a', '0b', '3c'), ('0a', '1b', '0c'), ('0a', '1b', '1c'),
                                        ('0a', '1b', '2c'), ('0a', '1b', '3c'), ('1a', '0b', '0c'),
                                        ('1a', '0b', '1c'), ('1a', '0b', '2c'), ('1a', '0b', '3c'),
                                        ('1a', '1b', '0c'), ('1a', '1b', '1c'), ('1a', '1b', '2c'),
                                        ('1a', '1b', '3c'), ('2a', '0b', '0c'), ('2a', '0b', '1c'),
                                        ('2a', '0b', '2c'), ('2a', '0b', '3c'), ('2a', '1b', '0c'),
                                        ('2a', '1b', '1c'), ('2a', '1b', '2c'), ('2a', '1b', '3c')])
        numpy.testing.assert_equal(stile.ExpandBinList(None), [])
        numpy.testing.assert_equal(stile.ExpandBinList([]), [])
        bin_obj0 = stile.BinStep(field='column_0', low=0, high=6, n_bins=2)
        bin_obj1 = stile.BinList([0, 2, 4], 'column_1')
        results = stile.ExpandBinList([bin_obj0, bin_obj1])
        expected_results = [(stile.binning.SingleBin('column_0', low=0, high=3, short_name='b'),
                             stile.binning.SingleBin('column_1', low=0, high=2, short_name='b')),
                            (stile.binning.SingleBin('column_0', low=0, high=3, short_name='b'),
                             stile.binning.SingleBin('column_1', low=2, high=4, short_name='b')),
                            (stile.binning.SingleBin('column_0', low=3, high=6, short_name='b'),
                             stile.binning.SingleBin('column_1', low=0, high=2, short_name='b')),
                            (stile.binning.SingleBin('column_0', low=3, high=6, short_name='b'),
                             stile.binning.SingleBin('column_1', low=2, high=4, short_name='b'))]
        numpy.testing.assert_equal(len(results), len(expected_results))
        for rpair, epair in zip(results, expected_results):
            self.assertTrue(compare_single_bin(rpair[0], epair[0]))
            self.assertTrue(compare_single_bin(rpair[1], epair[1]))
        self.assertRaises(TypeError, stile.ExpandBinList, bin_obj0, bin_obj1)


if __name__ == '__main__':
    unittest.main()
