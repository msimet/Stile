import numpy
import sys
import math
import unittest
try:
    import stile
except ImportError:
    sys.path.append('..')
    import stile


def funcname():
    import inspect
    return inspect.stack()[1][3]


class TestStats(unittest.TestCase):
    def setUp(self):
        self.rand_seed = 314159  # random seed for tests
        self.n_points_test = 800000  # number of random points to use for unit tests of stats module
        self.gaussian_mean = 27.0  # Mean value for the Gaussian from which to draw random points
        self.gaussian_sigma = 4.0  # Sigma value for the Gaussian from which to draw random points

    def check_results(self, stats_obj):
        """A utility to check whether a stats object contains results consistent with inputs."""
        numpy.testing.assert_equal(self.n_points_test, stats_obj.N)
        numpy.testing.assert_almost_equal(self.gaussian_mean/stats_obj.mean-1., 0., decimal=3,
                                       err_msg='Unexpected result for mean of random numbers!')
        numpy.testing.assert_almost_equal(self.gaussian_mean/stats_obj.median-1., 0., decimal=3,
                                       err_msg='Unexpected result for median of random numbers!')
        numpy.testing.assert_almost_equal(self.gaussian_sigma/stats_obj.stddev-1., 0., decimal=3,
                                       err_msg='Unexpected result for mean of random numbers!')
        numpy.testing.assert_almost_equal(self.gaussian_sigma**2/stats_obj.variance-1., 0.,
                                       decimal=3,
                                       err_msg='Unexpected result for mean of random numbers!')
        for ind in range(len(stats_obj.percentiles)):
            perc = stats_obj.percentiles[ind]
            val = stats_obj.values[ind]
            expected_perc = 50.*(1.0+
                             math.erf((val-self.gaussian_mean)/(math.sqrt(2.)*self.gaussian_sigma)))
            numpy.testing.assert_almost_equal(expected_perc/perc-1., 0., decimal=2,
                                              err_msg='Unexpected percentile result for randoms!')

    def test_statsystest_basic(self):
        """Various tests of the basic functionality of the StatSysTest class."""

        # Make a load of Gaussian random numbers with the given mean, sigma as a 1d NumPy array.
        numpy.random.seed(self.rand_seed)
        test_vec = self.gaussian_sigma*numpy.random.randn(self.n_points_test) + self.gaussian_mean

        # Run it through the StatSysTest framework and check that the outputs are as expected.
        test_obj = stile.StatSysTest()
        result = test_obj(test_vec)
        self.check_results(result)

        # Do the same with it as a tuple, list, and reshaped into a multi-dimensional array.
        result = test_obj(list(test_vec))
        self.check_results(result)
        result = test_obj(tuple(test_vec))
        self.check_results(result)
        result = test_obj(test_vec.reshape((int(0.5*self.n_points_test), 2)))
        self.check_results(result)

    def test_statsystest_exceptions(self):
        """Make sure StatSysTest throws exceptions at appropriate times."""
        foo = stile.StatSysTest()
        # Input 'array' is something silly, like a float, or None, or a string.
        self.assertRaises(RuntimeError, foo, 3.)
        self.assertRaises(RuntimeError, foo, None)
        self.assertRaises(RuntimeError, foo, 'bar')

        # Called on a catalog without specifying field.
        schema = [("item1", float), ("item2", float)]
        test_arr = numpy.zeros(20, dtype=numpy.dtype(schema))
        self.assertRaises(RuntimeError, foo, test_arr)
        # Called on a catalog that doesn't contain that field.
        self.assertRaises(RuntimeError, foo, test_arr, field='item3')
        # Called on something with NaN without keyword to ignore them.
        x = numpy.empty(10)
        x[3] = numpy.nan
        self.assertRaises(RuntimeError, foo, x)
        # Called on something that has no entries that aren't NaN, with exclusion flag, so we end up
        # with a useful array of length zero.
        x = numpy.empty(10)
        x.fill(numpy.nan)
        self.assertRaises(RuntimeError, foo, x, ignore_bad=True)

    def test_statsystest_catalogs(self):
        """Test the StatSysTest functionality working directly from catalogs."""
        test_len = 10

        schema = [("item1", float), ("item2", float)]
        test_arr = numpy.zeros(test_len, dtype=numpy.dtype(schema))
        test_arr["item1"] = numpy.arange(test_len)
        test_arr["item2"] = 2*numpy.arange(test_len)

        # Make sure it can get the stats for each field appropriately, for a catalog with two
        # fields.
        foo = stile.StatSysTest()
        res1 = foo(test_arr, field='item1')
        res2 = foo(test_arr, field='item2')
        numpy.testing.assert_equal(test_len, res1.N,
                                   err_msg='Wrong length recorded for array')
        numpy.testing.assert_equal(test_len, res2.N,
                                   err_msg='Wrong length recorded for array')
        numpy.testing.assert_almost_equal(0.5*(test_len-1.), res1.mean, decimal=7,
                                          err_msg='Wrong mean for structured array')
        numpy.testing.assert_almost_equal((test_len-1.), res2.mean, decimal=7,
                                          err_msg='Wrong mean for structured array')

        # Make sure that if you set `field` at initialization, it always uses that field for
        # multiple calls, even reverting back after a single call using another field.
        bar = stile.StatSysTest(field='item1')
        res1 = bar(test_arr)
        res2 = bar(test_arr)
        res3 = bar(test_arr, field='item2')
        res4 = bar(test_arr)
        numpy.testing.assert_almost_equal(0.5*(test_len-1.), res1.mean, decimal=7)
        numpy.testing.assert_almost_equal(0.5*(test_len-1.), res2.mean, decimal=7)
        numpy.testing.assert_almost_equal((test_len-1.), res3.mean, decimal=7)
        numpy.testing.assert_almost_equal(0.5*(test_len-1.), res4.mean, decimal=7)

if __name__ == '__main__':
    unittest.main()
