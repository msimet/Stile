import numpy
import sys
import time
import math
try:
    import stile
except ImportError:
    sys.path.append('..')
    import stile

def funcname():
    import inspect
    return inspect.stack()[1][3]

rand_seed = 314159 # random seed for tests
n_points_test = 1000000 # number of random points to use for unit tests of stats module
gaussian_mean = 27.0 # Mean value for the Gaussian from which to draw random points
gaussian_sigma = 4.0 # Sigma value for the Gaussian from which to draw random points

def test_statsystest_basic():
    """Various tests of the basic functionality of the StatSysTest class."""
    t1 = time.time()

    # Make a load of Gaussian random numbers with the given mean, sigma as a 1d NumPy array.
    numpy.random.seed(rand_seed)
    test_vec = gaussian_sigma*numpy.random.randn(n_points_test) + gaussian_mean

    # Run it through the StatSysTest framework and check that the outputs are as expected.
    test_obj = stile.sys_tests.StatSysTest()
    result =  test_obj(test_vec)
    numpy.testing.assert_equal(n_points_test, result.N)
    numpy.testing.assert_almost_equal(gaussian_mean/result.mean-1., 0., decimal=3,
                                   err_msg='Unexpected result for mean of random numbers!')
    numpy.testing.assert_almost_equal(gaussian_mean/result.median-1., 0., decimal=3,
                                   err_msg='Unexpected result for median of random numbers!')
    numpy.testing.assert_almost_equal(gaussian_sigma/result.stddev-1., 0., decimal=3,
                                   err_msg='Unexpected result for mean of random numbers!')
    numpy.testing.assert_almost_equal(gaussian_sigma**2/result.variance-1., 0., decimal=3,
                                   err_msg='Unexpected result for mean of random numbers!')
    for ind in range(len(result.percentiles)):
        perc = result.percentiles[ind]
        val = result.values[ind]
        expected_perc = 50.*(1.0 + math.erf((val-gaussian_mean)/(math.sqrt(2.)*gaussian_sigma)))
        numpy.testing.assert_almost_equal(expected_perc/perc-1., 0., decimal=2,
                                          err_msg='Unexpected percentile result for randoms!')

    # Do the same with it as a tuple, list, and reshaped into a multi-dimensional array

    t2 = time.time()
    print 'time for %s = %.2f'%(funcname(),t2-t1)


if __name__ =='__main__':
    test_statsystest_basic()
    #test_statsystest_exceptions()
    #test_statsystest_catalogs()
    #test_statsystest_badvalues()
    #test_statsystest_percentiles()
