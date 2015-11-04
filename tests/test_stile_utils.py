import numpy
import unittest
try:
    import stile
except ImportError:
    import sys
    sys.path.append('..')
    import stile

class TestStileUtils(unittest.TestCase):

    def test_FormatArray(self):
        """Test formatted array routines and behavior."""
        data_raw = [(1, 'hello', 2.0), (3, 'boo', 5.0)]
        data0 = numpy.array(data_raw)
        data1 = numpy.array(data_raw, dtype='l, S5, d')
        old_dnames = data1.dtype.names
        # Check that FormatArray(stuff) still contains the right data
        result = stile.FormatArray(data0)
        numpy.testing.assert_equal(result, data1.astype(result.dtype))
        # And check that FormatArray(formatted stuff) keeps its formatting
        result = stile.FormatArray(data1)
        numpy.testing.assert_equal(result, data1)
        # Now check that the field names are added correctly either by dict or by list
        result = stile.FormatArray(data0, fields=['one', 'two', 'three'])
        data1.dtype.names = result.dtype.names
        numpy.testing.assert_equal(result, data1.astype(result.dtype))
        numpy.testing.assert_equal(result.dtype.names, ['one', 'two', 'three'])
        result2 = stile.FormatArray(data0, fields={'one': 0, 'two': 1, 'three': 2})
        numpy.testing.assert_equal(result, result2)
        data1.dtype.names = old_dnames  # reset to old names
        # Now check that fields names are added correctly to formatted arrays
        result = stile.FormatArray(data1, fields=['one', 'two', 'three'])
        data1.dtype.names = result.dtype.names
        numpy.testing.assert_equal(result, data1.astype(result.dtype))
        numpy.testing.assert_equal(result.dtype.names, ['one', 'two', 'three'])
        data1.dtype.names = old_dnames
        result2 = stile.FormatArray(data1, fields={'one': 0, 'two': 1, 'three': 2})
        numpy.testing.assert_equal(result, result2)
        # And one quick check for non-NumPy arrays, ie, assume a 1d array is a *row* not a *field*
        # and that everything else works
        numpy.testing.assert_equal(stile.FormatArray([1, 2]), numpy.array([(1, 2)], dtype='l, l'))


if __name__ == '__main__':
    unittest.main()
