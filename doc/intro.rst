============
Introduction
============

Stile is the **Systematics Tests In LEnsing package**.  It's designed to run systematics tests on
lensing data; it's especially designed for tests that you run on the data itself, such as PSF-
galaxy shape correlation functions, as opposed to tests you run against an external data set,
although you can do that too.  The outputs of Stile are the systematic test results themselves;
these can be compared against requirements that are set based on considerations and
calculations that are external to Stile.

The tests we've coded up live in the :mod:`sys_tests` module and we call them SysTests (for
systematics tests, to distinguish from other kinds of tests).  They're all objects that you call to
run a test.  All have a plotting function as well (which doesn't do anything if there's nothing to
plot), and possibly other helper functions. We've structured the code so that all the tests have
the same kind of call signature: a dataset or datasets in a specific format described below, then
any kwargs that control specific operation or plotting.  The tests are defined in rough categories
such as correlation functions, whisker plots, etc, and you can create either a flexible generic one
or a use-specific version that requires fewer inputs on each run.  For example, you could say:

>>> sys_test = stile.CorrelationFunctionSysTest('galaxy shear')

and then that object would *only* perform galaxy shear-shear correlations.  The advantage is that
the call signature is simple, so you don't have to remember very much:

>>> corr_func = sys_test(galaxy_data)

Alternately, you could say:

>>> sys_test = stile.CorrelationFunctionSysTest()

and then you have an object you can call to make lots of different correlations.  For example,

>>> corr_func = sys_test('gg', galaxy_data)

returns a shear-shear autocorrelation function of galaxies, or

>>> corr_func = sys_test('ng', lens_data, galaxy_data)

returns tangential shear around the data in ``lens_data``.  This more flexible version may be more
useful for exploratory data studies, while the more specific version is best suited for automatic
processing of data.  (We have some tools in the works to aid automatic processing, but they are not
yet complete.)

The data needs to be in a specific form, which is detailed further in the :doc:`data` documentation.

Stile also contains some code to do simple binning of your data.  This is described more in
the :doc:`binning` introduction.  We also have a number of helper functions
in :doc:`file_io`, :doc:`stile_utils`, and :doc:`treecorr_utils`, all described elsewhere in the
documentation.

