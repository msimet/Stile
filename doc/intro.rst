============
Introduction
============

Stile is the **Systematics Tests In LEnsing package**.  It's designed to run systematics tests on lensing data; it's especially designed for tests that you run on the data itself, such as PSF-galaxy shape correlation functions, as opposed to tests you run against an external data set, although you can do that too.

The tests we've coded up live in the :module:sys_tests module and we call them SysTests (for systematics tests, to distinguish from e.g. unit testing on the software).  They're all objects that you call to run a test; all have a plotting function as well (which doesn't do anything if there's nothing to plot), and possibly other helper functions. We've structured the code so that all the tests have the same kind of call signature: a dataset or datasets in a specific format described below, then any kwargs that control specific operation or plotting.  The tests are defined in rough categories such as correlation functions, whisker plots, etc, and you can create either a flexible generic one or a use-specific version that requires fewer inputs on each run.  For example, you could say:

>>> sys_test = stile.CorrelationFunctionSysTest()

and then you have an object you could call with

>>> corr_func = sys_test('gg', galaxy_data)

to get a shear-shear correlation function of galaxies, or

>>> corr_func = sys_test('ng', lens_data, galaxy_data)

to get tangential shear around the data in ``lens_data``.  Alternately, you could say:

>>> sys_test = stile.CorrelationFunctionSysTest('galaxy shear')

and then that object would *only* perform galaxy shear-shear correlations.  The advantage is that the call signature is simpler:

>>> corr_func = sys_test(galaxy_data)

That way, you don't have to remember which type of correlation function all of these tests are, for example.  The more flexible version may be more useful for exploratory data studies, while the more specific version is best suited for automatic processing of data.  (We have some tools in the works to aid automatic processing, but they are not yet complete.)

The data needs to be in a specific form, which is detailed further in the :data: documentation.

Stile also contains some code to do simple binning of your data.  This is described more in the :doc:`binning` introduction.  We also have a number of helper functions in :doc:`file_io`, :doc:`stile_utils`, and :doc:`treecorr_utils`, all described elsewhere in the documentation.
