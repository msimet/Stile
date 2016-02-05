Stile is the Systematics Tests In LEnsing package.  It's designed to run systematics tests on lensing data; it's especially designed for tests that you run on the data itself, such as PSF-galaxy shape correlation functions, as opposed to tests you run against an external data set, although you can do that too.

The tests we've coded up live in the :module:sys_tests module and we call them SysTests (for systematics tests, to distinguish from e.g. unit testing on the software).  They're all objects that you call to run a test; all have a plotting function as well (which doesn't do anything if there's nothing to plot), and possibly other helper functions. We've structured the code so that all the tests have the same kind of call signature: a dataset or datasets in a specific format described below, then any kwargs that control specific operation or plotting.  The tests are defined in rough categories such as correlation functions, whisker plots, etc, and you can create either a flexible generic one or a use-specific version that requires fewer inputs on each run.  For example, you could say:

>>> sys_test = stile.CorrelationFunctionSysTest()

and then you have an object you could call with

>> corr_func = sys_test('gg', galaxy_data)

to get a shear-shear correlation function of galaxies, or 

>> corr_func = sys_test('ng', lens_data, galaxy_data)

to get tangential shear around the data in ``lens_data``.  Alternately, you could say:

>> sys_test = stile.CorrelationFunctionSysTest('galaxy shear')

and then that object would *only* perform galaxy shear-shear correlations, but the call signature is simply

>> corr_func = sys_test(galaxy_data)

The former may be easier for exploratory data studies, while the latter is best suited for automatic processing of data.  (We have some tools in the works to aid automatic processing, but they are not yet complete.)

Stile also contains some code to do simple binning of your data.  This is described more in the :module:`binning` module.  We also have a number of helper functions in :module:`file_io`, :module:`stile_utils`, and :module:`treecorr_utils`, all described elsewhere in the documentation.

DATA FORMAT

Stile requires that data be in a format that can be indexed by a column name: that is, you should be able to do

>>> ra = data['ra']

to get the right ascension of your data.  There is a standard set of Stile column names:

TODO

For some tests, a dict would be sufficient; however, we generally assume that the data is in a contiguous array so we can do

>>> masked_data = data[mask]

which does not work on a dict.  The data types that can handle both masking and column calls via names are: the FITS catalogs of ``pyfits`` or ``astropy``; NumPy formatted arrays; and NumPy record arrays.  The difference between a formatted array and a recarray is that a recarray has an extra layer of Python code between the xxxindexing and the underlying NumPy code, which could cause some slowdown if you're doing a large number of xxxindexing calls.  On the other hand, they're a lot easier to make.

To make a recarray from vectors ``ra``, ``dec``, ``g1``, ``g2``, and ``w``:

>>>

To make a formatted array from an existing array, we have a Stile helper function called :func:`stile.FormatArray`.  From a data array ``arr`` with ``ra`` in the 0th column, ``dec`` in the 1st, ``g1`` in the 2nd, ``g2`` in the 3rd, and ``w`` in the 4th, you can do either:

>>> stile.FormatArray(arr, fields=['ra', 'dec', 'g1', 'g2', 'w'])

or

>>> stile.FormatArray(arr, fields={'ra': 0, 'dec': 1, 'g1': 2, 'g2': 3, 'w': 4})

The dict form doesn't need to give names to all the columns, but the list form does.  The dict form can also have strings as the values if you're rewriting column names from an existing formatted array/recarray/FITS catalog.

Once you have this array, it can be reused for all tests on that data.