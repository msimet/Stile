Systematics tests overview
==========================

.. toctree::
  :maxdepth: 2

Basic structure
---------------

:class:`SysTests <stile.sys_tests.SysTest>` are the name within Stile for the systematics tests to
be run on the data.  Individual tests are objects.  You create them using a function type plus, in
some cases, a specific name:

>>> stat_test = stile.StatSysTest()
>>> rho1_test = stile.CorrelationFunctionSysTest('Rho1')

Then those objects can be called with some data to get the result:

>>> rho1_result = rho1_test(star_data)

And most objects have a :func:`.plot` method to then plot this data, which you call using the
result as the argument:

>>> rho1_plot = rho1_test.plot(rho1_result)
>>> rho1_plot.savefig('rho1.png')

All of these calls may have allowable kwargs--see the documentation for the functions for more
information.

To figure out which specific types of each test are available, you can look at the docstrings:

>>> help(stile.WhiskerPlotSysTest)
WhiskerPlotSysTest(type=None)
    Initialize an instance of a :class:`BaseWhiskerPlotSysTest` class, based on the
    ``type`` kwarg given.  Options are:
        - **Star**: whisker plot of shapes of PSF stars
        - **PSF**: whisker plot of PSF shapes at the location of PSF stars
        - **Residual**: whisker plot of (star shape-PSF shape)
        - **None**: an empty :class:`BaseWhiskerPlotSysTest` class instance, which
          can be used for multiple types of whisker plots.  See the documentation for
          :class:`BaseWhiskerPlotSysTest` (especially the method :func:`whiskerPlot`)
          for more details.  Note that this type has a different call signature than
          the other methods and that it lacks many of the convenience variables the
          other WhiskerPlots have, such as self.objects_list and 
          self.required_quantities.

or, alternately, head over to the :doc:`sys_tests` documentation page, which has nicer formatting.

To figure out what kind of data each test needs, you can check the documentation.  You can also
use attributes of the object itself, if you used a specific name in the object creation
(as in the creation of ``rho1_test`` but not ``stat_test``). To figure out which data set(s)
to use, look at the ``.objects_list`` attribute:

>>> print rho1_test.objects_list
['star PSF']

The ``stile.objectNames`` dict will tell you more about that kind of data:

>>> print stile.objectNames['star PSF']
stars used in PSF determination

You can also see the required fields of data (as described in the :doc:`data` documentation) using
the ``.required_quantities`` attribute of the SysTest object:

>>> print rho1_test.required_quantities
[('ra', 'dec', 'g1', 'g2', 'psf_g1', 'psf_g2', 'w')]

The first tuple correponds to the first list item from ``.objects_list``, the second tuple to the
second list item, etc.  If you want to know more about that field name,
the ``stile.fieldNames`` dict will explain:

>>> print stile.fieldNames['psf_g1']
the g1 of the psf at the location of this object
>>> print stile.fieldNames['w']
the weight to apply per object

We detail the correlation function types (:func:`stile.CorrelationFunctionSysTest
<stile.sys_tests.CorrelationFunctionSysTest>`, :func:`stile.WhiskerPlotSysTest
<stile.sys_tests.WhiskerPlotSysTest>`, :func:`stile.ScatterPlotSysTest
<stile.sys_tests.ScatterPlotSysTest>`, :class:`stile.HistogramSysTest
<stile.sys_tests.HistogramSysTest>`, and :class:`StatSysTest <stile.sys_tests.StatSysTest>`) below.

Correlation functions
---------------------

Correlation functions are child classes
of :class:`stile.sys_tests.BaseCorrelationFunctionSysTest`. You use the
function :func:`stile.CorrelationFunctionSysTest <stile.sys_tests.CorrelationFunctionSysTest>` to
create them.  They're all wrappers for
Mike Jarvis's `TreeCorr <https://github.com/rmjarvis/TreeCorr/>`_ code, so you'll need to have
that installed to use them.  The predefined types are:

- **GalaxyShear**: tangential and cross shear of galaxies around lenses (point-shear)
- **BrightStarShear**: tangential and cross shear of galaxies around bright stars (point-shear)
- **StarXGalaxyDensity**: number density of galaxies around stars (point-point)
- **StarXGalaxyShear**: shear-shear cross correlation of galaxies and stars (shear-shear)
- **StarXStarShear**: autocorrelation of the shapes of stars (shear-shear)
- **StarXStarSize**: autocorrelation of the size residuals for stars, relative to PSF sizes
  (scalar-scalar)
- **GalaxyDensityCorrelation**: position autocorrelation of galaxies (point-point)
- **StarDensityCorrelation**: position autocorrelation of stars (point-point)
- **Rho1**: rho1 statistics (autocorrelation of residual star shapes)  (shear-shear)
- **None**: creates a more flexible but less automatic class.  Instead of the usual call signature,
  which just takes one or more data sets, the first argument to an object created this way must
  be a TreeCorr correlation function type (such as ``gg`` or ``ng``).  This object also won't
  have the ``.objects_list`` and ``.required_quantities`` described above.

Calling these objects returns a formatted NumPy array of the correlation functions.  If the order
of the data sets matters--eg for **GalaxyShear**--any data set that only needs positions should go
first, and any data set with a scalar value you're using should go last.  For example:

>>> bright_star_shear = stile.CorrelationFunctionSysTest('BrightStarShear')
>>> results = bright_star_shear(bright_star_data, galaxy_data)

The plotting method returns a :class:`matplotlib.figure.Figure` instance that can be saved
by :func:`.savefig` or further altered if you like.

The estimators are different depending on the type.  Point-point estimates by default use
the Landy-Szalay estimator:

.. math::
        \xi = (DD-2DR+RR)/RR

and must include a random catalog.

All shears in the following descriptions are the complex form in the frame aligned with the
vector between the two points, that is, :math:`g = \gamma_t + i\gamma_x`.

Point-shear estimates are equivalent to average tangential shear, returned as real
:math:`\langle \gamma_t \rangle` and imaginary :math:`\langle \gamma_x \rangle`.  If random
catalogs are given, they are used as random lenses, and
:math:`\langle \gamma_t \rangle-\langle \gamma_{t, {\rm rand}} \rangle` is returned instead.

Shear-shear correlation functions are :math:`\xi_+` and :math:`\xi_-`.  They are functions of the
spatial correlation of the shears and their complex conjugates in sky coordinates; see e.g.
`Jarvis et al. 2016 <http://adsabs.harvard.edu/abs/2016MNRAS.460.2245J>`_ for complete definitions
of :math:`\xi_+` and :math:`\xi_-`.  The :math:`\xi` functions should be completely real (within 
noise), although the results will include the imaginary portions as a cross-check.

Point-scalar (point-kappa) estimates are equivalent to ``<scalar>``. Scalar-shear estimates are
equivalent to ``<scalar*Re(shear)>`` with a corresponding imaginary case ``<scalar*Im(shear)>``,
corresponding to the correlation between the scalar value of the first sample at that point and the
tangential shear (real) or the shear at 45 degrees from the tangent (imaginary) from the
second sample. Scalar-scalar estimates are ``<scalar1*scalar2>``.  Random catalogs result in
compensated estimators as in the point-shear case.

Whisker plots
-------------

Whisker plots are child classes of :class:`stile.sys_tests.BaseWhiskerPlotSysTest`.  They're the
standard weak lensing whisker plots: visualizations of the shear field using headless arrows.
The predefined types are:

- **Star**: whisker plot of shapes of PSF stars
- **PSF**: whisker plot of PSF shapes at the location of PSF stars
- **Residual**: whisker plot of (star shape-PSF shape)
- **None**: an empty :class:`stile.sys_tests.BaseWhiskerPlotSysTest` class instance, which can be
  used for multiple types of whisker plots.  See the documentation for
  :class:`stile.sys_tests.BaseWhiskerPlotSysTest` (especially the method
  :func:`whiskerPlot <stile.sys_tests.BaseWhiskerPlotSysTest.whiskerPlot>`) for more details.
  Note that this type has a different call signature than
  the other methods and that it lacks many of the convenience variables the other
  WhiskerPlots have, such as self.objects_list and self.required_quantities.

Calls to child classes of :class:`stile.sys_tests.BaseWhiskerPlotSysTest`
return :class:`matplotlib.figure.Figure` instances and can be saved directly as images, but
a :func:`.plot` method is included (that just returns the figure passed to it) so the plotting
interface is the same as other :class:`SysTests <stile.sys_tests.SysTest>`.

Scatter plots
-------------

Scatter plots are child classes of :class:`stile.sys_tests.BaseScatterPlotSysTest`.  They generate
scatter plots of the data plus a linear regression fit through the scattered points.  There is
also an optional comparison trendline at :math:`y=0` or :math:`x=y` (or a user-defined function).
The predefined types are:

- **StarVsPSFG1**: star vs PSF g1
- **StarVsPSFG2**: star vs PSF g2
- **StarVsPSFSigma**: star vs PSF sigma
- **ResidualVsPSFG1**: (star - PSF) g1 vs PSF g1
- **ResidualVsPSFG2**: (star - PSF) g2 vs PSF g2
- **ResidualVsPSFSigma**: (star - PSF) sigma vs PSF sigma
- **ResidualSigmaVsPSFMag**: (star - PSF)/PSF sigma vs PSF magnitude
- **None**: an empty :class:`stile.sys_tests.BaseScatterPlotSysTest` class instance, which can be
  used for multiple types of scatter plots.  See the documentation for
  :class:`stile.sys_tests.BaseScatterPlotSysTest` (especially the method
  :func:`scatterPlot <stile.sys_tests.BaseScatterPlotSysTest.scatterPlot>`) for more details.
  Note that this type has a different call signature than
  the other methods and that it lacks many of the convenience variables the other
  ScatterPlots have, such as self.objects_list and self.required_quantities.

Calls to child classes of :class:`stile.sys_tests.BaseScatterPlotSysTest`
return :class:`matplotlib.figure.Figure` instances and can be saved directly as images, but
a :func:`.plot` method is included (that just returns the figure passed to it) so the plotting
interface is the same as other :class:`SysTests <stile.sys_tests.SysTest>`.

You can turn the trendlines on and off with the kwarg ``reference_line='zero'`` (at :math:`y=0`),
``reference_line='one-to-one'`` (at :math:`x=y`), or ``reference_line=function`` (to
overplot the line given by ``function(x)``, which must return a 1D NumPy array given a vector
``x``).  Results binned on the basis of a data field labeled ``'CCD'`` can be generated
via the kwarg ``per_ccd_stat=True``.

Histograms
----------

The :class:`HistogramSysTest <stile.sys_tests.HistogramSysTest>` systematics tests generate 
histograms, optionally using optimized bin widths.  You can pass a one-dimensional dataset to
display its histogram; a list of one-dimensional datasets to generate histograms on the same axes;
a formatted array (see :doc:`data`) and a field to generate a histogram for that field; or a list of
formatted arrays and a field or list of fields to generate multiple histograms on the same axes.

To operate on a data vector:

>>> histogram_sys_test_vector = stile.HistogramSysTest()

To operate on a formatted array,

>>> histogram_sys_test_field = stile.HistogramSysTest('g1')

or pass the field at runtime:

>>> results = histogram_sys_test_vector(data, field='g1')

To operate on multiple data vectors:

>>> results = histogram_sys_test_vector([data, data2])

or on multiple formatted arrays:

>>> results = histogram_sys_test_field([data, data2])

or

>>> results = histogram_sys_test_vector([data, data2], field='g1').

To plot multiple fields from the same array, you'll need to pass the same array multiple times, due
to implementation details:

>>> results = histogram_sys_test_vector([data, data], field=['g1', 'g2'])

or of course you can just access those fields yourself:

>>> results = histogram_sys_test_vector([data['g1'], data['g2']]).

By default, the :class:`HistogramSysTest <stile.sys_tests.HistogramSysTest>` makes a histogram with
50 bins.  You probably want to choose different binning.  You can have Stile pick the binning for you
based on two different algorithms:

>>> # Scott's rule
>>> scott_rule_hist = stile.HistogramSysTest(binning_style='scott')  
>>> # Freedman-Diaconis rule
>>> freedman_rule_hist = stile.HistogramSysTest(binning_style='freedman')

Scott's rule chooses bins based on the sample size and the sample standard deviation, while the
Freedman-Diaconis rule uses the interquartile range instead of the sample standard deviation,
so it is less sensitive to outliers.

You can also just set your own number of bins.

>>> ten_bins_hist = stile.HistogramSysTest(nbins=10)

These plots are very configurable--you can add vertical lines, colors, text, and you can also change 
the appearance of the histogram itself.  See the documentation for :class:`HistogramSysTest 
<stile.sys_tests.HistogramSysTest>` for more information.

Basic statistical quantities
----------------------------

The :class:`StatSysTest <stile.sys_tests.StatSysTest>` systematics tests are designed to measure
basic statistical quantities on a vector of data or on a field from a data array.  To operate on a
data vector:

>>> stat_sys_test_vector = stile.StatSysTest()

To operate on a formatted array,

>>> stat_sys_test_field = stile.StatSysTest('g1')

or pass the field at runtime:

>>> results = stat_sys_test_vector(data, field='g1')

Currently, a :class:`stile.StatSysTest <stile.sys_tests.StatSysTest>` will compute min, max,
median, median absolute deviation (MAD), mean, standard deviation, variance, and number of
objects, and will additionally compute the skew and kurtosis if ``scipy.stats`` can be imported.
It will also compute percentiles in the data: by default, the percentiles correspond to 1, 2,
and 3 sigma values around the median for a Gaussian function, but this can be changed with the
``percentiles`` kwarg.

:class:`StatSysTests <stile.sys_tests.StatSysTest>` return a :class:`stile.Stats
<stile.stile_utils.Stats>` object, with all of the above quantities available as attributes (with
names given above, except for mad [MAD], stddev [standard deviation], and N [number of points]).
You can call a :func:`.plot` method, but it won't do anything.  Printing the object results in a
nicely-formatted summary.  Results will be automatically printed if you pass the kwarg
``verbose=True``.

