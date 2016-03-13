Systematics tests overview
==========================

.. toctree::
  :maxdepth: 2

Basic structure
---------------

:class:`SysTests <stile.sys_tests.SysTest>` are the name within Stile for the systematics tests to be run on the data.  Individual tests are objects.  You create them using a function type plus, in some cases, a specific name:

>>> stat_test = stile.StatSysTest()
>>> rho1_test = stile.CorrelationFunctionSysTest('Rho1')

Then those objects can be called with some data to get the result:

>>> rho1_result = rho1_test(star_data)

And most objects have a ``.plot()`` method to then plot this data, which you call using the result as the argument:

>>> rho1_plot = rho1_test.plot(rho1_result)
>>> rho1_plot.savefig('rho1.png')

All of these calls may have allowable kwargs--see the documentation for the functions for more information.

To figure out which specific types of each test are available, you can look at the docstrings:

>>> help(stile.WhiskerPlotSysTest)
WhiskerPlotSysTest(type=None)
    Initialize an instance of a :class:`BaseWhiskerPlotSysTest` class, based on the ``type`` kwarg given.
    Options are:
        - **Star**: whisker plot of shapes of PSF stars
        - **PSF**: whisker plot of PSF shapes at the location of PSF stars
        - **Residual**: whisker plot of (star shape-PSF shape)
        - **None**: an empty :class:`BaseWhiskerPlotSysTest` class instance, which can be used for multiple types
          of whisker plots.  See the documentation for :class:`BaseWhiskerPlotSysTest` (especially the method
          :func:`whiskerPlot`) for more details.  Note that this type has a different call signature than
          the other methods and that it lacks many of the convenience variables the other
          WhiskerPlots have, such as self.objects_list and self.required_quantities.

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

You can also see the required fields of data (as described in the :doc:`data` documentation) using the
``.required_quantities`` attribute of the SysTest object:

>>> print rho1_test.required_quantities
[('ra', 'dec', 'g1', 'g2', 'psf_g1', 'psf_g2', 'w')]

The first tuple correponds to the first list item from ``.objects_list``, the second tuple to the second list item, etc.  If you want to know more about that field name,
the ``stile.fieldNames`` dict will explain:

>>> print stile.fieldNames['psf_g1']
the g1 of the psf at the location of this object
>>> print stile.fieldNames['w']
the weight to apply per object

We detail the correlation function types (:class:`CorrelationFunctionSysTest`, :class:`WhiskerPlotSysTest`, :class:`ScatterPlotSysTest`, :class:`HistogramSysTest`, and :class:`StatSysTest`) below.

Correlation functions
---------------------

Correlation functions are child classes of :class:`BaseCorrelationFunctionSysTest`; you use the function :func:`CorrelationFunctionSysTest` to create them.  They're all wrappers for
Mike Jarvis's `TreeCorr <https://github.com/rmjarvis/TreeCorr/>`_ code, so you'll need to have that installed to use them.  The predefined types are:

- **GalaxyShear**: tangential and cross shear of galaxies around lenses (point-shear)
- **BrightStarShear**: tangential and cross shear of galaxies around bright stars (point-shear)
- **StarXGalaxyDensity**: number density of galaxies around stars (point-point)
- **StarXGalaxyShear**: shear-shear cross correlation of galaxies and stars (shear-shear)
- **StarXStarShear**: autocorrelation of the shapes of stars (shear-shear)
- **StarXStarSize**: autocorrelation of the size residuals for stars, relative to PSF sizes (scalar-scalar)
- **GalaxyDensityCorrelation**: position autocorrelation of galaxies (point-point)
- **StarDensityCorrelation**: position autocorrelation of stars (point-point)
- **Rho1**: rho1 statistics (autocorrelation of residual star shapes)  (shear-shear)
- **None**: creates a more flexible but less automatic class.  Instead of the usual call signature,
  which just takes one or more data sets, the first argument to an object created this way must
  be a TreeCorr correlation function type (such as ``gg`` or ``ng``).  This object also won't
  have the ``.objects_list`` and ``.required_quantities`` described above.
  
Calling these objects returns a formatted NumPy array of the correlation functions.  The plotting
method returns a ``matplotlib.figure.Figure`` instance that can be saved by ``.savefig`` or further
altered if you like.  
  
The estimators are different depending on the type.  Point-point estimates by default use
the Landy-Szalay estimator:

.. math::
        \xi = (DD-2DR+RR)/RR
        
and must include a random catalog.

All shears in the following descriptions are the complex form in the frame aligned with the
vector between the two points, that is, :math:`g = \gamma_t + i\gamma_x`. 

Point-shear estimates are equivalent to average tangential shear, returned as real :math:`\langle \gamma_t \rangle`
and imaginary :math:`\langle \gamma_x \rangle`.  If random catalogs are given, they are used as random lenses, and
:math:`\langle \gamma_t \rangle-\langle \gamma_{t, {\rm rand}} \rangle` is returned instead.

Shear-shear correlation functions are :math:`\xi_+` and :math:`\xi_-`.  :math:`\xi_+` is, nominally, *g1* times *g2\**, and is
given as both the real and imaginary components.  :math:`\xi_{+,im}` should be consistent with 0 to within
noise. :math:`\xi_-` on the other hand is *g1* times *g2* (not complex conjugate).  Similarly, :math:`\xi_{-,im}` should
be 0.  (Note that *g1* and *g2* here are two *complex* shears :math:`g_{1t} + ig_{1x}` and :math:`g_{2t} + ig_{2x}` from the
two catalogs, not the two components of a single shear in sky coordinates or chip frame.)

Point-scalar (point-kappa) estimates are equivalent to <scalar>; scalar-shear estimates are
equivalent to <scalar*Re(shear)> with a corresponding imaginary case; scalar-scalar estimates
are <scalar1*scalar2>.  Random catalogs result in compensated estimators as in the point-shear
case.    
 
Whisker plots
-------------

Whisker plots are child classes of :class:`BaseWhiskerPlotSysTest`.  They're the standard weak lensing whisker plots: visualizations of the shear field using headless arrows.  
The predefined types are:

- **Star**: whisker plot of shapes of PSF stars
- **PSF**: whisker plot of PSF shapes at the location of PSF stars
- **Residual**: whisker plot of (star shape-PSF shape)
- **None**: an empty :class:`BaseWhiskerPlotSysTest` class instance, which can be used for multiple types
  of whisker plots.  See the documentation for :class:`BaseWhiskerPlotSysTest` (especially the method
  :func:`whiskerPlot`) for more details.  Note that this type has a different call signature than
  the other methods and that it lacks many of the convenience variables the other
  WhiskerPlots have, such as self.objects_list and self.required_quantities.

Calls to child classes of :class:`BaseWhiskerPlotSysTest` return ``matplotlib.figure.Figure`` instances and can be saved directly as images, but a ``.plot()`` attribute is included
(that just returns the figure passed to it) so the plotting interface is the same as other :class:`SysTests <SysTest>`.

Scatter plots
-------------

Scatter plots are child classes of :class:`BaseScatterPlotSysTest`.  They generate scatter plots of the data plus a linear regression fit through the scattered points.  There is also an optional comparison trendline at :math:`y=0` or :math:`x=y` (or a user-defined function).  The predefined types are:

- **StarVsPSFG1**: star vs PSF g1
- **StarVsPSFG2**: star vs PSF g2
- **StarVsPSFSigma**: star vs PSF sigma
- **ResidualVsPSFG1**: (star - PSF) g1 vs PSF g1
- **ResidualVsPSFG2**: (star - PSF) g1 vs PSF g2
- **ResidualVsPSFSigma**: (star - PSF) g1 vs PSF sigma
- **ResidualSigmaVsPSFMag**: (star - PSF)/PSF sigma vs PSF magnitude
- **None**: an empty :class:`BaseScatterPlotSysTest` class instance, which can be used for multiple types
  of scatter plots.  See the documentation for :class:`BaseScatterPlotSysTest` (especially the method
  :func:`scatterPlot <BaseScatterPlotSysTest.scatterPlot>`) for more details.  Note that this type has a different call signature than
  the other methods and that it lacks many of the convenience variables the other
  ScatterPlots have, such as self.objects_list and self.required_quantities.

Calls to child classes of :class:`BaseScatterPlotSysTest` return ``matplotlib.figure.Figure`` instances and can be saved directly as images, but a ``.plot()`` attribute is included
(that just returns the figure passed to it) so the plotting interface is the same as other :class:`SysTests <SysTest>`.  

You can turn the trendlines on and off with the kwarg ``reference_line='zero'`` (at :math:`y=0`), ``reference_line='one-to-one'`` (at :math:`x=y`), or ``reference_line=function`` (to
overplot the line given by ``function(x)``, which must return a 1D NumPy array given a vector ``x``).  Results binned on the basis of a data field labeled ``'CCD'`` can be generated
via the kwarg ``per_ccd_stat=True``.

Histograms
----------

Basic statistical quantities
----------------------------

The :class:`StatSysTest` systematics tests are designed to measure basic statistical quantities on a vector of data or on a field from a data array.  To operate on a data vector:

>>> stat_sys_test_vector = stile.StatSysTest()

To operate on a field,

>>> stat_sys_test_field = stile.StatSysTest('g1')

or pass the field at runtime:

>>> results = stat_sys_test_vector(data, field='g1')

Currently, a :class:`StatSysTest` will compute min, max, median, median absolute deviation (MAD), mean, standard deviation, variance, and number of objects, and will additionally compute the skew and kurtosis if ``scipy.stats`` can be imported.  It will also compute percentiles in the data: by default, 1, 2, and 3 sigma values around the median, but this can be changed with the ``percentiles`` kwarg.

:class:`StatSysTests <StatSysTest>` return a :class:`stile.Stats` object, with all of the above quantities available as attributes (with names given above, except for mad [MAD], stddev [standard deviation], and N [number of points]). You can call a ``.plot()`` method, but it won't do anything.  Printing the object results in a nicely-formatted summary; results will be automatically printed if you pass the kwarg ``verbose=True``.
