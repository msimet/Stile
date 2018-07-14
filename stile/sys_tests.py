"""
sys_tests.py: Contains the class definitions of the Stile systematics tests.
"""

"""
This file contains some code from the AstroML package (http://github.com/astroML/astroML).
For that code:

Copyright (c) 2012-2013, Jacob Vanderplas
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted
provided that the following conditions are met:

    Redistributions of source code must retain the above copyright notice, this list of conditions
    and the following disclaimer.
    Redistributions in binary form must reproduce the above copyright notice, this list of
    conditions and the following disclaimer in the documentation and/or other materials provided
    with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF
THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""


import numpy
import stile
from . import stile_utils
try:
    import treecorr
    from treecorr.corr2 import corr2_valid_params
    has_treecorr = True
except ImportError:
    has_treecorr = False
    import warnings
    warnings.warn("treecorr package cannot be imported. You may "+
                  "wish to install it if you would like to use the correlation functions within "+
                  "Stile.")

try:
    import matplotlib
    # We should decide which backend to use (this line allows running matplotlib even on sessions
    # without properly defined displays, eg through PBS)
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    has_matplotlib = True
except ImportError:
    has_matplotlib = False

class PlotNone(object):
    """
    An empty class with a :func:`.savefig` method that does nothing, for code automation purposes.
    """
    def savefig(self, filename):
        pass

class SysTest(object):
    """
    A SysTest is a lensing systematics test of some sort.  It should define the following
    attributes:

        * ``short_name``: a string that can be used in filenames to denote this systematics test
        * ``long_name``: a string to denote this systematics test within program text outputs
        * ``objects_list``: a list of objects that the test should operate on.  We expect these
          objects to be from the list::

                ['galaxy', 'star',    # all such objects
                 'galaxy lens',       # only galaxies to be used as lenses in galaxy-galaxy lensing
                 'star PSF',          # stars used in PSF determination
                 'star bright',       # especially bright stars
                 'galaxy random',     # random catalogs corresponding to the
                 'star random']       # 'galaxy' or 'star' samples.

        * ``required_quantities``: a list of tuples.  Each tuple is the list of fields/quantities
          that should be given for the corresponding object from the objects_list.  We expect the
          quantities to be from the list::

                ['ra', 'dec',         # Position on the sky
                 'x', 'y',            # Position in CCD/detector coords (or any flat projection)
                 'g1', 'g1_err',      # Two components of shear and their errors
                 'g2', 'g2_err',
                 'sigma', 'sigma_err',# Object size and its error
                 'w',                 # Per-object weight
                 'psf_g1', 'psf_g2',  # PSF shear and size at the object location
                 'psf_sigma']

    It should define the following methods:
        * ``__call__(self, data, data2=None, random=None, random2=None, config=None, **kwargs)``:
            Run a test on a set of data, or a test involving two data sets ``data`` and ``data2``,
            with optional corresponding randoms ``random`` and ``random2``.  Keyword args can be in
            a dict passed to ``config`` or as explicit kwargs.  Explicit kwargs should override
            ``config`` arguments.  Additional args may be required by the base versions of different
            tests, but the specific implementations of those base versions should always have this
            call signature.
    """
    short_name = ''
    long_name = ''
    def __init__(self):
        pass
    def __call__(self):
        raise NotImplementedError()
    def plot(self, results):
        """
        If the results returned from the :func:`__call__` function of this class have
        a :func:`.savefig` method, return that object.  Otherwise, return an object with
        a :func:`.savefig` method that doesn't do anything.  :func:`plot` should be overridden by
        child classes to actually generate plots if desired.
        """
        if hasattr(results, 'savefig'):
            return results
        else:
            return PlotNone()

class PlotDetails(object):
    """
    A container class to hold details about field names, titles for legends, and y-axis labels for
    plots of correlation functions.
    """
    def __init__(self, t_field=None, t_title=None, x_field=None, x_title=None,
                 t_im_field=None, t_im_title=None, x_im_field=None, x_im_title=None,
                 datarandom_t_field=None, datarandom_t_title=None,
                 datarandom_x_field=None, datarandom_x_title=None,
                 sigma_field=None, y_title=None):
        self.t_field = t_field  # Field of t-mode/+-mode shear correlation functions
        self.t_title = t_title  # Legend title for previous line
        self.x_field = x_field  # Field of b-mode/x-mode shear correlation functions
        self.x_title = x_title  # Legend title for previous line
        self.t_im_field = t_im_field  # Imaginary part of t-mode/+-mode
        self.t_im_title = t_im_title  # Legend title for previous line
        self.x_im_field = x_im_field  # Imaginary part of b-mode/x-mode
        self.x_im_title = x_im_title  # Legend title for previous line
        self.datarandom_t_field = datarandom_t_field  # If data or randoms are available separately,
                                                      # this +'d' or +'r' is the t-mode field name
        self.datarandom_t_title = datarandom_t_title  # Legend title for previous line
        self.datarandom_x_field = datarandom_x_field  # As above, for b-mode
        self.datarandom_x_title = datarandom_x_title  # Legend title for previous line
        self.sigma_field = sigma_field  # 1-sigma error bar field
        self.y_title = y_title  # y-axis label

if has_treecorr and treecorr.version < '3.1':
    treecorr_func_dict = {'gg': treecorr.G2Correlation,
                          'm2': treecorr.G2Correlation,
                          'ng': treecorr.NGCorrelation,
                          'nm': treecorr.NGCorrelation,
                          'norm': treecorr.NGCorrelation,
                          'nn': treecorr.N2Correlation,
                          'kk': treecorr.K2Correlation,
                          'nk': treecorr.NKCorrelation,
                          'kg': treecorr.KGCorrelation}
elif has_treecorr:
    treecorr_func_dict = {'gg': treecorr.GGCorrelation,
                          'm2': treecorr.GGCorrelation,
                          'ng': treecorr.NGCorrelation,
                          'nm': treecorr.NGCorrelation,
                          'norm': treecorr.NGCorrelation,
                          'nn': treecorr.NNCorrelation,
                          'kk': treecorr.KKCorrelation,
                          'nk': treecorr.NKCorrelation,
                          'kg': treecorr.KGCorrelation}

def CorrelationFunctionSysTest(type=None):
    """
    Initialize an instance of a :class:`BaseCorrelationFunctionSysTest` type, based on the ``type``
    kwarg  given.  Options are:

        - **GalaxyShear**: tangential and cross shear of ``galaxy`` type objects around
          ``galaxy lens``  type objects
        - **BrightStarShear**: tangential and cross shear of ``galaxy`` type objects around
          ``star bright`` type objects
        - **StarXGalaxyDensity**: number density of ``galaxy`` objects around ``star`` objects
        - **StarXGalaxyShear**: shear-shear cross correlation of ``galaxy`` and ``star`` type
          objects
        - **StarXStarShear**: autocorrelation of the shapes of ``star`` type objects
        - **StarXStarSizeResidual**: autocorrelation of the size residuals for ``star`` type objects
          relative to PSF sizes
        - **GalaxyDensityCorrelation**: position autocorrelation of ``galaxy`` type objects
        - **StarDensityCorrelation**: position autocorrelation of ``star`` type objects
        - **Rho1**: rho1 statistics (autocorrelation of residual star shapes)
        - **Rho2**: rho2 statistics (correlation of star and PSF shapes)
        - **Rho3**: rho3 statistics (autocorrelation of star shapes weighted by the residual size)
        - **Rho4**: rho4 statistics (correlation of residual star shapes weighted by residual size)
        - **Rho5**: rho5 statistics (correlation of star and PSF shapes weighted by the residual size)
        - **None**: an empty BaseCorrelationFunctionSysTest class instance, which can be used for
          multiple types of correlation functions.  See the documentation for
          BaseCorrelationFunctionSysTest for more details.  Note that this type has a
          slightly different call signature than the other methods (with the correlation function
          type given as the first argument) and that it lacks many of the convenience variables the
          other CorrelationFunctions have, such as self.objects_list and self.required_quantities.

    These produce different estimators depending on the type.  Point-point estimates by default use
    the Landy-Szalay estimator:
        xi = (DD-2DR+RR)/RR
    and must include a random catalog.
    
    All shears in the following descriptions are the complex form in the frame aligned with the
    vector between the two points, that is, g = gamma_t + i*gamma_x. 
    
    Point-shear estimates are equivalent to average tangential shear, returned as real <gamma_t>
    and imaginary <gamma_x>.  If random catalogs are given, they are used as random lenses, and
    data*shear-random*shear is returned instead.
    
    Shear-shear correlation functions are xi_+ and xi_-.  Xi_+ is, nominally, g1 times g2*, and is
    given as both the real and imaginary components.  xi_+,im should be consistent with 0 to within
    noise. Xi_- on the other hand is g1 times g2 (not complex conjugate).  Similarly, xi_-,im should
    be 0.  (Note that g1 and g2 here are two *complex* shears g1_t + i*g1_x and g1_t+i*g2_x from the
    two catalogs, not the two components of a single shear in sky coordinates or chip frame.)
    
    Point-scalar (point-kappa) estimates are equivalent to <scalar>; scalar-shear estimates are
    equivalent to <scalar*Re(shear)> with a corresponding imaginary case; scalar-scalar estimates
    are <scalar1*scalar2>.  Random catalogs result in compensated estimators as in the point-shear
    case.  
    
    Aperture mass statistics of various kinds are also available via the
    BaseCorrelationFunctionSysTest class; as we do not implement those for any standard Stile tests,
    interested users are directed to the TreeCorr documentation for further information.

    Users who are writing their own code can of course pass other data types to these functions
    than the ones given (eg sending galaxy data to a StarXStarSize correlation function); we
    include options for A) automatic processing,  B) ease of understanding code, and C) suggesting
    tests that are useful to run.
    """
    if type is None:
        return BaseCorrelationFunctionSysTest()
    elif type=='GalaxyShear':
        return GalaxyShearSysTest()
    elif type=='BrightStarShear':
        return BrightStarShearSysTest()
    elif type=='StarXGalaxyDensity':
        return StarXGalaxyDensitySysTest()
    elif type=='StarXGalaxyShear':
        return StarXGalaxyShearSysTest()
    elif type=='StarXStarShear':
        return StarXStarShearSysTest()
    elif type=='StarXStarSizeResidual':
        return StarXStarSizeResidualSysTest()
    elif type=='GalaxyDensityCorrelation':
        return GalaxyDensityCorrelationSysTest()
    elif type=='StarDensityCorrelation':
        return StarDensityCorrelationSysTest()
    elif type=='Rho1':
        return Rho1SysTest()
    elif type=='Rho2':
        return Rho2SysTest()
    elif type=='Rho3':
        return Rho3SysTest()
    elif type=='Rho4':
        return Rho4SysTest()
    elif type=='Rho5':
        return Rho5SysTest()
    else:
        raise ValueError('Unknown correlation function type %s given to type kwarg'%type)


class BaseCorrelationFunctionSysTest(SysTest):
    """
    A base class for the Stile systematics tests that use correlation functions. This implements the
    class method :func:`getCF()` which runs a TreeCorr correlation function on a given set of data.
    Exact arguments to this method should be created by child classes of this class;
    see the docstring for :func:`getCF` for information on how to write
    further tests using it.
    """
    short_name = 'corrfunc'
    # Set the details (such as field names and titles) for all the possible plots generated by
    # TreeCorr
    plot_details = [
        PlotDetails(t_field='gamT', t_title=r'$\langle \gamma_T \rangle$',
                    x_field='gamX', x_title=r'$\langle \gamma_X \rangle$',
                    datarandom_t_field='gamT_', datarandom_t_title='$\gamma_{T',
                    datarandom_x_field='gamX_', datarandom_x_title='$\gamma_{X',
                    sigma_field='sigma', y_title="$\gamma$"),  # ng
        PlotDetails(t_field='xip', t_title=r'$\xi_+$', x_field='xim', x_title=r'$\xi_-$',
                    t_im_field='xip_im', t_im_title=r'$\xi_{+,im}$',
                    x_im_field='xip_im', x_im_title=r'$\xi_{-,im}$',
                    sigma_field='sigma_xi', y_title=r"$\xi$"),  # gg
        PlotDetails(t_field='kappa', t_title=r'$\langle \kappa \rangle$',
                    datarandom_t_field='kappa_', datarandom_t_title='$kappa_{',
                    sigma_field='sigma', y_title="$\kappa$"),  # nk
        PlotDetails(t_field='xi', t_title=r'$\xi$', sigma_field='sigma_xi', y_title=r"$\xi$"),  # n2 or k2
        PlotDetails(t_field='kgamT', t_title=r'$\langle \kappa \gamma_T\rangle$',
                    x_field='kgamX', x_title=r'$\langle \kappa \gamma_X\rangle$',
                    datarandom_t_field='kgamT_', datarandom_t_title=r'$\kappa \gamma_{T',
                    datarandom_x_field='kgamX_', datarandom_x_title=r'$\kappa \gamma_{X',
                    sigma_field='sigma', y_title="$\kappa\gamma$"),  # kg
        PlotDetails(t_field='Mapsq', t_title=r'$\langle M_{ap}^2 \rangle$',
                    x_field='Mxsq', x_title=r'$\langle M_x^2\rangle$',
                    t_im_field='MMxa', t_im_title=r'$\langle MM_x \rangle(a)$',
                    x_im_field='Mmxb', x_im_title=r'$\langle MM_x \rangle(b)$',
                    sigma_field='sig_map', y_title="$M_{ap}^2$"),  # m2
        PlotDetails(t_field='NMap', t_title=r'$\langle NM_{ap} \rangle$',
                    x_field='NMx', x_title=r'$\langle NM_{x} \rangle$',
                    sigma_field='sig_nmap', y_title="$NM_{ap}$"),  # nm or norm
        # For TreeCorr versions <= 3.1, these had different column names.
        PlotDetails(t_field='<gamT>', t_title=r'$\langle \gamma_T \rangle$',
                    x_field='<gamX>', x_title=r'$\langle \gamma_X \rangle$',
                    datarandom_t_field='gamT_', datarandom_t_title='$\gamma_{T',
                    datarandom_x_field='gamX_', datarandom_x_title='$\gamma_{X',
                    sigma_field='sigma', y_title="$\gamma$"),  # ng
        PlotDetails(t_field='xi+', t_title=r'$\xi_+$', x_field='xi-', x_title=r'$\xi_-$',
                    t_im_field='xi+_im', t_im_title=r'$\xi_{+,im}$',
                    x_im_field='xi-_im', x_im_title=r'$\xi_{-,im}$',
                    sigma_field='sigma_xi', y_title=r"$\xi$"),  # gg
        PlotDetails(t_field='<kappa>', t_title=r'$\langle \kappa \rangle$',
                    datarandom_t_field='kappa_', datarandom_t_title='$kappa_{',
                    sigma_field='sigma', y_title="$\kappa$"),  # nk
        PlotDetails(t_field='xi', t_title=r'$\xi_{\mathrm{re}}$', sigma_field='sigma_xi', y_title=r"$\xi$"),  # k2
        PlotDetails(t_field='<kgamT>', t_title=r'$\langle \kappa \gamma_T\rangle$',
                    x_field='<kgamX>', x_title=r'$\langle \kappa \gamma_X\rangle$',
                    datarandom_t_field='kgamT_', datarandom_t_title=r'$\kappa \gamma_{T',
                    datarandom_x_field='kgamX_', datarandom_x_title=r'$\kappa \gamma_{X',
                    sigma_field='sigma', y_title="$\kappa\gamma$"),  # kg
        PlotDetails(t_field='<Map^2>', t_title=r'$\langle M_{ap}^2 \rangle$',
                    x_field='<Mx^2>', x_title=r'$\langle M_x^2\rangle$',
                    t_im_field='<MMx>(a)', t_im_title=r'$\langle MM_x \rangle(a)$',
                    x_im_field='<Mmx>(b)', x_im_title=r'$\langle MM_x \rangle(b)$',
                    sigma_field='sig_map', y_title="$M_{ap}^2$"),  # m2
        PlotDetails(t_field='<NMap>', t_title=r'$\langle NM_{ap} \rangle$',
                    x_field='<NMx>', x_title=r'$\langle NM_{x} \rangle$',
                    sigma_field='sig_nmap', y_title="$NM_{ap}$"),  # nm or norm
        ]

    def makeCatalog(self, data, config=None, use_as_k=None, use_chip_coords=False):
        if data is None or isinstance(data, treecorr.Catalog):
            return data
        catalog_kwargs = {}
        fields = data.dtype.names
        if 'ra' in fields and 'dec' in fields:
            if not use_chip_coords:
                catalog_kwargs['ra'] = data['ra']
                catalog_kwargs['dec'] = data['dec']
            elif 'x' in fields and 'y' in fields:
                catalog_kwargs['x'] = data['x']
                catalog_kwargs['y'] = data['y']
            else:
                raise ValueError('Chip coordinates requested, but "x" and "y" fields not found '
                                 'in data')
        elif 'x' in fields and 'y' in fields:
            catalog_kwargs['x'] = data['x']
            catalog_kwargs['y'] = data['y']
        else:
            raise ValueError("Data must contain (ra,dec) or (x,y) in order to do correlation "
                             "function tests.")
        if 'g1' in fields and 'g2' in fields:
            catalog_kwargs['g1'] = data['g1']
            catalog_kwargs['g2'] = data['g2']
        if use_as_k:
            if use_as_k in fields:
                catalog_kwargs['k'] = data[use_as_k]
        elif 'k' in fields:
            catalog_kwargs['k'] = data['k']
        # Quirk of length-1 formatted arrays: the fields will be floats, not
        # arrays, which would break the Catalog init.
        try:
            len(data)
        except:
            if not hasattr(data, 'len') and isinstance(data, numpy.ndarray):
                for key in catalog_kwargs:
                    catalog_kwargs[key] = numpy.array([catalog_kwargs[key]])
        catalog_kwargs['config'] = config
        return treecorr.Catalog(**catalog_kwargs)

    def getCF(self, correlation_function_type, data, data2=None,
                    random=None, random2=None, use_as_k=None, use_chip_coords=False,
                    config=None, **kwargs):
        """
        Sets up and calls TreeCorr on the given set of data and possibly randoms.

        The user needs to specify the type of correlation function requested.  The available types
        are:

            * **nn**: a 2-point correlation function
            * **ng**: a point-shear correlation function (eg galaxy-galaxy lensing)
            * **gg**: a shear-shear correlation function (eg cosmic shear)
            * **nk**: a point-scalar [such as convergence, hence k meaning "kappa"] correlation
              function
            * **kk**: a scalar-scalar correlation function
            * **kg**: a scalar-shear correlation function
            * **m2**: an aperture mass measurement
            * **nm**: an <N aperture mass> measurement
            * **norm**: ``nm`` properly normalized by the average values of n and aperture mass to
              return something like a correlation coefficient.

        More details can be found in the ``Readme.md`` for TreeCorr.

        Additionally, for the **nn**, **ng**, **nk**, **nm** and **norm** options, the
        user can pass a kwarg ``nn_statistic = 'compensated'`` or ``nn_statistic = 'simple'`` (or
        similarly for **ng** and **nk**; note that the **nm** type checks the
        ``ng_statistic`` kwarg and the **norm** type checks the ``nn_statistic`` kwarg!).  For
        **nn** and **norm** correlation functions, ``'compensated'`` is the Landy-Szalay
        estimator, while ``'simple'`` is just (data/random - 1).  For the other kinds,
        ``'compensated'`` means the random-shear or random-kappa correlation function is subtracted
        from the data correlation function,  while ``'simple'`` merely returns the data correlation
        function.  Again, the TreeCorr documentation contains more information.  The
        ``*_statistic`` kwarg will be ignored if it is passed for any other correlation function
        type.  The default is to use ``'compensated'`` if randoms are present and ``'simple'``
        otherwise.

        This function accepts all (self-consistent) sets of data, data2, random, and random2.
        Including ``data2`` and possibly ``random2`` will return a cross-correlation; otherwise
        the program returns an autocorrelation.  ``Random`` datasets are necessary for the **nn**
        form of the correlation function, and can be used (but are not necessary) for **ng**,
        **nk**, and **kg**.

        :param stile_args:    The dict containing the parameters that control Stile's behavior
        :param correlation_function_type: The type of correlation function (``'nn', 'ng', 'gg',
                              'nk', 'k2', 'kg', 'm2', 'nm', 'norm'``) to request from
                              TreeCorr--see above.
        :param data:          NumPy array of data with fields using the field name
                              strings given in the ``stile.fieldNames`` dict.
        :param data2:         Optional cross-correlation data set
        :param random:        Optional random dataset corresponding to `data`
        :param random2:       Optional random dataset corresponding to `data2`
        :param kwargs:        Any other TreeCorr parameters (will silently supercede anything in
                              ``stile_args``).
        :returns:             a numpy array of the TreeCorr outputs.
        """
        import tempfile
        import os

        if not correlation_function_type in treecorr_func_dict:
            raise ValueError('Unknown correlation function type: %s'%correlation_function_type)

        handle, output_file = tempfile.mkstemp()

        # First, pull out the TreeCorr-relevant parameters from the stile_args dict, and add
        # anything passed as a kwarg to that dict.
        if (random and len(random)) or (random2 and len(random2)):
            treecorr_kwargs[correlation_function_type+'_statistic'] = \
                treecorr_kwargs.get(correlation_function_type+'_statistic', 'compensated')
        treecorr_kwargs = stile.treecorr_utils.PickTreeCorrKeys(config)
        treecorr_kwargs.update(stile.treecorr_utils.PickTreeCorrKeys(kwargs))
        treecorr.config.check_config(treecorr_kwargs, corr2_valid_params)

        if data is None:
            raise ValueError('Must include a data array!')
        if correlation_function_type == 'nn':
            if random is None or ((data2 is not None or random2 is not None) and not
                                  (data2 is not None and random2 is not None)):
                raise ValueError('Incorrect data types for correlation function: must have '
                                   'data and random, and random2 if data2.')
        elif correlation_function_type in ['gg', 'm2', 'kk']:
            if random or random2:
                print("Warning: randoms ignored for this correlation function type")
        elif correlation_function_type in ['ng', 'nm', 'nk']:
            if data2 is None:
                raise ValueError('Must include data2 for this correlation function type')
            if random2 is not None:
                print("Warning: random2 ignored for this correlation function type")
        elif correlation_function_type == 'norm':
            if data2 is None:
                raise ValueError('Must include data2 for this correlation function type')
            if random is None:
                raise ValueError('Must include random for this correlation function type')
            if random2 is None:
                print("Warning: random2 ignored for this correlation function type")
        elif correlation_function_type == 'kg':
            if data2 is None:
                raise ValueError('Must include data2 for this correlation function type')
            if random is not None or random2 is not None:
                print("Warning: randoms ignored for this correlation function type")

        data = self.makeCatalog(data, config=treecorr_kwargs, use_as_k=use_as_k,
                                      use_chip_coords=use_chip_coords)
        data2 = self.makeCatalog(data2, config=treecorr_kwargs, use_as_k=use_as_k,
                                        use_chip_coords=use_chip_coords)
        random = self.makeCatalog(random, config=treecorr_kwargs, use_as_k=use_as_k,
                                          use_chip_coords=use_chip_coords)
        random2 = self.makeCatalog(random2, config=treecorr_kwargs, use_as_k=use_as_k,
                                            use_chip_coords=use_chip_coords)

        treecorr_kwargs[correlation_function_type+'_file_name'] = output_file

        func = treecorr_func_dict[correlation_function_type](treecorr_kwargs)
        func.process(data, data2)
        if correlation_function_type in ['ng', 'nm', 'nk']:
            comp_stat = {'ng': 'ng', 'nm': 'ng', 'nk': 'nk'}  # which _statistic kwarg to check
            if treecorr_kwargs.get(comp_stat[correlation_function_type]+'_statistic',
               self.compensateDefault(data, data2, random, random2)) == 'compensated':
                func_random = treecorr_func_dict[correlation_function_type](treecorr_kwargs)
                func_random.process(random, data2)
            else:
                func_random = None
        elif correlation_function_type == 'norm':
            func_gg = treecorr_func_dict['gg'](treecorr_kwargs)
            func_gg.process(data2)
            func_dd = treecorr_func_dict['nn'](treecorr_kwargs)
            func_dd.process(data)
            func_rr = treecorr_func_dict['nn'](treecorr_kwargs)
            func_rr.process(data)
            if treecorr_kwargs.get('nn_statistic',
               self.compensateDefault(data, data2, random, random2, both=True)) == 'compensated':
                func_dr = treecorr_func_dict['nn'](treecorr_kwargs)
                func_dr.process(data, random)
            else:
                func_dr = None
        elif correlation_function_type == 'nn':
            func_random = treecorr_func_dict[correlation_function_type](treecorr_kwargs)
            if len(random2):
                func_random.process(random, random2)
            else:
                func_random.process(random)
            if not len(data2):
                func_rr = treecorr_func_dict['nn'](treecorr_kwargs)
                func_rr.process(data, random)
                if treecorr_kwargs.get(['nn_statistic'],
                   self.compensateDefault(data, data2, random, random2, both=True)
                   ) == 'compensated':
                    func_dr = treecorr_func_dict['nn'](treecorr_kwargs)
                    func_dr.process(data, random)
                    func_rd = None
                else:
                    func_dr = None
                    func_rd = None
            else:
                func_rr = treecorr_func_dict['nn'](treecorr_kwargs)
                func_rr.process(random, random2)
                if treecorr_kwargs.get(['nn_statistic'],
                   self.compensateDefault(data, data2, random, random2, both=True)
                   ) == 'compensated':
                    func_dr = treecorr_func_dict['nn'](treecorr_kwargs)
                    func_dr.process(data, random2)
                    func_rd = treecorr_func_dict['nn'](treecorr_kwargs)
                    func_rd.process(random, data2)
        else:
            func_random = None
        if correlation_function_type == 'm2':
            func.writeMapSq(output_file)
        elif correlation_function_type == 'nm':
            func.writeNMap(output_file, func_random)
        elif correlation_function_type == 'norm':
            func.writeNorm(output_file, func_gg, func_dd, func_rr, func_dr, func_rg)
        elif correlation_function_type == 'nn':
            func.write(output_file, func_rr, func_dr, func_rd)
        elif func_random:
            func.write(output_file, func_random)
        else:
            func.write(output_file)
        results = stile.ReadTreeCorrResultsFile(output_file)
        os.close(handle)
        os.remove(output_file)
        names = results.dtype.names
        # Add the sep units to the column names of radial bins from TreeCorr outputs
        names = [n+' [%s]'%treecorr_kwargs['sep_units'] if 'R' in n else n for n in names]
        results.dtype.names = names
        return results

    def compensateDefault(self, data, data2, random, random2, both=False):
        """
        Figure out if a compensated statistic can be used from the data present.  Keyword ``both``
        indicates that both data sets if present must have randoms; the default, False, means only
        the first data set must have an associated random.
        """
        if not random or (random and not len(random)):  # No random
            return 'simple'
        elif both and data2 and len(data2):  # Second data set exists and must have a random
            if random2 and len(random2):
                return 'compensated'
            else:
                return 'simple'
        else:  # There's a random, and we can ignore 'both' since this is an autocorrelation
            return 'compensated'

    def plot(self, data, colors=['r', 'b'], log_yscale=False,
                   plot_bmode=True, plot_data_only=True, plot_random_only=True):
        """
        Plot the data returned from a :class:`BaseCorrelationFunctionSysTest` object.  This chooses
        some sensible defaults, but much of its behavior can be changed.

        :param data:       The data returned from a :class:`BaseCorrelationFunctionSysTest`, as-is.
        :param colors:     A tuple of 2 colors, used for the first and second lines on any given
                           plot
        :param log_yscale: Whether to use a logarithmic y-scale [default: False]
        :param plot_bmode: Whether to plot the b-mode signal, if there is one [default: True]
        :param plot_data_only:   Whether to plot the data-only correlation functions, if present
                                 [default: True]
        :param plot_random_only: Whether to plot the random-only correlation functions, if present
                                 [default: True]
        :returns:          A matplotlib ``Figure`` which may be written to a file with
                           :func:`.savefig()`, if matplotlib can be imported; else None.
        """

        if not has_matplotlib:
            return None
        fields = data.dtype.names
        # Pick which radius measurement to use
        # TreeCorr changed the name of the output columns
        # This catches the case where we added the units to the label
        fields_no_units = [f.split(' [')[0] for f in fields]
        for t_r in ['meanR', 'R_nom', '<R>', 'R_nominal', 'R']:
            if t_r in fields:
                # Protect underscores since they blow up the plotting routines
                r = '\\_'.join(t_r.split('\\'))
                break
            elif t_r in fields_no_units:
                t_i = fields_no_units.index(t_r)
                r = '\\_'.join(fields[t_i].split('\\'))
                break
        else:
            raise ValueError('No radius parameter found in data')

        # Logarithmic x-axes have stupid default ranges: fix this.
        rstep = data[r][1]/data[r][0]
        xlim = [min(data[r])/rstep, max(data[r])*rstep]
        # Check what kind of data is in the array that .plot() received.
        for plot_details in self.plot_details:
            # Pick the one the data contains and use it; break before trying the others.
            if plot_details.t_field in fields:
                pd = plot_details
                break
        else:
            raise ValueError("No valid y-values found in data")
        if log_yscale:
            yscale = 'log'
        else:
            yscale = 'linear'
        fig = plt.figure()
        fig.subplots_adjust(hspace=0)  # no space between stacked plots
        plt.subplots(sharex=True)  # share x-axes
        # Figure out how many plots you'll need--never more than 3, so we just use a stacked column.
        if pd.datarandom_t_field:
            plot_data_only &= pd.datarandom_t_field+'d' in fields
            plot_random_only &= pd.datarandom_t_field+'r' in fields
        if plot_bmode and pd.x_field and pd.t_im_field:
            nrows = 2
        elif pd.datarandom_t_field:
            nrows = 1 + plot_data_only + plot_random_only
        else:
            nrows = 1
        # Plot the first thing
        curr_plot = 0
        ax = fig.add_subplot(nrows, 1, 1)
        ax.axhline(0, alpha=0.7, color='gray')
        ax.errorbar(data[r], data[pd.t_field], yerr=data[pd.sigma_field], color=colors[0],
                    label=pd.t_title)
        if pd.x_title and plot_bmode:
            ax.errorbar(data[r], data[pd.x_field], yerr=data[pd.sigma_field], color=colors[1],
                        label=pd.x_title)
        elif pd.t_im_title:  # Plot y and y_im if not plotting yb (else it goes on a separate plot)
            ax.errorbar(data[r], data[pd.t_im_field], yerr=data[pd.sigma_field], color=colors[1],
                        label=pd.t_im_title)
        ax.set_xscale('log')
        ax.set_yscale(yscale)
        ax.set_xlim(xlim)
        # To prevent too-long decimal y-axis ticklabels that push the label out of frame
        ax.ticklabel_format(axis='y', style='sci', scilimits=(-3,5))
        ax.set_ylabel(pd.y_title)
        ax.legend()
        if pd.x_field and plot_bmode and pd.t_im_field:
            # Both yb and y_im: plot (y, yb) on one plot and (y_im, yb_im) on the other.
            ax = fig.add_subplot(nrows, 1, 2)
            ax.axhline(0, alpha=0.7, color='gray')
            ax.errorbar(data[r], data[pd.t_im_field], yerr=data[pd.sigma_field], color=colors[0],
                        label=pd.t_im_title)
            ax.errorbar(data[r], data[pd.x_im_field], yerr=data[pd.sigma_field], color=colors[1],
                        label=pd.x_im_title)
            ax.set_xscale('log')
            ax.set_yscale(yscale)
            ax.ticklabel_format(axis='y', style='sci', scilimits=(-3,5))
            ax.set_xlim(xlim)
            ax.set_ylabel(pd.y_title)
            ax.legend()
        if plot_data_only and pd.datarandom_t_field:  # Plot the data-only measurements if requested
            curr_plot += 1
            ax = fig.add_subplot(nrows, 1, 2)
            ax.axhline(0, alpha=0.7, color='gray')
            ax.errorbar(data[r], data[pd.datarandom_t_field+'d'], yerr=data[pd.sigma_field],
                        color=colors[0], label=pd.datarandom_t_title+'d}$')
            if plot_bmode and pd.datarandom_x_field:
                ax.errorbar(data[r], data[pd.datarandom_x_field+'d'], yerr=data[pd.sigma_field],
                        color=colors[1], label=pd.datarandom_x_title+'d}$')
            ax.set_xscale('log')
            ax.set_yscale(yscale)
            ax.ticklabel_format(axis='y', style='sci', scilimits=(-3,5))
            ax.set_xlim(xlim)
            ax.set_ylabel(pd.y_title)
            ax.legend()
        # Plot the randoms-only measurements if requested
        if plot_random_only and pd.datarandom_t_field:
            ax = fig.add_subplot(nrows, 1, nrows)
            ax.axhline(0, alpha=0.7, color='gray')
            ax.errorbar(data[r], data[pd.datarandom_t_field+'r'], yerr=data[pd.sigma_field],
                        color=colors[0], label=pd.datarandom_t_title+'r}$')
            if plot_bmode and pd.datarandom_x_field:
                ax.errorbar(data[r], data[pd.datarandom_x_field+'r'], yerr=data[pd.sigma_field],
                        color=colors[1], label=pd.datarandom_x_title+'r}$')
            ax.set_xscale('log')
            ax.set_yscale(yscale)
            ax.ticklabel_format(axis='y', style='sci', scilimits=(-3,5))
            ax.set_xlim(xlim)
            ax.set_ylabel(pd.y_title)
            ax.legend()
        ax.set_xlabel(r)
        plt.tight_layout()
        return fig

    def __call__(self, *args, **kwargs):
        return self.getCF(*args, **kwargs)


class GalaxyShearSysTest(BaseCorrelationFunctionSysTest):
    """
    Compute the tangential and cross shear around a set of real galaxies.
    """
    short_name = 'shear_around_galaxies'
    long_name = 'Shear of galaxies around real objects'
    objects_list = ['galaxy lens', 'galaxy']
    required_quantities = [('ra', 'dec'), ('ra', 'dec', 'g1', 'g2', 'w')]

    def __call__(self, data, data2=None, random=None, random2=None, config=None, **kwargs):
        return self.getCF('ng', data, data2, random, random2, config=config, **kwargs)

class BrightStarShearSysTest(BaseCorrelationFunctionSysTest):
    """
    Compute the tangential and cross shear around a set of bright stars.
    """
    short_name = 'shear_around_bright_stars'
    long_name = 'Shear of galaxies around bright stars'
    objects_list = ['star bright', 'galaxy']
    required_quantities = [('ra', 'dec'), ('ra', 'dec', 'g1', 'g2', 'w')]

    def __call__(self, data, data2=None, random=None, random2=None, config=None, **kwargs):
        return self.getCF('ng', data, data2, random, random2, config=config, **kwargs)

class StarXGalaxyDensitySysTest(BaseCorrelationFunctionSysTest):
    """
    Compute the number density of galaxies around stars.
    """
    short_name = 'star_x_galaxy_density'
    long_name = 'Density of galaxies around stars'
    objects_list = ['star', 'galaxy', 'star random', 'galaxy random']
    required_quantities = [('ra', 'dec'), ('ra', 'dec'), ('ra', 'dec'), ('ra', 'dec')]

    def __call__(self, data, data2=None, random=None, random2=None, config=None, **kwargs):
        return self.getCF('nn', data, data2, random, random2, config=config, **kwargs)

class StarXGalaxyShearSysTest(BaseCorrelationFunctionSysTest):
    """
    Compute the cross-correlation of galaxy and star shapes.
    """
    short_name = 'star_x_galaxy_shear'
    long_name = 'Cross-correlation of galaxy and star shapes'
    objects_list = ['star', 'galaxy']
    required_quantities = [('ra', 'dec', 'g1', 'g2', 'w'), ('ra', 'dec', 'g1', 'g2', 'w')]

    def __call__(self, data, data2=None, random=None, random2=None, config=None, **kwargs):
        return self.getCF('gg', data, data2, random, random2, config=config, **kwargs)

class StarXStarShearSysTest(BaseCorrelationFunctionSysTest):
    """
    Compute the auto-correlation of star shapes.
    """
    short_name = 'star_x_star_shear'
    long_name = 'Auto-correlation of star shapes'
    objects_list = ['star']
    required_quantities = [('ra', 'dec', 'g1', 'g2', 'w')]

    def __call__(self, data, data2=None, random=None, random2=None, config=None, **kwargs):
        return self.getCF('gg', data, data2, random, random2, config=config, **kwargs)

class StarXStarSizeResidualSysTest(BaseCorrelationFunctionSysTest):
    """
    Compute the auto correlation of star-PSF size residuals.
    """
    short_name = 'star_x_star_size_residual'
    long_name = 'Auto-correlation of residual star sizes'
    objects_list = ['star PSF']
    required_quantities = [('ra', 'dec', 'sigma', 'psf_sigma')]
    def __call__(self, data, data2=None, random=None, random2=None, config=None, **kwargs):
        new_kwargs = kwargs.copy()
        new_kwargs['use_as_k'] = 'sigma'
        data_list = []
        for data_item in [data, data2, random, random2]:
            if data_item is not None:
                new_data = data_item.copy()
                new_data['sigma'] = ((new_data['psf_sigma'] - new_data['sigma'])/
                                     new_data['psf_sigma'])
                data_list.append(new_data)
            else:
                data_list.append(data_item)
        return self.getCF('kk', config=config, *data_list, **new_kwargs)


class Rho1SysTest(BaseCorrelationFunctionSysTest):
    """
    Compute the auto-correlation of residual star shapes (star shapes - psf shapes).
    """
    short_name = 'rho1'
    long_name = 'Rho1 statistics (Auto-correlation of star-PSF shapes)'
    objects_list = ['star PSF']
    required_quantities = [('ra', 'dec', 'g1', 'g2', 'psf_g1', 'psf_g2', 'w')]

    def __call__(self, data, data2=None, random=None, random2=None, config=None, **kwargs):
        new_data = data.copy()
        new_data['g1'] = new_data['g1'] - new_data['psf_g1']
        new_data['g2'] = new_data['g2'] - new_data['psf_g2']
        if data2 is not None:
            new_data2 = data2.copy()
            new_data2['g1'] = new_data2['g1'] - new_data2['psf_g1']
            new_data2['g2'] = new_data2['g2'] - new_data2['psf_g2']
        else:
            new_data2 = data2
        if random is not None:
            new_random = random.copy()
            new_random['g1'] = new_random['g1'] - new_random['psf_g1']
            new_random['g2'] = new_random['g2'] - new_random['psf_g2']
        else:
            new_random = random
        if random2 is not None:
            new_random2 = random2.copy()
            new_random2['g1'] = new_random2['g1'] - new_random2['psf_g1']
            new_random2['g2'] = new_random2['g2'] - new_random2['psf_g2']
        else:
            new_random2 = random2
        return self.getCF('gg', new_data, new_data2, new_random, new_random2,
                          config=config, **kwargs)

class Rho2SysTest(BaseCorrelationFunctionSysTest):
    """
    Compute the correlation of PSF shapes with residual star shapes (star shapes - psf shapes).
    """
    short_name = 'rho2'
    long_name = 'Rho2 statistics (Correlation of PSF shapes with star-PSF shapes)'
    objects_list = ['star PSF']
    required_quantities = [('ra', 'dec', 'g1', 'g2', 'psf_g1', 'psf_g2', 'w')]

    def __call__(self, data, data2=None, random=None, random2=None, config=None, **kwargs):
        new_data = numpy.rec.fromarrays([data['ra'], data['dec'], data['psf_g1'],
                                         data['psf_g2'], data['w']],
                                         names = ['ra', 'dec', 'g1', 'g2', 'w'])
        if data2 is None:
            data2 = data
        new_data2 = numpy.rec.fromarrays([data2['ra'], data2['dec'], data2['g1']-data2['psf_g1'],
                                          data2['g2']-data2['psf_g2'], data2['w']],
                                          names = ['ra', 'dec', 'g1', 'g2', 'w'])
        if random is not None:
            new_random = numpy.rec.fromarrays([random['ra'], random['dec'], random['psf_g1'],
                                               random['psf_g2'], random['w']],
                                               names = ['ra', 'dec', 'g1', 'g2', 'w'])

        else:
            new_random = random
        if random2 is None:
            random2 = random
        if random2 is not None:
            new_random2 = numpy.rec.fromarrays([data2['ra'], data2['dec'],
                                                data2['g1']-data2['psf_g1'],
                                                data2['g2']-data2['psf_g2'], data2['w']],
                                                names = ['ra', 'dec', 'g1', 'g2', 'w'])
        else:
            new_random2 = random2
        return self.getCF('gg', new_data, new_data2, new_random, new_random2,
                          config=config, **kwargs)


class Rho3SysTest(BaseCorrelationFunctionSysTest):
    """
    Compute the correlation of star shapes weighted by the residual size.
    """
    short_name = 'rho3'
    long_name = 'Rho3 statistics (Auto-correlation of star shapes weighted by the residual size)'
    objects_list = ['star PSF']
    required_quantities = [('ra', 'dec', 'sigma',
                            'psf_g1', 'psf_g2', 'psf_sigma', 'w')]

    def __call__(self, data, data2=None, random=None, random2=None, config=None, **kwargs):
        new_data = numpy.rec.fromarrays([data['ra'], data['dec'],
                                 data['psf_g1']*(data['sigma']-data['psf_sigma'])/data['psf_sigma'],
                                 data['psf_g2']*(data['sigma']-data['psf_sigma'])/data['psf_sigma'],
                                 data['w']],
                                 names = ['ra', 'dec', 'g1', 'g2', 'w'])
        if data2 is not None:
            new_data2 = numpy.rec.fromarrays([data2['ra'], data2['dec'],
                             data2['psf_g1']*(data2['sigma']-data2['psf_sigma'])/data2['psf_sigma'],
                             data2['psf_g2']*(data2['sigma']-data2['psf_sigma'])/data2['psf_sigma'],
                             data2['w']],
                             names = ['ra', 'dec', 'g1', 'g2', 'w'])
        else:
            new_data2 = data2
        if random is not None:
            new_random = numpy.rec.fromarrays([random['ra'], random['dec'],
                         random['psf_g1']*(random['sigma']-random['psf_sigma'])/random['psf_sigma'],
                         random['psf_g2']*(random['sigma']-random['psf_sigma'])/random['psf_sigma'],
                         random['w']],
                         names = ['ra', 'dec', 'g1', 'g2', 'w'])
        else:
            new_random = random

        if random2 is not None:
            new_random2 = numpy.rec.fromarrays([random2['ra'], random2['dec'],
                    random2['psf_g1']*(random2['sigma']-random2['psf_sigma'])/random2['psf_sigma'],
                    random2['psf_g2']*(random2['sigma']-random2['psf_sigma'])/random2['psf_sigma'],
                    random2['w']],
                    names = ['ra', 'dec', 'g1', 'g2', 'w'])
        else:
            new_random2 = random2
        return self.getCF('gg', new_data, new_data2, new_random, new_random2,
                          config=config, **kwargs)

class Rho4SysTest(BaseCorrelationFunctionSysTest):
    """
    Compute the correlation of star shapes weighted by the residual size.
    """
    short_name = 'rho4'
    long_name = 'Rho4 statistics (Correlation of residual star shapes weighted by residual size)'
    objects_list = ['star PSF']
    required_quantities = [('ra', 'dec', 'g1', 'g2', 'sigma',
                            'psf_g1', 'psf_g2', 'psf_sigma', 'w')]

    def __call__(self, data, data2=None, random=None, random2=None, config=None, **kwargs):
        new_data = numpy.rec.fromarrays([data['ra'], data['dec'], data['g1'] - data['psf_g1'],
                                         data['g2']-data['psf_g2'], data['w']],
                                        names = ['ra', 'dec', 'g1', 'g2', 'w'])
        if data2 is None:
            data2 = data
        new_data2 = numpy.rec.fromarrays([data2['ra'], data2['dec'],
                             data2['psf_g1']*(data2['sigma']-data2['psf_sigma'])/data2['psf_sigma'],
                             data2['psf_g2']*(data2['sigma']-data2['psf_sigma'])/data2['psf_sigma'],
                             data2['w']],
                             names = ['ra', 'dec', 'g1', 'g2', 'w'])
        if random is not None:
            new_random = numpy.rec.fromarrays([random['ra'], random['dec'],
                                               random['g1']-random['psf_g1'],
                                               random['g2']-random['psf_g2'], random['w']],
                                              names = ['ra', 'dec', 'g1', 'g2', 'w'])
        else:
            new_random = random
        if random2 is None:
            random2 = random
        if random2 is not None:
            new_random2 = numpy.rec.fromarrays([random2['ra'], random2['dec'],
                    random2['psf_g1']*(random2['sigma']-random2['psf_sigma'])/random2['psf_sigma'],
                    random2['psf_g2']*(random2['sigma']-random2['psf_sigma'])/random2['psf_sigma'],
                    random2['w']],
                   names = ['ra', 'dec', 'g1', 'g2', 'w'])
        else:
            new_random2 = random2
        return self.getCF('gg', new_data, new_data2, new_random, new_random2,
                          config=config, **kwargs)

class Rho5SysTest(BaseCorrelationFunctionSysTest):
    """
    Compute the correlation of star shapes weighted by the residual size.
    """
    short_name = 'rho5'
    long_name = 'Rho5 statistics (Correlation of star and PSF shapes weighted by residual size)'
    objects_list = ['star PSF']
    required_quantities = [('ra', 'dec', 'sigma',
                            'psf_g1', 'psf_g2', 'psf_sigma', 'w')]

    def __call__(self, data, data2=None, random=None, random2=None, config=None, **kwargs):
        new_data = numpy.rec.fromarrays([data['ra'], data['dec'],data['psf_g1'],
                                         data['psf_g2'], data['w']],
                                        names = ['ra', 'dec', 'g1', 'g2', 'w'])
        if data2 is None:
            data2 = data
        new_data2 = numpy.rec.fromarrays([data2['ra'], data2['dec'],
                             data2['psf_g1']*(data2['sigma']-data2['psf_sigma'])/data2['psf_sigma'],
                             data2['psf_g2']*(data2['sigma']-data2['psf_sigma'])/data2['psf_sigma'],
                             data2['w']],
                             names = ['ra', 'dec', 'g1', 'g2', 'w'])
        if random is not None:
            new_random = numpy.rec.fromarrays([random['ra'], random['dec'],
                                               random['psf_g1'], random['psf_g2'], random['w']],
                                              names = ['ra', 'dec', 'g1', 'g2', 'w'])
        else:
            new_random = random
        if random2 is None:
            random2 = random
        if random2 is not None:
            new_random2 = numpy.rec.fromarrays([random2['ra'], random2['dec'],
                    random2['psf_g1']*(random2['sigma']-random2['psf_sigma'])/random2['psf_sigma'],
                    random2['psf_g2']*(random2['sigma']-random2['psf_sigma'])/random2['psf_sigma'],
                    random2['w']],
                   names = ['ra', 'dec', 'g1', 'g2', 'w'])
        else:
            new_random2 = random2
        return self.getCF('gg', new_data, new_data2, new_random, new_random2,
                          config=config, **kwargs)


class GalaxyDensityCorrelationSysTest(BaseCorrelationFunctionSysTest):
    """
    Compute the galaxy position autocorrelations.
    """
    short_name = 'galaxy_density'
    long_name = 'Galaxy position autocorrelation'
    objects_list = ['galaxy', 'galaxy random']
    required_quantities = [('ra', 'dec'), ('ra', 'dec')]

    def __call__(self, data, data2=None, random=None, random2=None, config=None, **kwargs):
        return self.getCF('nn', data, data2, random, random2, config=config, **kwargs)

class StarDensityCorrelationSysTest(BaseCorrelationFunctionSysTest):
    """
    Compute the star position autocorrelations.
    """
    short_name = 'star_density'
    long_name = 'Star position autocorrelation'
    objects_list = ['star', 'star random']
    required_quantities = [('ra', 'dec'), ('ra', 'dec')]

    def __call__(self, data, data2=None, random=None, random2=None, config=None, **kwargs):
        return self.getCF('nn', data, data2, random, random2, config=config, **kwargs)


class StatSysTest(SysTest):
    """
    A class for the Stile systematics tests that use basic statistical quantities. It uses NumPy
    routines for all the innards, and saves the results in a :class:`stile.stile_utils.Stats` object
    that can carry around the information, print the results in a useful
    format, write to file, or (eventually) become an argument to plotting routines that might output
    some of the results on plots.

    One of the calculations it does is find the percentiles of the given quantity.  The percentile
    levels to use can be set when the StatSysTest is initialized or when it is called.  These
    percentiles must be provided as an iterable (list, tuple, or NumPy array).

    The objects on which this systematics test is used should be either (a) a simple iterable like a
    list, tuple, or NumPy array, or (b) a structured NumPy array with fields.  In case (a), the
    dimensionality of the NumPy array is ignored, and statistics are calculated over all
    dimensions.  In case (b), the user must give a field name using the `field` keyword argument,
    either at initialization or when calling the test.

    For both the ``percentile`` and ``field`` arguments, the behavior is different if the keyword
    argument is used at the time of initialization or calling.  When used at the time of
    initialization, that value will be used for all future calls unless called with another value
    for those arguments.  However, the value of ``percentile`` and ``field`` for calls after that
    will revert back to the original value from the time of initialization.

    By default, this object will simply return a Stats object for the user.  However,
    calling it with ``verbose=True`` will result in the statistics being printed directly.

    Ordinarily, a StatSysTest object will throw an exception if asked to run on an array that has
    any ``NaN``\s or infinite values.  The ``ignore_bad`` keyword (at the time when
    the :class:`StatSytTest` is called, not initialized) changes this behavior so these bad values
    are quietly ignored.

    Options to consider adding in future: weighted sums and other weighted statistics; outlier
    rejection.
    """
    short_name = 'stats'
    long_name = 'Calculate basic statistics of a given quantity'

    def __init__(self, percentiles=[2.2, 16., 50., 84., 97.8], field=None):
        """Function to initialize a :class:`StatSysTest` object.

        :param percentiles:     The percentile levels at which to find the value of the input array
                                when called.  [default: ``[2.2, 16., 50., 84., 97.8]``.]
        :param field:           The name of the field to use in a NumPy structured array / catalog.
                                [default: None, meaning we're using a simple array without field
                                names.]

        :returns: the requested :class:`StatSysTest` object.
        """
        self.percentiles = percentiles
        self.field = field

    def __call__(self, array, percentiles=None, field=None, verbose=False, ignore_bad=False):
        """Calling a :class:`StatSysTest` with a given array argument as ``array`` will cause it to
        carry out all the statistics tests and populate a :class:`stile.Stats` object with the
        results, which it returns to the user.

        :param array:           The tuple, list, NumPy array, or structured NumPy array/catalog on
                                which to carry out the calculations.
        :param percentiles:     The percentile levels to use for this particular calculation.
                                [default: None, meaning use whatever levels were defined when
                                initializing this :class:`StatSysTest` object]
        :param field:           The name of the field to use in a NumPy structured array / catalog.
                                [default: None, meaning use whatever field was defined when
                                initializing this :class:`StatSysTest` object]
        :param verbose:         If True, print the calculated statistics of the input ``array``
                                to screen.  If False, silently return the
                                :class:`Stats <stile.stile_utils.Stats>` object.
                                [default: False.]
        :param ignore_bad:      If True, search for values that are ``NaN`` or ``Inf``, and
                                remove them before doing calculations.  [default: False.]

        :returns: a :class:`stile.stile_utils.Stats` object
        """
        # Set the percentile levels and field, if the user provided them.  Otherwise use what was
        # set up at the time of initialization.
        use_percentiles = percentiles if percentiles is not None else self.percentiles
        use_field = field if field is not None else self.field

        # Check to make sure that percentiles is iterable (list, numpy array, tuple, ...)
        if not hasattr(use_percentiles, '__iter__'):
            raise RuntimeError('List of percentiles is not an iterable (list, tuple, NumPy array)!')

        # Check types for input things and make sure it all makes sense, including consistency with
        # the field.  First of all, it should be iterable:
        if not hasattr(array, '__iter__'):
            raise RuntimeError('Input array is not an iterable (list, tuple, NumPy array)!')
        # If it's a multi-dimensional NumPy array, tuple, or list, we don't care - the functions
        # we'll use below will simply work as if it's a 1d NumPy array, collapsing all rows of a
        # multi-dimensional array implicitly.  The only thing we have to worry about is if this is
        # really a structured catalog.  The cases to check are:
        # (a) Is it a structured catalog?  If so, we must have some value for `use_field` that is
        #     not None and that is in the catalog.  We can check the values in the catalog using
        #     array.dtype.field.keys(), which returns a list of the field names.
        # (b) Is `use_field` set, but this is not a catalog?  If so, we'll issue a warning (not
        #     exception!) and venture bravely onwards using the entire array, leaving it to the user
        #     to decide if they are okay with that.
        # We begin with taking care of case (a).  Just be careful not to modify input.
        use_array = numpy.array(array)
        if use_array.dtype.fields is not None:
            # It's a catalog, not a simple array
            if use_field is None:
                raise RuntimeError('StatSysTest called on a catalog without specifying a field!')
            if use_field not in list(use_array.dtype.fields.keys()):
                raise RuntimeError('Field %s is not in this catalog, which contains %s!'%
                                   (use_field, list(use_array.dtype.fields.keys())))
            # Select the appropriate field for this catalog.
            use_array = use_array[use_field]
        # Now take care of case (b):
        elif use_array.dtype.fields is None and use_field is not None:
            import warnings
            warnings.warn('Field is selected, but input array is not a catalog! '
                          'Ignoring field choice and continuing')

        # Reject NaN / Inf values, if requested to do so.
        if ignore_bad:
            cond = numpy.logical_and.reduce(
                [numpy.isnan(use_array) == False,
                 numpy.isinf(use_array) == False]
                )
            use_array = use_array[cond]
            if len(use_array) == 0:
                raise RuntimeError("No good entries left to use after excluding bad values!")

        # Create the output object, a stile.Stats() object.  We gave to tell it which simple
        # statistics to calculate.  If we want to change this list, we need to change both the
        # `simple_stats` list below, and the code afterwards that calculates and populates the
        # `result` Stats object with the statistics.  (By default it always does percentiles, though
        # we could choose to change the percentile levels.)  Also note that if we want things like
        # skewness and kurtosis, we either need to calculate them directly or use scipy, since numpy
        # does not include those.  For now we use a try/except block to import scipy and calculate
        # those values if possible, but silently ignore the import failure if scipy is not
        # available.
        try:
            import scipy.stats
            simple_stats = ['min', 'max', 'median', 'mad', 'mean', 'stddev', 'variance', 'N',
                          'skew', 'kurtosis']
        except ImportError:
            simple_stats = ['min', 'max', 'median', 'mad', 'mean', 'stddev', 'variance', 'N']

        result = stile.stile_utils.Stats(simple_stats=simple_stats)

        # Populate the basic entries, like median, mean, standard deviation, etc.
        result.min = numpy.min(use_array)
        # Now do a check for NaN / inf, and raise an exception.
        if numpy.isnan(result.min) or numpy.isinf(result.min):
            raise RuntimeError("NaN or Inf values detected in input array!")
        result.max = numpy.max(use_array)
        # To get the length, be careful: multi-dimensional arrays need flattening!
        if hasattr(use_array, 'dtype'):
            result.N = len(use_array.flatten())
        else:
            result.N = len(use_array)
        result.median = numpy.median(use_array)
        result.mad = numpy.median(numpy.abs(use_array - result.median))
        result.stddev = numpy.std(use_array)
        result.variance = numpy.var(use_array)
        result.mean = numpy.mean(use_array)

        if 'skew' in simple_stats:
            # We were able to import SciPy, so calculate skewness and kurtosis.
            result.skew = scipy.stats.skew(use_array)
            result.kurtosis = scipy.stats.kurtosis(use_array)

        # Populate the percentiles and values.
        result.percentiles = use_percentiles
        result.values = numpy.percentile(use_array, use_percentiles)

        # Print, if verbose=True.
        if verbose:
            print(result.__str__())

        # Return.
        return result

def WhiskerPlotSysTest(type=None):
    """
    Initialize an instance of a :class:`BaseWhiskerPlotSysTest` class, based on the ``type`` kwarg
    given. Options are:

        - **Star**: whisker plot of shapes of PSF stars
        - **PSF**: whisker plot of PSF shapes at the location of PSF stars
        - **Residual**: whisker plot of (star shape-PSF shape)
        - **None**: an empty :class:`BaseWhiskerPlotSysTest` class instance, which can be used for
          multiple types of whisker plots.  See the documentation
          for :class:`BaseWhiskerPlotSysTest` (especially the method
          :func:`whiskerPlot`) for more details.  Note that this type has a different call
          signature than the other methods and that it lacks many of the convenience variables the
          other WhiskerPlots have, such as self.objects_list and self.required_quantities.
    """
    if type=='Star':
        return WhiskerPlotStarSysTest()
    elif type=='PSF':
        return WhiskerPlotPSFSysTest()
    elif type=='Residual':
        return WhiskerPlotResidualSysTest()
    elif type is None:
        return BaseWhiskerPlotSysTest()
    else:
        raise ValueError('Unknown whisker plot type %s given to type kwarg'%type)

class BaseWhiskerPlotSysTest(SysTest):
    """
    A base class for Stile systematics tests that generate whisker plots. This implements the class
    method :func:`whiskerPlot`. Every child class of this class
    should use :func:`whiskerPlot` through its :func:`__call__` method. See the docstring for
    :func:`whiskerPlot` for information on how to write further tests using it.
    """
    short_name = 'whiskerplot'
    def whiskerPlot(self, x, y, g1, g2, size=None, linewidth=0.01, scale=None,
                    keylength=0.05, figsize=None, xlabel=None, ylabel=None,
                    size_label=None, xlim=None, ylim=None, equal_axis=False):
        """
        Draw a whisker plot and return a :class:`matplotlib.figure.Figure` object.
        This method has a bunch of options for controlling the appearance of a plot, which are
        explained below. To implement a child class, call this function
        within the :func:`__call__` method of the child class and return
        the :class:`matplotlib.figure.Figure` that it returns.

        :param x:               The tuple, list, or NumPy array for the x-position of objects.
        :param y:               The tuple, list, or NumPy array for the y-position of objects.
        :param g1:              The tuple, list, or Numpy array for the 1st ellipticity component
                                of objects.
        :param g2:              The tuple, list, or Numpy array for the 2nd ellipticity component
                                of objects.
        :param size:            The tuple, list, or Numpy array for the size of objects. The size
                                information is shown as color gradation.
                                [default: None, meaning do not show the size information]
        :param linewidth:       Width of whiskers in units of inches.
                                [default: 0.01]
        :param scale:           Data units per inch for the whiskers; a smaller scale is a longer
                                whisker.
                                [default: None, meaning follow the default autoscaling algorithm
                                from matplotlib]
        :param keylength:       Length of a key.
                                [default: 0.05]
        :param figsize:         Size of a figure ``(x, y)`` in units of inches.
                                [default: None, meaning use the default value of matplotlib]
        :param xlabel:          The x-axis label.
                                [default: None, meaning do not show a label for the x-axis]
        :param ylabel:          The y-axis label.
                                [default: None, meaning do not show a label for the y-axis]
        :param size_label:      The label for ``size``, which is shown at the right of the color
                                bar. [default: None, meaning do not show a size label]
        :param xlim:            Limits of x-axis ``(min, max)``.
                                [default: None, meaning do not set any limits for x]
        :param ylim:            Limits of y-axis ``(min, max)``.
                                [default: None, meaning do not set any limits for y]
        :param equal_axis:      If True, force equal scaling for the x and y axes (distance between
                                ticks of the same numerical values are equal on the x and y axes).
                                [default: False]
        :returns: a :class:`matplotlib.figure.Figure` object.
        """
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1)

        # mask data with nan
        sel = numpy.logical_and.reduce(
            [numpy.isnan(x) == False, numpy.isnan(y) == False,
             numpy.isnan(g1) == False, numpy.isnan(g2) == False])
        sel = numpy.logical_and(sel, numpy.isnan(size) == False) if size is not None else sel
        x = x[sel]
        y = y[sel]
        g1 = g1[sel]
        g2 = g2[sel]
        size = size[sel] if size is not None else size

        # plot
        g = numpy.sqrt(g1*g1+g2*g2)
        theta = numpy.arctan2(g2, g1)/2
        gx = g * numpy.cos(theta)
        gy = g * numpy.sin(theta)
        if size is None:
            q = ax.quiver(x, y, gx, gy, units='inches',
                          headwidth=0., headlength=0., headaxislength=0.,
                          pivot='middle', width=linewidth,
                          scale=scale)
        else:
            q = ax.quiver(x, y, gx, gy, size, units='inches',
                          headwidth=0., headlength=0., headaxislength=0.,
                          pivot='middle', width=linewidth,
                          scale=scale)
            cb = fig.colorbar(q)
            if size_label is not None:
                cb.set_label(size_label)

        qk = plt.quiverkey(q, 0.5, 0.92, keylength, r'$g= %s$' % str(keylength), labelpos='W')
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        if equal_axis:
            ax.axis('equal')
        if xlim is not None:
            ax.set_xlim(*xlim)
        if ylim is not None:
            ax.set_ylim(*ylim)
        return fig
    def __call__(self, *args, **kwargs):
        return self.whiskerPlot(*args, **kwargs)
    def getData(self):
        return self.data

class WhiskerPlotStarSysTest(BaseWhiskerPlotSysTest):
    """
    A class to make WhiskerPlots of stars.
    """
    short_name = 'whiskerplot_star'
    long_name = 'Make a Whisker plot of stars'
    objects_list = ['star PSF']
    required_quantities = [('x', 'y', 'g1', 'g2', 'sigma')]

    def __call__(self, array, linewidth=0.01, scale=None, figsize=None,
                 xlim=None, ylim=None):
        if 'CCD' in array.dtype.names:
            fields = list(self.required_quantities[0]) + ['CCD']
        else:
            fields = list(self.required_quantities[0])
        self.data = numpy.rec.fromarrays([array[field] for field in fields], names=fields)
        return self.whiskerPlot(array['x'], array['y'], array['g1'], array['g2'], array['sigma'],
                                linewidth=linewidth, scale=scale, figsize=figsize,
                                xlabel=r'$x$ [pixel]', ylabel=r'$y$ [pixel]',
                                size_label=r'$\sigma$ [pixel]',
                                xlim=xlim, ylim=ylim, equal_axis=True)


class WhiskerPlotPSFSysTest(BaseWhiskerPlotSysTest):
    """
    A class to make WhiskerPlots of PSF shapes.
    """
    short_name = 'whiskerplot_psf'
    long_name = 'Make a Whisker plot of PSFs'
    objects_list = ['star PSF']
    required_quantities = [('x', 'y', 'psf_g1', 'psf_g2', 'psf_sigma')]

    def __call__(self, array, linewidth=0.01, scale=None, figsize=None,
                 xlim=None, ylim=None):
        if 'CCD' in array.dtype.names:
            fields = list(self.required_quantities[0]) + ['CCD']
        else:
            fields = list(self.required_quantities[0])
        self.data = numpy.rec.fromarrays([array[field] for field in fields], names=fields)
        return self.whiskerPlot(array['x'], array['y'], array['psf_g1'], array['psf_g2'],
                                array['psf_sigma'], linewidth=linewidth, scale=scale,
                                figsize=figsize, xlabel=r'$x$ [pixel]', ylabel=r'$y$ [pixel]',
                                size_label=r'$\sigma$ [pixel]',
                                xlim=xlim, ylim=ylim, equal_axis=True)


class WhiskerPlotResidualSysTest(BaseWhiskerPlotSysTest):
    """A class to make WhiskerPlots of the (star-PSF) residuals.
    """
    short_name = 'whiskerplot_residual'
    long_name = 'Make a Whisker plot of residuals'
    objects_list = ['star PSF']
    required_quantities = [('x', 'y', 'g1', 'g2', 'sigma', 'psf_g1', 'psf_g2', 'psf_sigma')]

    def __call__(self, array, linewidth=0.01, scale=None, figsize=None,
                 xlim=None, ylim=None):
        data = [array['x'], array['y'], array['g1'] - array['psf_g1'],
                array['g2'] - array['psf_g2'], array['sigma'] - array['psf_sigma']]
        fields = ['x', 'y', 'g1-psf_g1', 'g2-psf_g2', 'sigma-psf_sigma']
        if 'CCD' in array.dtype.names:
            data += [array['CCD']]
            fields += ['CCD']
        self.data = numpy.rec.fromarrays(data, names=fields)
        return self.whiskerPlot(array['x'], array['y'], array['g1'] - array['psf_g1'],
                                array['g2'] - array['psf_g2'], array['sigma'] - array['psf_sigma'],
                                linewidth=linewidth, scale=scale,
                                figsize=figsize, xlabel=r'$x$ [pixel]', ylabel=r'$y$ [pixel]',
                                size_label=r'$\sigma$ [pixel]',
                                xlim=xlim, ylim=ylim, equal_axis=True)

class HistogramSysTest(SysTest):
    """
    A base class for Stile systematics tests that generate histograms.

    Like the :class:`StatSysTest`, :class:`HistogramSysTest` has a number of options which can be
    set either upon initialization or at runtime.  When set at initialization, the options will hold
    for any call to the object that doesn't explicitly override them; when set during a call, the
    options will hold only for that call.

    See the documentation for the method :func:`HistoPlot` for a list of available kwargs.

    This class uses some code from the AstroML package, (c) Jake Vanderplas 2012-2013, under a
    BSD license--please see the code file for the full text of the license.
    """

    short_name = 'histogram'
    # Note: if you change the defaults here, change the docstring for the HistoPlot method.
    def __init__(self, field=None, binning_style='manual', nbins=50,
                 weights=None, limits=None, figsize=None, normed=False,
                 histtype='stepfilled', xlabel=None, ylabel=None,
                 xlim=None, ylim=None, hide_x=False, hide_y=False,
                 cumulative=False, align='mid', rwidth=0.9,
                 log=False, color='k', alpha=1.0,
                 text=None, text_x=0.90, text_y=0.90, fontsize=12,
                 linewidth=2.0, vlines=None, vcolor='k'):
        self.field = field
        self.binning_style = binning_style
        self.nbins = nbins
        self.weights = weights
        self.limits = limits
        self.figsize = figsize
        self.normed = normed
        self.histtype = histtype
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.xlim = xlim
        self.ylim = ylim
        self.hide_x = hide_x
        self.hide_y = hide_y
        self.cumulative = cumulative
        self.align = align
        self.rwidth = rwidth
        self.log = log
        self.color = color
        self.alpha = alpha
        self.text = text
        self.text_x = text_x
        self.text_y = text_y
        self.fontsize = fontsize
        self.linewidth = linewidth
        self.vlines = vlines
        self.vcolor = vcolor

    def get_param_value(self, param, ii, data_dim, multihist=False):
        if type(param) is list and multihist:
            if len(param) == data_dim:
                param_use = param[ii]
            else:
                param_use = param[0]
        elif type(param) is list:
            param_use = param[0]
        else:
            param_use = param
        return param_use

    """
    The Scott rule for bin size
    This function is directly copied from the astroML library
    (astroMl/density_estimation/histtool.py)
    with some updates to the doc style since we're not using numpydoc.
    """
    def scotts_bin_width(self, data, return_bins=False):
        r"""Return the optimal histogram bin width using Scott's rule:

        :param array-like data: observed (one-dimensional) data
        :param bool return_bins:  (optional) if True, then return the bin edges

        :returns: width(float), optimal bin width using Scott's rule; bins(ndarray), bin edges
                  returned if `return_bins` is True

        Notes:
        The optimal bin width is

        .. math::
            \Delta_b = \frac{3.5\sigma}{n^{1/3}}

        where :math:`\sigma` is the standard deviation of the data, and
        :math:`n` is the number of data points.

        See Also: knuth_bin_width; freedman_bin_width; astroML.plotting.hist
        """
        data = numpy.asarray(data)
        if data.ndim != 1:
            raise ValueError("data should be one-dimensional")

        n = data.size
        sigma = numpy.std(data)

        dx = 3.5 * sigma * 1. / (n ** (1. / 3))

        if return_bins:
            Nbins = numpy.ceil((data.max() - data.min()) * 1. / dx)
            Nbins = max(1, Nbins)
            bins = data.min() + dx * numpy.arange(Nbins + 1)
            return dx, bins
        else:
            return dx

    """
    The Freedman-Diaconis rule of bin size
    This function is directly copied from the astroML library
    (astroMl/density_estimation/histtool.py)
    with some updates to the doc style since we're not using numpydoc.
    """
    def freedman_bin_width(self, data, return_bins=False):
        r"""Return the optimal histogram bin width using the Freedman-Diaconis
            rule

        :param array-like data: observed (one-dimensional) data
        :param bool return_bins: (optional) if True, then return the bin edges

        :returns: width(float), optimal bin width using the Freedman-Diaconis rule; bins(ndarray),
                  bin edges returned if `return_bins` is True

        Notes:
        The optimal bin width is

        .. math::
            \Delta_b = \frac{2(q_{75} - q_{25})}{n^{1/3}}

        where :math:`q_{N}` is the :math:`N` percent quartile of the data, and
        :math:`n` is the number of data points.

        See Also: knuth_bin_width; scotts_bin_width; astroML.plotting.hist
        """
        data = numpy.asarray(data)
        if data.ndim != 1:
            raise ValueError("data should be one-dimensional")

        n = data.size
        if n < 4:
            raise ValueError("data should have more than three entries")

        dsorted = numpy.sort(data)
        v25 = dsorted[n / 4 - 1]
        v75 = dsorted[(3 * n) / 4 - 1]

        dx = 2 * (v75 - v25) * 1. / (n ** (1. / 3))

        if return_bins:
            Nbins = numpy.ceil((dsorted[-1] - dsorted[0]) * 1. / dx)
            Nbins = max(1, Nbins)
            bins = dsorted[0] + dx * numpy.arange(Nbins + 1)
            return dx, bins
        else:
            return dx

    """
    Generate the histogram
    """
    # All of these defaults are None because they're set in the initalization and we want to be able
    # to tell the difference between "I don't care, use the default" and "override initialization,
    # use this value". Otherwise there could be a conflict for kwargs that have non-None defaults.
    def HistoPlot(self, data_list, field=None, binning_style=None, nbins=None,
                  weights=None, limits=None, figsize=None, normed=None,
                  histtype=None, xlabel=None, ylabel=None,
                  xlim=None, ylim=None, hide_x=None, hide_y=None,
                  cumulative=None, align=None, rwidth=None,
                  log=None, color=None, alpha=None,
                  text=None, text_x=None, text_y=None, fontsize=None,
                  linewidth=None, vlines=None, vcolor=None):

        """
        Draw a histogram and return a :class:`matplotlib.figure.Figure` object.

        This method has a bunch of options for controlling the appearance of
        the histogram, which are explained below.

        :param data_list:    The 1-dimensional NumPy array or a list of Numpy arrays
                             for plotting histograms; or, a formatted array plus a `field`
                             parameter (either at class initalization or as a kwarg).
        :param field:        The field of data to be used, if data_list is a formatted array.
                             This can be iterable if multiple formatted arrays are passed to
                             data_list, but must have the same length as data_list.
                             If multiple formatted arrays are passed to data_list but only one
                             field kwarg is given, the same field will be used in every array.
        :param binning_style: Different selections of Histogram bin size.

                              - 'scott' :   Use Scott's rule to decide the bin size.
                              - 'freedman': Use the Freedman-Diaconis rule to decide the bin
                                size.
                              - 'manual' :  Manually select a fixed number of bins.

                             [default: binning_style='manual']
        :param nbins:        The number of bins if binning_style = 'manual' is selected.
                             [Default: nbins = 50]
        :param weights:      An array of weights, or True to use the 'w' column from
                             the data array. [Default: None]
        :param limits:       The [min, max] limits to trim the data before the
                             histogram is made.
                             [Default: limits = None]
        :param normed:       Whether the normalized histogram is shown.
                             [Default: normed = False]
        :param cumulative:   Whether the cumulative histogram is shown.
                             [Default: cumulative = False]
        :param histtype:     The type of histogram to show.

                             - 'bar'        : Tradition bar-type histogram.
                             - 'step'       : Unfilled lineplot-type histogram.
                             - 'stepfilled' : Filled lineplot-type histogram.

                             [Default: histtype = 'stepfilled']
        :param align:        Where the bars are centered relative to bin edges
                             = 'left', 'mid', or 'right'.
                             [Default: align = 'mid' ]
        :param rwidth:       The relative width of the bars as a fraction of the
                             bin width. Ignored for histtype = 'step' or
                             'stepfilled'.
                             [Default = None]
        :param log:          If True, the histogram axis will be set to a log scale.
                             [Default = False]
        :param color:        Color of the histogram.
                             [Default = None, which will use the standard color sequence]
        :param figsize:      Size of a figure (x, y) in units of inches..
                             [Default: None, meaning use the default value of matplotlib]
        :param xlabel:       The x-axis label.
                             [Default: None, meaning do not show a label for the x-axis]
        :param ylabel:       The y-axis label.
                             [Default: None, meaning do not show a label for the y-axis]
        :param xlim:         Limits of x-axis (min, max).
                             [Default: None, meaning do not set any limits for x]
        :param ylim:         Limits of y-axis (min, max).
                             [Default: None, meaning do not set any limits for y]
        :param hide_x:       Whether hide the labels for x-axis.
                             [Default: hide_x = False]
        :param hide_y:       Whether hide the labels for y-axis.
                             [Default: hide_y = False]
        :param alpha:        0.0 transparent through 1.0 opaque
                             [Default: alpha = 1.0]
        :param linewidth:    With of the vertical lines
                             [Default: linewidth = 2.0]
        :param text:         Text to put on the figure
                             [Default: None]
        :param text_x:       The X-coordinate of the text on the plot
                             [Default: text_x = 0.9]
        :param text_y:       The Y-coordinate of the test on the plot
                             [Default: text_y = 0.9]
        :param fontsize:     Font size of the text
                             [Default: fontsize = 12]
        :param vlines:       Locations to plot vertical lines to indicate interesting
                             values.
                             [Default: None]
        :param vcolor:       Color or a list of color for vertical lines to plot.
                             [Default: 'k']

        :returns: a :class:`matplotlib.figure.Figure` object.
        """

        # Get defaults from the class attributes if necessary
        for key_name in ['field', 'binning_style', 'nbins', 'weights', 'limits', 'figsize',
                         'normed', 'histtype', 'xlabel', 'ylabel', 'xlim', 'ylim', 'hide_x',
                         'hide_y', 'cumulative', 'align', 'rwidth', 'log', 'color', 'alpha', 'text',
                         'text_x', 'text_y', 'fontsize', 'linewidth', 'vlines', 'vcolor']:
            exec('if %s is None: %s = self.%s'%(key_name, key_name, key_name))

        ## Define the plot
        hist = plt.figure(figsize=figsize)
        ax   = hist.add_subplot(1, 1, 1)

        data_dim = len(data_list)
        for ii in range(data_dim):

            if type(data_list[0]) is list or type(data_list[0]) is numpy.ndarray:
                multihist = True
                data = data_list[ii]
            else:
                multihist = False
                data = data_list

            if field is not None:
                if not isinstance(field, str) and hasattr(field, '__iter__'):
                    if len(field)!=data_dim or not multihist:
                        raise RuntimeError('Different length lists of data & lists of fields!')
                    data = data[field][ii]
                else:
                    data = data[field]


            # mask data with NaN
            data = data[numpy.isnan(data) == False]
            data = numpy.asarray(data)

            # trim the data if necessary
            if limits is not None:
                data = data[(data >= limits[0]) & (data <= limits[1])]

            # decide which bin style to use
            style_use = self.get_param_value(binning_style, ii, data_dim,
                                             multihist=multihist)

            # now support constant bin size, Scott rule, and Freedman rule
            if style_use in ['scott', 'freedman', 'manual']:
                if (style_use is 'scott'):
                    "Use the Scott rule"
                    dx, bins = self.scotts_bin_width(data, True)
                elif style_use is 'freedman':
                    "Use the Freedman rule"
                    dx, bins = self.freedman_bin_width(data, True)
                elif style_use is 'manual':
                    bins = nbins
            else:
                print("Unrecognized code for binning style, use default instead!")
                bins = nbins

            if weights is True:
                weights = data['w']

            # decide if weight is presented
            if weights is not None and multihist:
                if len(weights) == data_dim:
                    weight_use = weights[ii]
                else:
                    import warnings
                    warnings.warn("Inconsistent shape between data and weights! No weight is used!")
                    weight_use = None
            elif weights is not None:
                if len(weights) == len(data):
                    weight_use = weights
                else:
                    import warnings
                    warnings.warn("Inconsistent shape between data and weights! No weight is used!")
                    weight_use = None
            else:
                import warnings
                warnings.warn("The format of given weights cannot be understood! No weight is used!")
                weight_use = None

            # decide which histtype to use
            hist_use = self.get_param_value(histtype, ii, data_dim,
                                            multihist=multihist)
            # the color of the histogram
            color_use = self.get_param_value(color, ii, data_dim,
                                             multihist=multihist)
            # the transparency of the histogram
            alpha_use = self.get_param_value(alpha, ii, data_dim,
                                             multihist=multihist)
            # the relative width of the bar
            rwidth_use = self.get_param_value(rwidth, ii, data_dim,
                                              multihist=multihist)
            # the width of the vertical line
            lwidth_use = self.get_param_value(linewidth, ii, data_dim,
                                              multihist=multihist)

            # make the histogram
            counts, edges, patches = ax.hist(data, bins,
                                             weights = weight_use,
                                             histtype = hist_use,
                                             color = color_use,
                                             normed = normed,
                                             cumulative = cumulative,
                                             alpha = alpha_use,
                                             rwidth = rwidth_use,
                                             log = log,
                                             align = align,
                                             linewidth = lwidth_use
                                            )

            # outline the filled region
            if hist_use is 'stepfilled':
                counts, edges, patches = ax.hist(data, bins,
                                                 weights = weight_use,
                                                 histtype = 'step',
                                                 color = 'k',
                                                 normed = normed,
                                                 cumulative = cumulative,
                                                 alpha = 1.0,
                                                 rwidth = rwidth_use,
                                                 log = log,
                                                 align = align,
                                                 linewidth = 1.0
                                                )

            ymin, ymax = ax.get_ylim()

            if not multihist:
                break

        # add the text when necessary
        if text is not None:
            ax.text(text_x, text_y, text, transform=ax.transAxes,
                    ha='right', va='top', fontsize=fontsize)

        # add vertical lines when necessary
        if vlines is not None and not hasattr(vlines, '__iter__'):
            for jj in range(len(vlines)):
                vline_use = vlines[jj]

                if type(vcolor) == list:
                    if len(vcolor) == len(vlines):
                        vcolor_use = vcolor[jj]
                    else:
                        vcolor_use = vcolor[0]
                else:
                    vcolor_use = vcolor

                ax.vlines(vline_use, ymin, ymax, colors=vcolor_use,
                          linestyle='dashed',
                          linewidth=1.8)

        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)

        if xlim is not None:
            ax.set_xlim(*xlim)
        if ylim is not None:
            ax.set_ylim(*ylim)

        if hide_x:
            ax.xaxis.set_major_formatter(plt.NullFormatter())
        if hide_y:
            ax.yaxis.set_major_formatter(plt.NullFormatter())

        return hist
    def __call__(self, *args, **kwargs):
        return self.HistoPlot(*args, **kwargs)

def ScatterPlotSysTest(type=None):
    """
    Initialize an instance of a :class:`BaseScatterPlotSysTest` class, based on the ``type`` kwarg
    given. Options are:

        - **StarVsPSFG1**: star vs PSF g1
        - **StarVsPSFG2**: star vs PSF g2
        - **StarVsPSFSigma**: star vs PSF sigma
        - **ResidualVsPSFG1**: (star - PSF) g1 vs PSF g1
        - **ResidualVsPSFG2**: (star - PSF) g2 vs PSF g2
        - **ResidualVsPSFSigma**: (star - PSF) sigma vs PSF sigma
        - **ResidualSigmaVsPSFMag**: (star - PSF)/PSF sigma vs PSF magnitude
        - **None**: an empty :class:`BaseScatterPlotSysTest` class instance, which can be used for
          multiple types of scatter plots.  See the documentation
          for :class:`BaseScatterPlotSysTest` (especially the method
          :func:`scatterPlot <BaseScatterPlotSysTest.scatterPlot>`) for more details.  Note that
          this type has a different call signature than the other methods and that it lacks many of
          the convenience variables the other ScatterPlots have, such as self.objects_list and
          self.required_quantities.
    """
    if type=='StarVsPSFG1':
        return ScatterPlotStarVsPSFG1SysTest()
    elif type=='StarVsPSFG2':
        return ScatterPlotStarVsPSFG2SysTest()
    elif type=='StarVsPSFSigma':
        return ScatterPlotStarVsPSFSigmaSysTest()
    elif type=='ResidualVsPSFG1':
        return ScatterPlotResidualVsPSFG1SysTest()
    elif type=='ResidualVsPSFG2':
        return ScatterPlotResidualVsPSFG2SysTest()
    elif type=='ResidualVsPSFSigma':
        return ScatterPlotResidualVsPSFSigmaSysTest()
    elif type is None:
        return BaseScatterPlotSysTest()
    else:
        raise ValueError('Unknown scatter plot type %s given to type kwarg'%type)

class BaseScatterPlotSysTest(SysTest):
    """
    A base class for Stile systematics tests that generate scatter plots. This implements the class
    method :func:`scatterPlot` and a :func:`__call__` method that sets up the data.
    Child classes should use :func:`super()` to access this call method; see examples in the
    existing code base.
    """
    short_name = 'scatterplot'
    def __call__(self, array, x_field, y_field, yerr_field, z_field=None, residual=False,
                 per_ccd_stat=None, xlabel=None, ylabel=None, zlabel=None, color="",
                 lim=None, equal_axis=False, linear_regression=False, reference_line=None,
                 histogram=False, histogram_n_bins=40, histogram_cmap='Blues'):
        """
        Draw a scatter plot and return a :class:`matplotlib.figure.Figure` object.
        This method has a bunch of options for controlling appearance of a plot, which is
        explained below.

        :param array:           A structured NumPy array which contains data to be plotted.
        :param x_field:         The name of the field in ``array`` to be used for x.
        :param y_field:         The name of the field in ``array`` to be used for y.
        :param yerr_field:      The name of the field in ``array`` to be used for y error.
        :param z_field:         The name of the field in ``array`` to be used for z, which appears
                                as the colors of scattered points.
                                [default: None, meaning there is no additional quantity]
        :param residual:        Show residual between x and y on the y-axis.
                                [default: False, meaning y value itself is on the y-axis]
        :param per_ccd_stat:    Which statistics (median, mean, or None) to be calculated within
                                each CCD.
                                [default: None, meaning no statistics are calculated]
        :param xlabel:          The label for the x-axis.
                                [default: None, meaning do not show a label on the x-axis]
        :param ylabel:          The label for the y-axis.
                                [default: None, meaning do not show a label on the y-axis]
        :param zlabel:          The label for the z values which appears at the side of the color
                                bar.
                                [default: None, meaning do not show a label of z values]
        :param color:           The color of scattered points. This color is also applied to
                                linear regression if argument ``linear_regression`` is True. This
                                parameter is ignored when z is not None. In this case, the
                                color of linear regression is set to blue.
                                [default: None, meaning follow a matplotlib's default color]
        :param lim:             The limit of the axes. This can be specified explicitly by
                                using tuples such as ``((xmin, xmax), (ymin, ymax))``.
                                If one passes float ``p``, this routine calculates the p%-percentile
                                around the median for each axis.
                                [default: None, meaning do not set any limits]
        :param equal_axis:      If True, force ticks of the x-axis and y-axis equal to each other.
                                [default: False]
        :param linear_regression: If True, perform linear regression for x and y and plot a
                                regression line. If yerr is not None, perform the linear
                                regression incorporating the error into the standard chi^2
                                and plot a regression line with a 1-sigma allowed region.
                                [default: False]
        :param reference_line:  Draw a reference line. If ``reference_line == 'one-to-one'``,
                                ``x=y`` is drawn. If ``reference_line == 'zero'``, ``y=0`` is drawn.
                                A user-specific function can be used by passing an object which
                                has an attribute :func:`__call__` and returns a 1-d Numpy array.
        :param histogram:       Plot a 2-d histogram (instead of plotting each point individually)
                                for crowded plots. Setting `histogram=True` will cause any data
                                given for `z` to be ignored, and any uncertainty on `y` will not be
                                plotted (though it will still be used to compute a trendline).
                                [default: False]
        :param histogram_n_bins: The number of bins along *each* axis for the histogram; ignored if
                                `histogram=False`. [default: 40] 
        :param histogram_cmap:   The matplotlib colormap used for the histogram; ignored if
                                `histogram=False`. [default: 'Blues'] 
        :returns:               a :class:`matplotlib.figure.Figure` object
        """
        if per_ccd_stat:
            if z_field is None:
                z = None
                x, y, yerr = self.getStatisticsPerCCD(array['CCD'], array[x_field],
                                                      array[y_field], yerr=array[yerr_field],
                                                      stat=per_ccd_stat)
                self.data = numpy.rec.fromarrays([list(set(array['CCD'])), x,
                                                  y, yerr],
                                                 names=['ccd',
                                                        x_field,
                                                        y_field,
                                                        yerr_field])
            else:
                x, y, yerr, z = self.getStatisticsPerCCD(array['CCD'], array[x_field],
                                                      array[y_field], yerr=array[yerr_field],
                                                      z=array[z_field], stat=per_ccd_stat)
                self.data = numpy.rec.fromarrays([list(set(array['CCD'])), x,
                                                  y, yerr, zz],
                                                 names=['ccd',
                                                        x_field,
                                                        y_field,
                                                        yerr_field,
                                                        z_field])
        else:
            if z_field is None:
                z = None
                x, y, yerr = array[x_field], array[y_field], array[yerr_field]
                self.data = numpy.rec.fromarrays([x, y, yerr],
                                                 names=[x_field,
                                                        y_field,
                                                        yerr_field])
            else:
                x, y, yerr, z = array[x_field], array[y_field], array[yerr_field], array[z_field]
                self.data = numpy.rec.fromarrays([x, y, yerr, z],
                                                 names=[x_field,
                                                        y_field,
                                                        yerr_field,
                                                        z_field])
        y = y-x if residual else y
        return self.scatterPlot(x, y, yerr, z,
                                xlabel=xlabel, ylabel=ylabel,
                                color=color, lim=lim, equal_axis=False,
                                linear_regression=True, reference_line=reference_line,
                                histogram=histogram, histogram_n_bins=histogram_n_bins,
                                histogram_cmap=histogram_cmap)

    def getData(self):
        """
        Returns data used for scatter plot.

        :returns: :func:`stile_utils.FormatArray <stile.stile_utils.FormatArray>` object
        """

        return self.data

    def scatterPlot(self, x, y, yerr=None, z=None, xlabel=None, ylabel=None, zlabel=None, color="",
                    lim=None, equal_axis=False, linear_regression=False, reference_line=None,
                    histogram=False, histogram_n_bins=40, histogram_cmap='Blues'):
        """
        Draw a scatter plot and return a :class:`matplotlib.figure.Figure` object.
        This method has a bunch of options for controlling appearance of a plot, which is
        explained below.

        :param x:               The tuple, list, or NumPy array for x-axis.
        :param y:               The tuple, list, or NumPy array for y-axis.
        :param yerr:            The tuple, list, or Numpy array for error of the y values.
                                [default: None, meaning do not plot an error]
        :param z:               The tuple, list, or Numpy array for an additional quantity
                                which appears as colors of scattered points.
                                [default: None, meaning there is no additional quantity]
        :param xlabel:          The label of x-axis.
                                [default: None, meaning do not show a label of x-axis]
        :param ylabel:          The label of y-axis.
                                [default: None, meaning do not show a label of y-axis]
        :param zlabel:          The label of z values which appears at the side of color bar.
                                [default: None, meaning do not show a label of z values]
        :param color:           The color of scattered points. This color is also applied to linear
                                regression if argument ``linear_regression`` is True. This parameter
                                is ignored when z is not None. In this case, the color of linear
                                regression is set to blue.
                                [default: None, meaning follow a matplotlib's default color]
        :param lim:             The limit of axis. This can be specified explicitly by
                                using tuples such as ``((xmin, xmax), (ymin, ymax))``.
                                If one passes float p, it calculate p%-percentile around median
                                for each of x-axis and y-axis.
                                [default: None, meaning do not set any limits]
        :param equal_axis:      If True, force ticks of the x-axis and y-axis equal to each other.
                                [default: False]
        :param linear_regression: If True, perform linear regression for x and y and plot a
                                regression line. If yerr is not None, perform the linear
                                regression incorporating the error into the standard chi^2
                                and plot a regression line with a 1-sigma allowed region.
        :param reference_line:  Draw a reference line. If ``reference_line == 'one-to-one'``,
                                ``x=y`` is drawn. If ``reference_line == 'zero'``, ``y=0`` is drawn.
                                A user-specific function can be used by passing an object which has
                                an attribute :func:`__call__` and returns a 1-d Numpy array.
                                [default: False]
        :param histogram:       Plot a 2-d histogram (instead of plotting each point individually)
                                for crowded plots. Setting `histogram=True` will cause any data
                                given for `z` to be ignored, and any uncertainty on `y` will not be
                                plotted (though it will still be used to compute a trendline).
                                [default: False]
        :param histogram_n_bins: The number of bins along *each* axis for the histogram; ignored if
                                `histogram=False`. [default: 40] 
        :param histogram_cmap:   The matplotlib colormap used for the histogram; ignored if
                                `histogram=False`. [default: 'Blues'] 
        :returns:                a :class:`matplotlib.figure.Figure` object
        """
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        print histogram, "histogram"
        # mask data with nan. Emit a warning if an array has nan in it.
        x_isnan = numpy.isnan(x)
        y_isnan = numpy.isnan(y)
        import warnings
        if numpy.sum(x_isnan) != 0:
            warnings.warn('There are %s nans in x, out of %s.' % (numpy.sum(x_isnan), len(x_isnan)))
        if numpy.sum(y_isnan) != 0:
            warnings.warn('There are %s nans in y, out of %s.' % (numpy.sum(y_isnan), len(y_isnan)))
        sel = numpy.logical_and(numpy.invert(x_isnan), numpy.invert(y_isnan))
        if yerr is not None:
            yerr_isnan = numpy.isnan(yerr)
            if numpy.sum(yerr_isnan) != 0:
                warnings.warn('There are %s nans in yerr, out of %s.'
                             % (numpy.sum(yerr_isnan), len(yerr_isnan)))
            sel = numpy.logical_and(sel, numpy.invert(yerr_isnan))
        if z is not None:
            if histogram:
                warnings.warn('Plotting a histogram - z (color) data will be ignored.')
            z_isnan = numpy.isnan(z)
            if numpy.sum(z_isnan) != 0:
                warnings.warn('There are %s nans in z, out of %s.'
                             % (numpy.sum(z_isnan), len(z_isnan)))
            sel = numpy.logical_and(sel, numpy.invert(z_isnan))
        x = x[sel]
        y = y[sel]
        yerr = yerr[sel] if yerr is not None else None
        z = z[sel] if z is not None else None

        # load axis limits if argument lim is ((xmin, xmax), (ymin, ymax))
        if isinstance(lim, tuple):
            xlim = lim[0]
            ylim = lim[1]

        # calculate n-sigma limits around mean if lim is float
        elif isinstance(lim, float):
            p = lim
            xlim = (numpy.percentile(x, 50.-0.5*p), numpy.percentile(x, 50.+0.5*p))
            ylim = (numpy.percentile(y, 50.-0.5*p), numpy.percentile(y, 50.+0.5*p))
        # in other cases (except for the default value None), raise an exception
        elif lim is not None:
            raise TypeError('lim should be ((xmin, xmax), (ymin, ymax)) or'
                            '`float` to indicate p%-percentile around median.')
        else:
            # Even if lim = None, we want to set limits. Limits set by matplotlib looks uneven
            # probably because it seems to pick round numbers for the endpoints (eg -0.2 and 0.2).
            xlim = (numpy.min(x)-0.05*(numpy.max(x)-numpy.min(x)),
                    numpy.max(x)+0.05*(numpy.max(x)-numpy.min(x)))
            # We apply the same thing to y. However, when y has error, setting the limit may cut out
            # error, so we just leave it.
            if yerr is None:
                ylim = (numpy.min(y)-0.05*(numpy.max(y)-numpy.min(y)),
                        numpy.max(y)+0.05*(numpy.max(y)-numpy.min(y)))
            else:
                ylim = None

        # plot
        if not histogram:
            if z is None:
                if yerr is None:
                    p = ax.plot(x, y, ".%s" % color)
                else:
                    p = ax.errorbar(x, y, yerr, fmt=".%s" % color)
                # store color for latter use
                used_color = p[0].get_color()
            else:
                if yerr is not None:
                    plt.errorbar(x, y, yerr=yerr, linestyle="None", color="k", zorder=0)
                plt.scatter(x, y, c=z, zorder=1)
                cb = plt.colorbar()
                used_color = "b"
        else:
            plt.hist2d(x, y, bins=histogram_n_bins, cmap=histogram_cmap)
            cb = plt.colorbar()
            used_color = color

        # make axes ticks equal to each other if specified
        if equal_axis:
            ax.axis("equal")

        # set axis limits if specified
        if xlim is not None:
            ax.set_xlim(*xlim)
        if ylim is not None:
            ax.set_ylim(*ylim)

        # set up x value for linear regression or a reference line
        if linear_regression or reference_line is not None:
            # set x limits of a regression line.
            # If equal_axis is False, just use x limits set to axis.
            if not equal_axis:
                xlimtmp = ax.get_xlim()
            # If equal_axis is True, x limits may not reflect an actual limit of a plot,e.g., if
            # y limits are wider than x limits, an actual limit along the x-axis becomes wider
            # than what we specified although a value tied to a matplotlib.axes object remains
            # the same, which can result in a regression line truncated in smaller range along
            # x-axis if we simply use ax.get_xlim() to the regression line. To avoid this,
            # take a wider range between x limits and y limits, and set this range to
            # the x limit of a regression line.
            else:
                d = numpy.max([ax.get_xlim()[1] - ax.get_xlim()[0],
                               ax.get_ylim()[1] - ax.get_ylim()[0]])
                xlimtmp = [numpy.average(x)-0.5*d, numpy.average(x)+0.5*d]
            xtmp = numpy.linspace(*xlimtmp)

        # perform linear regression if specified
        if linear_regression:
            if yerr is None:
                m, c = self.linearRegression(x, y)
                ax.plot(xtmp, m*xtmp+c, "--%s" % used_color)
            else:
                m, c, cov_m, cov_c, cov_mc = self.linearRegression(x, y, err=yerr)
                ax.plot(xtmp, m*xtmp+c, "--%s" % used_color)
                y = m*xtmp+c
                # calculate yerr using the covariance
                yerr = numpy.sqrt(xtmp**2*cov_m + 2.*xtmp*cov_mc + cov_c)
                ax.fill_between(xtmp, y-yerr, y+yerr, facecolor=used_color,
                                edgecolor=used_color, alpha=0.5)
                ax.annotate(r"$m=%.4f\pm%.4f$" %(m, numpy.sqrt(cov_m))+"\n"+
                            r"$c=%.4f\pm%.4f$" %(c, numpy.sqrt(cov_c)), xy=(0.75, 0.05),
                            xycoords='axes fraction')

        # draw a reference line
        if reference_line is not None:
            if isinstance(reference_line, str):
                if reference_line == "one-to-one":
                    ax.plot(xtmp, xtmp, "--k")
                elif reference_line == "zero":
                    ax.plot(xtmp, numpy.zeros(xtmp.shape), "--k")
            elif hasattr(reference_line, '__call__'):
                y = reference_line(xtmp)
                if len(numpy.array(y).shape) != 1 or len(y) != len(xtmp):
                    raise TypeError('an object for reference_line should return a 1-d array whose'
                                    'size is the same as input')
                ax.plot(xtmp, y, "--k")
            else:
                raise TypeError("reference_line should be str 'one-to-one' or 'zero',"
                                "or an object which has atttibute '__call__'.")

        # set axis labels if specified
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        if zlabel is not None and not histogram:
            cb.set_label(zlabel)

        fig.tight_layout()

        return fig

    def linearRegression(self, x, y, err=None):
        """
        Perform linear regression (y=mx+c). If error is given, it returns covariance.

        :param x:               NumPy array for x.
        :param y:               NumPy array for y.
        :param err:             Numpy array for y error.
                                [default: None, meaning do not consider y error]
        :returns:               m, c. If err is not None, m, c, cov_m, cov_c, cov_mc.
        """

        e = numpy.ones(x.shape) if err is None else err
        S = numpy.sum(1./e**2)
        Sx = numpy.sum(x/e**2)
        Sy = numpy.sum(y/e**2)
        Sxx = numpy.sum(x**2/e**2)
        Sxy = numpy.sum(x*y/e**2)
        Delta = S*Sxx - Sx**2
        m = (S*Sxy-Sx*Sy)/Delta
        c = (Sxx*Sy-Sx*Sxy)/Delta
        if err is None:
            return m, c
        else:
            cov_m = S/Delta
            cov_c = Sxx/Delta
            cov_mc = -Sx/Delta
            return m, c, cov_m, cov_c, cov_mc

    def getStatisticsPerCCD(self, ccds, x, y, yerr=None, z=None, stat="median"):
        """
        Calculate median or mean for x and y (and z if specified) for each ccd.

        :param ccds:       NumPy array for ccds, an array in which each element indicates
                           ccd id of each data point.
        :param x:          NumPy array for x.
        :param y:          NumPy array for y.
        :param yerr:       Numpy array for y error.
                           [default: None, meaning do not consider y error]
        :param z:          NumPy array for z.
                           [default: None, meaning do not statistics for z]
        :returns:          x_ave, y_ave, y_ave_std.
        """
        if stat == "mean":
            x_ave = numpy.array([numpy.average(x[ccds == ccd])
                                 for ccd in set(ccds)])
            if yerr is None:
                y_ave = numpy.array([numpy.average(y[ccds == ccd])
                                     for ccd in set(ccds)])
                y_ave_std = numpy.array([numpy.std(y[ccds == ccd])/
                                         numpy.sqrt(len(y[ccds == ccd]))
                                         for ccd in set(ccds)])
            # calculate y and its std under the inverse variance weight if yerr is given
            else:
                y_ave = numpy.array([numpy.sum(y[ccds == ccd]/
                                               yerr[ccds == ccd]**2)/
                                     numpy.sum(1./yerr[ccds == ccd]**2)
                                     for ccd in set(ccds)])
                y_ave_std = numpy.array([numpy.sqrt(1./numpy.sum(1./yerr[ccds == ccd]**2
                                                                 ))
                                         for ccd in set(ccds)])
            if z is not None:
                z_ave = numpy.array([numpy.average(z[ccds == ccd])
                                     for ccd in set(ccds)])
                return x_ave, y_ave, y_ave_std, z_ave
            else:
                return x_ave, y_ave, y_ave_std
        elif stat == "median":
            x_med = numpy.array([numpy.median(x[ccds == ccd])
                                 for ccd in set(ccds)])
            y_med = numpy.array([numpy.median(y[ccds == ccd])
                                 for ccd in set(ccds)])
            y_med_std = numpy.array([numpy.sqrt(numpy.pi/2.)*numpy.std(y[ccds == ccd])
                                     /numpy.sqrt(len(y[ccds == ccd]))
                                     for ccd in set(ccds)])
            if z is not None:
                z_med = numpy.array([numpy.median(z[ccds == ccd])
                                     for ccd in set(ccds)])
                return x_med, y_med, y_med_std, z_med
            else:
                return x_med, y_med, y_med_std
        else:
            raise ValueError('stat should be mean or median.')

class ScatterPlotStarVsPSFG1SysTest(BaseScatterPlotSysTest):
    """
    A class to make ScatterPlots of star vs PSF g1 values
    """
    short_name = 'scatterplot_star_vs_psf_g1'
    long_name = 'Make a scatter plot of star g1 vs psf g1'
    objects_list = ['star PSF']
    required_quantities = [('g1', 'g1_err', 'psf_g1')]

    def __call__(self, array, per_ccd_stat=None, color='', lim=None,
                    histogram=False, histogram_n_bins=40, histogram_cmap='Blues'):
        print "lalala", histogram
        return super(ScatterPlotStarVsPSFG1SysTest,
                     self).__call__(array, 'psf_g1', 'g1', 'g1_err', residual=False,
                                    per_ccd_stat=per_ccd_stat, xlabel=r'$g^{\rm PSF}_1$',
                                    ylabel=r'$g^{\rm star}_1$', color=color, lim=lim,
                                    equal_axis=False, linear_regression=True,
                                    reference_line='one-to-one',
                                    histogram=histogram, histogram_n_bins=histogram_n_bins,
                                    histogram_cmap=histogram_cmap)


class ScatterPlotStarVsPSFG2SysTest(BaseScatterPlotSysTest):
    """
    A class to make ScatterPlots of star vs PSF g2 values
    """
    short_name = 'scatterplot_star_vs_psf_g2'
    long_name = 'Make a scatter plot of star g2 vs psf g2'
    objects_list = ['star PSF']
    required_quantities = [('g2', 'g2_err', 'psf_g2')]

    def __call__(self, array, per_ccd_stat=None, color='', lim=None,
                    histogram=False, histogram_n_bins=40, histogram_cmap='Blues'):
        return super(ScatterPlotStarVsPSFG2SysTest,
                     self).__call__(array, 'psf_g2', 'g2', 'g2_err', residual=False,
                                    per_ccd_stat=per_ccd_stat, xlabel=r'$g^{\rm PSF}_2$',
                                    ylabel=r'$g^{\rm star}_2$', color=color, lim=lim,
                                    equal_axis=False, linear_regression=True,
                                    reference_line='one-to-one',
                                    histogram=histogram, histogram_n_bins=histogram_n_bins,
                                    histogram_cmap=histogram_cmap)


class ScatterPlotStarVsPSFSigmaSysTest(BaseScatterPlotSysTest):
    """
    A class to make ScatterPlots of star vs PSF sigma values
    """
    short_name = 'scatterplot_star_vs_psf_sigma'
    long_name = 'Make a scatter plot of star sigma vs psf sigma'
    objects_list = ['star PSF']
    required_quantities = [('sigma', 'sigma_err', 'psf_sigma')]

    def __call__(self, array, per_ccd_stat=None, color='', lim=None,
                    histogram=False, histogram_n_bins=40, histogram_cmap='Blues'):
        return super(ScatterPlotStarVsPSFSigmaSysTest,
                     self).__call__(array, 'psf_sigma', 'sigma', 'sigma_err', residual=False,
                                    per_ccd_stat=per_ccd_stat,
                                    xlabel=r'$\sigma^{\rm PSF}$ [arcsec]',
                                    ylabel=r'$\sigma^{\rm star}$ [arcsec]',
                                    color=color, lim=lim, equal_axis=False,
                                    linear_regression=True, reference_line='one-to-one',
                                    histogram=histogram, histogram_n_bins=histogram_n_bins,
                                    histogram_cmap=histogram_cmap)


class ScatterPlotResidualVsPSFG1SysTest(BaseScatterPlotSysTest):
    """
    A class to make ScatterPlots of (star-PSF) residual vs PSF g1 values
    """
    short_name = 'scatterplot_residual_vs_psf_g1'
    long_name = 'Make a scatter plot of residual g1 vs psf g1'
    objects_list = ['star PSF']
    required_quantities = [('g1', 'g1_err', 'psf_g1')]

    def __call__(self, array, per_ccd_stat=None, color='', lim=None,
                    histogram=False, histogram_n_bins=40, histogram_cmap='Blues'):
        return super(ScatterPlotResidualVsPSFG1SysTest,
                     self).__call__(array, 'psf_g1', 'g1', 'g1_err', residual=True,
                                    per_ccd_stat=per_ccd_stat, xlabel=r'$g^{\rm PSF}_1$',
                                    ylabel=r'$g^{\rm star}_1 - g^{\rm PSF}_1$',
                                    color=color, lim=lim, equal_axis=False,
                                    linear_regression=True, reference_line='zero',
                                    histogram=histogram, histogram_n_bins=histogram_n_bins,
                                    histogram_cmap=histogram_cmap)


class ScatterPlotResidualVsPSFG2SysTest(BaseScatterPlotSysTest):
    """
    A class to make ScatterPlots of (star-PSF) residual vs PSF g2 values
    """
    short_name = 'scatterplot_residual_vs_psf_g2'
    long_name = 'Make a scatter plot of residual g2 vs psf g2'
    objects_list = ['star PSF']
    required_quantities = [('g2', 'g2_err', 'psf_g2')]

    def __call__(self, array, per_ccd_stat=None, color='', lim=None,
                    histogram=False, histogram_n_bins=40, histogram_cmap='Blues'):
        return super(ScatterPlotResidualVsPSFG2SysTest,
                     self).__call__(array, 'psf_g2', 'g2', 'g2_err', residual=True,
                                    per_ccd_stat=per_ccd_stat, xlabel=r'$g^{\rm PSF}_2$',
                                    ylabel=r'$g^{\rm star}_2 - g^{\rm PSF}_2$',
                                    color=color, lim=lim, equal_axis=False,
                                    linear_regression=True, reference_line='zero',
                                    histogram=histogram, histogram_n_bins=histogram_n_bins,
                                    histogram_cmap=histogram_cmap)


class ScatterPlotResidualVsPSFSigmaSysTest(BaseScatterPlotSysTest):
    """
    A class to make ScatterPlots of (star-PSF) residual vs PSF sigma values
    """
    short_name = 'scatterplot_residual_vs_psf_sigma'
    long_name = 'Make a scatter plot of residual sigma vs psf sigma'
    objects_list = ['star PSF']
    required_quantities = [('sigma', 'sigma_err', 'psf_sigma')]

    def __call__(self, array, per_ccd_stat=None, color='', lim=None,
                    histogram=False, histogram_n_bins=40, histogram_cmap='Blues'):
        return super(ScatterPlotResidualVsPSFSigmaSysTest,
                     self).__call__(array, 'psf_sigma', 'sigma', 'sigma_err', residual=True,
                                    per_ccd_stat=per_ccd_stat,
                                    xlabel=r'$\sigma^{\rm PSF}$ [arcsec]',
                                    ylabel=r'$\sigma^{\rm star} - \sigma^{\rm PSF}$ [arcsec]',
                                    color=color, lim=lim, equal_axis=False,
                                    linear_regression=True, reference_line='zero',
                                    histogram=histogram, histogram_n_bins=histogram_n_bins,
                                    histogram_cmap=histogram_cmap)


class ScatterPlotResidualSigmaVsPSFMagSysTest(BaseScatterPlotSysTest):
    """
    A class to make ScatterPlots of (star-PSF) residual sigma vs PSF magnitude
    """
    short_name = 'scatterplot_residual_sigma_vs_psf_magnitude'
    long_name = 'Make a scatter plot of residual sigma vs PSF magnitude'
    objects_list = ['star PSF']
    required_quantities = [('sigma', 'sigma_err', 'psf_sigma', 'mag_inst')]

    def __call__(self, array, per_ccd_stat='None', color='', lim=None,
                    histogram=False, histogram_n_bins=40, histogram_cmap='Blues'):
        self.per_ccd_stat = None if per_ccd_stat == 'None' else per_ccd_stat
        import numpy.lib.recfunctions
        use_array = numpy.copy(array)
        use_array = numpy.lib.recfunctions.append_fields(use_array, 'sigma_residual_frac',
                                                         (use_array['sigma'] -
                                                          use_array['psf_sigma'])
                                                         /use_array['psf_sigma'])
        use_array = numpy.lib.recfunctions.append_fields(use_array, 'sigma_residual_frac_err',
                                                         use_array['sigma_err']
                                                         /use_array['psf_sigma'])
        return super(ScatterPlotResidualSigmaVsPSFMagSysTest,
                     self).__call__(use_array, 'mag_inst', 'sigma_residual_frac',
                                    'sigma_residual_frac_err', residual=False,
                                    per_ccd_stat=self.per_ccd_stat,
                                    xlabel=r'Instrumental PSF magnitude',
                                    ylabel=
                                    r'$(\sigma^{\rm star} - \sigma^{\rm PSF})/\sigma^{\rm PSF}$',
                                    color=color, lim=lim, equal_axis=False,
                                    linear_regression=True, reference_line='zero',
                                    histogram=histogram, histogram_n_bins=histogram_n_bins,
                                    histogram_cmap=histogram_cmap)

