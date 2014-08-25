"""@file sys_tests.py
Contains the class definitions of the Stile systematics tests.
"""
import numpy
import stile
import treecorr
from treecorr.corr2 import corr2_valid_params
try:
    import matplotlib
    # We should decide which backend to use (this line allows running matplotlib even on sessions
    # without properly defined displays, eg through PBS)
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    has_matplotlib = True
except ImportError:
    has_matplotlib = False

# Silly class so we can call savefig() on something returned from a plot() class that doesn't
# actually do anything.
class PlotNone(object):
    def savefig(self,filename):
        pass
    
class SysTest:
    """
    A SysTest is a lensing systematics test of some sort.  It should define the following 
    attributes:
        short_name: a string that can be used in filenames to denote this systematics test
        long_name: a string to denote this systematics test within program text outputs
        objects_list: a list of objects that the test should operate on.  We expect these objects
            to be from the list:
            ['galaxy', 'star',  # all such objects,
             'galaxy lens',     # only galaxies to be used as lenses in galaxy-galaxy lensing tests,
             'star PSF',        # stars used in PSF determination,
             'star bright',     # especially bright stars,
             'galaxy random',   # random catalogs with the same spatial distribution as the 
             'star random']     # 'galaxy' or 'star' samples.
        required_quantities: a list of tuples.  Each tuple is the list of fields/quantities that 
            should be given for the corresponding object from the objects_list.  We expect the
            quantities to be from the list:
            ['ra', 'dec',       # Position on the sky
             'x', 'y',          # Position in CCD/detector coordinates (or any flat projection)
             'g1', g2', 'g1_err', 'g2'_err', # Two components of shear and their errors
             'sigma', 'sigma_err', # Object size and its error
             'w',               # Per-object weight
             'psf_g1', 'psf_g2', 'psf_sigma'] # PSF shear and size at the object location
    
    It should define the following methods:
        __call__(self, ...) = run the SysTest. There are two typical call signatures for SysTests:
            __call__(self,data[,data2],**kwargs): run a test on a set of data, or a test involving 
                two data sets data and data2.
            __call__(self,stile_args_dict,data=None,data2=None,random=None,random2=None,**kwargs):
                the call signature for the CorrelationFunctionSysTests, which leave the data as 
                kwargs because the CorrelationFunctionSysTests() can also take filenames as kwargs 
                from the function corr2_utils.MakeCorr2FileKwargs(), rather than ingesting the data 
                directly, though they can also ingest the data directly as well.
        
        In both cases, the kwargs should be able to handle a "bin_list=" kwarg which will bin the 
        data accordingly--see the classes defined in binning.py for more.
        
    """
    short_name = ''
    long_name = ''
    def __init__(self):
        pass
    def __call__(self):
        raise NotImplementedError()
    def plot(self, results):
        """
        If the results returned from the __call__() function of this class have a .savefig() 
        method, return that object.  Otherwise, return an object with a .savefig() method that 
        doesn't do anything.  plot() should be overridden by child classes to actually generate 
        plots if desired.
        """
        if hasattr(results,'savefig'):
            return results
        else:
            return PlotNone()
        
class PlotDetails(object):
    """
    A container class to hold details about field names, titles for legends, and y-axis labels for
    plots of correlation functions.
    """
    def __init__(self, t_field=None, t_title=None, b_field=None, b_title=None,
                 t_im_field=None, t_im_title=None, b_im_field=None, b_im_title=None,
                 datarandom_t_field=None, datarandom_t_title=None,
                 datarandom_b_field=None, datarandom_b_title=None,
                 sigma_field=None, y_title=None):
        self.t_field = t_field  # Field of t-mode/+-mode shear correlation functions
        self.t_title = t_title  # Legend title for previous line
        self.b_field = b_field  # Field of b-mode/x-mode shear correlation functions 
        self.b_title = b_title  # Legend title for previous line
        self.t_im_field = t_im_field  # Imaginary part of t-mode/+-mode
        self.t_im_title = t_im_title  # Legend title for previous line
        self.b_im_field = b_im_field  # Imaginary part of b-mode/x-mode
        self.b_im_title = b_im_title  # Legend title for previous line
        self.datarandom_t_field = datarandom_t_field  # If data or randoms are available separately,
                                                      # this +'d' or +'r' is the t-mode field name
        self.datarandom_t_title = datarandom_t_title  # Legend title for previous line
        self.datarandom_b_field = datarandom_b_field  # As above, for b-mode
        self.datarandom_b_title = datarandom_b_title  # Legend title for previous line
        self.sigma_field = sigma_field  # 1-sigma error bar field
        self.y_title = y_title  # y-axis label

corr2_func_dict = {'g2': treecorr.G2Correlation,
                   'm2': treecorr.G2Correlation,
                   'ng': treecorr.NGCorrelation,
                   'nm': treecorr.NGCorrelation,
                   'norm': treecorr.NGCorrelation,
                   'n2': treecorr.N2Correlation,
                   'k2': treecorr.K2Correlation,
                   'nk': treecorr.NKCorrelation,
                   'kg': treecorr.KGCorrelation}

class CorrelationFunctionSysTest(SysTest):
    """
    A base class for the Stile systematics tests that use correlation functions. This implements the
    class method getCF(), which runs corr2 (via a call to the subprocess module) on a given set of 
    data.  Exact arguments to this method should be created by child classes of
    CorrelationFunctionSysTest; see the docstring for CorrelationFunctionSysTest.getCF() for 
    information on how to write further tests using it.
    """
    short_name = 'corrfunc'
    # Set the details (such as field names and titles) for all the possible plots generated by corr2
    plot_details = [PlotDetails(t_field='omega', t_title='$\omega$', 
                                sigma_field='sig_omega', y_title="$\omega$"),  # n2
        PlotDetails(t_field='<gamT>', t_title=r'$\langle \gamma_T \rangle$', 
                    b_field='<gamX>', b_title=r'$\langle \gamma_X \rangle$',
                    datarandom_t_field='gamT_', datarandom_t_title='$\gamma_{T',
                    datarandom_b_field='gamX_', datarandom_b_title='$\gamma_{X',
                    sigma_field='sigma', y_title="$\gamma$"),  #ng
        PlotDetails(t_field='xi+', t_title=r'$\xi_+$', b_field='xi-', b_title=r'$\xi_-$',
                    t_im_field='xi+_im', t_im_title=r'$\xi_{+,im}$', 
                    b_im_field='xi-_im', b_im_title=r'$\xi_{-,im}$', 
                    sigma_field='sig_xi', y_title=r"$\xi$"),  #g2
        PlotDetails(t_field='<kappa>', t_title=r'$\langle \kappa \rangle$', 
                    datarandom_t_field='kappa_', datarandom_t_title='$kappa_{',
                    sigma_field='sigma', y_title="$\kappa$"),  # nk 
        PlotDetails(t_field='xi', t_title=r'$\xi$', sigma_field='sig_xi', y_title=r"$\xi$"),  # k2 
        PlotDetails(t_field='<kgamT>', t_title=r'$\langle \kappa \gamma_T\rangle$',
                    b_field='<kgamX>', b_title=r'$\langle \kappa \gamma_X\rangle$',
                    datarandom_t_field='kgamT_', datarandom_t_title=r'$\kappa \gamma_{T', 
                    datarandom_b_field='kgamX_', datarandom_b_title=r'$\kappa \gamma_{X',
                    sigma_field='sigma', y_title="$\kappa\gamma$"),  # kg 
        PlotDetails(t_field='<Map^2>', t_title=r'$\langle M_{ap}^2 \rangle$', 
                    b_field='<Mx^2>', b_title=r'$\langle M_x^2\rangle$',
                    t_im_field='<MMx>(a)', t_im_title=r'$\langle MM_x \rangle(a)$', 
                    b_im_field='<Mmx>(b)', b_im_title=r'$\langle MM_x \rangle(b)$',
                    sigma_field='sig_map', y_title="$M_{ap}^2$"),  # m2
        PlotDetails(t_field='<NMap>', t_title=r'$\langle NM_{ap} \rangle$', 
                    b_field='<NMx>', b_title=r'$\langle NM_{x} \rangle$',
                    sigma_field='sig_nmap', y_title="$NM_{ap}$")  # nm or norm
        ]

    def makeCatalog(self, data, use_as_k=None, use_chip_coords=False):
        if data is None or isinstance(data,treecorr.Catalog):
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
        return treecorr.Catalog(**catalog_kwargs)
        
    def getCF(self, stile_args, correlation_function_type, data, data2=None,
                    random=None, random2=None, use_as_k = None, use_chip_coords=False, **kwargs):
        """
        Sets up and calls treecorr on the given set of data.  
        
        The user needs to specify the type of correlation function requested.  The available types
        are:
            'n2': a 2-point correlation function
            'ng': a point-shear correlation function (eg galaxy-galaxy lensing)
            'g2': a shear-shear correlation function (eg cosmic shear)
            'nk': a point-scalar [such as convergence, hence k meaning "kappa"] correlation function
            'k2': a scalar-scalar correlation function
            'kg': a scalar-shear correlation function
            'm2': an aperture mass measurement
            'nm': an <N aperture mass> measurement
            'norm': 'nm' properly normalized by the average values of n and aperture mass to return
                    something like a correlation coefficient. 
        More details can be found in the Read.me for corr2.
        
        This function accepts all (self-consistent) sets of data, data2, random, and random2.  
        Including "data2" and possibly "random2" will return a cross-correlation; otherwise the 
        program returns an autocorrelation.  "Random" keys are necessary for the 'n2' form of the 
        correlation function, and can be used (but are not necessary) for 'ng', 'nk', and 'kg'.
        
        @param stile_args    The dict containing the parameters that control Stile's behavior
        @param correlation_function_type The type of correlation function ('n2','ng','g2','nk','k2',
                             'kg','m2','nm','norm') to request from corr2--see above.
        @param data, data2, random, random2: NumPy arrays of data with fields using the field name
                             strings given in the stile.fieldNames dict.
        @param kwargs        Any other corr2 parameters to be written to the corr2 param file (will
                             silently supercede anything in stile_args).
        @returns             a numpy array of the corr2 outputs.
        """
        #TODO: know what kinds of data this needs and make sure it has it
        import tempfile
        import os
        handle, output_file = tempfile.mkstemp()

        # First, pull out the corr2-relevant parameters from the stile_args dict, and add anything
        # passed as a kwarg to that dict.
        corr2_kwargs = stile.corr2_utils.PickTreeCorrKeys(stile_args)
        corr2_kwargs.update(stile.corr2_utils.PickTreeCorrKeys(kwargs))
        treecorr.config.check_config(corr2_kwargs,corr2_valid_params)

        data = self.makeCatalog(data, use_as_k = use_as_k, use_chip_coords = use_chip_coords)
        data2 = self.makeCatalog(data2, use_as_k = use_as_k, use_chip_coords = use_chip_coords)
        random = self.makeCatalog(random, use_as_k = use_as_k, use_chip_coords = use_chip_coords)
        random2 = self.makeCatalog(random2, use_as_k = use_as_k, use_chip_coords = use_chip_coords)

        corr2_kwargs[correlation_function_type+'_file_name'] = output_file
       
        func = corr2_func_dict[correlation_function_type](corr2_kwargs)
        func.process(data,data2)
        func.write(output_file)
        results = stile.ReadCorr2ResultsFile(output_file)
        os.close(handle)
        if os.path.isfile(output_file):
            os.remove(output_file)
        return results
        
    def plot(self, data, colors=['r', 'b'], log_yscale=False,
                   plot_bmode=True, plot_data_only=True, plot_random_only=True):
        """
        Plot the data returned from a CorrelationFunctionSysTest object.  This chooses some 
        sensible defaults, but much of its behavior can be changed.
        
        @param data       The data returned from a CorrelationFunctionSysTest, as-is.
        @param colors     A tuple of 2 colors, used for the first and second lines on any given plot
        @param log_yscale Whether to use a logarithmic y-scale (default: False)
        @param plot_bmode Whether to plot the b-mode signal, if there is one (default: True)
        @param plot_data_only   Whether to plot the data-only correlation functions, if present
                                (default: True)
        @param plot_random_only Whether to plot the random-only correlation functions, if present
                                (default: True)
        @returns          A matplotlib Figure which may be written to a file with .savefig(), if
                          matplotlib can be imported; else None.
        """

        if not has_matplotlib:
            return None
        fields = data.dtype.names
        # Pick which radius measurement to use
        for t_r in ['<R>', 'R_nominal', 'R']:
            if t_r in fields:
                r = t_r
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
        plot_data_only &= pd.datarandom_t_field+'d' in fields
        plot_random_only &= pd.datarandom_t_field+'r' in fields
        if plot_bmode and pd.b_field and pd.t_im_field:
            nrows = 2
        elif pd.datarandom_t_field:
            nrows = 1 + plot_data_only + plot_random_only
        else:
            nrows = 1
        if 'sigma' in fields and pd.sigma_field=='sig':
            pd.sigma_field='sigma'
        # Plot the first thing
        curr_plot = 0
        ax = fig.add_subplot(nrows, 1, 1)
        ax.errorbar(data[r], data[pd.t_field], yerr=data[pd.sigma_field], color=colors[0], 
                    label=pd.t_title)
        if pd.b_title and plot_bmode:
            ax.errorbar(data[r], data[pd.b_field], yerr=data[pd.sigma_field], color=colors[1], 
                        label=pd.b_title)
        elif pf.t_im_title:  # Plot y and y_im if not plotting yb (else it goes on a separate plot)
            ax.errorbar(data[r], data[pd.t_im_field], yerr=data[pd.sigma_field], color=colors[1], 
                        label=pd.t_im_title)
        ax.set_xscale('log')
        ax.set_yscale(yscale)
        ax.set_xlim(xlim)
        ax.set_ylabel(pd.y_title)
        ax.legend()
        if pd.b_field and plot_bmode and pd.t_im_field:  
            # Both yb and y_im: plot (y,yb) on one plot and (y_im,yb_im) on the other.
            ax = fig.add_subplot(nrows, 1, 2)
            ax.errorbar(data[r], data[pd.t_im_field], yerr=data[pd.sigma_field], color=colors[0], 
                        label=pd.t_im_title)
            ax.errorbar(data[r], data[pd.b_im_field], yerr=data[pd.sigma_field], color=colors[1], 
                        label=pd.b_im_title)
            ax.set_xscale('log')
            ax.set_yscale(yscale)
            ax.set_xlim(xlim)
            ax.set_ylabel(pd.y_title)
            ax.legend()
        if plot_data_only and pd.datarandom_t_field:  # Plot the data-only measurements if requested
            curr_plot += 1
            ax = fig.add_subplot(nrows, 1, 2)
            ax.errorbar(data[r], data[pd.datarandom_t_field+'d'], yerr=data[pd.sigma_field], 
                        color=colors[0], label=pd.datarandom_t_title+'d}$')
            if plot_bmode and pd.datarandom_b_field:
                ax.errorbar(data[r], data[pd.datarandom_b_field+'d'], yerr=data[pd.sigma_field], 
                        color=colors[1], label=pd.datarandom_b_title+'d}$')
            ax.set_xscale('log')
            ax.set_yscale(yscale)
            ax.set_xlim(xlim)
            ax.set_ylabel(pd.y_title)
            ax.legend()
        if plot_random_only and pd.datarandom_t_field:  # Plot the randoms-only measurements if requested
            ax = fig.add_subplot(nrows, 1, nrows)
            ax.errorbar(data[r], data[pd.datarandom_t_field+'r'], yerr=data[pd.sigma_field], 
                        color=colors[0], label=pd.datarandom_t_title+'r}$')
            if plot_bmode and pd.datarandom_b_field:
                ax.errorbar(data[r], data[pd.datarandom_b_field+'r'], yerr=data[pd.sigma_field], 
                        color=colors[1], label=pd.datarandom_b_title+'r}$')
            ax.set_xscale('log')
            ax.set_yscale(yscale)
            ax.set_xlim(xlim)
            ax.set_ylabel(pd.y_title)
            ax.legend()
        ax.set_xlabel(r)
        return fig


class GalaxyShearSysTest(CorrelationFunctionSysTest):
    """
    Compute the tangential and cross shear around a set of real galaxies.
    """
    short_name = 'shear_around_galaxies'
    long_name = 'Shear of galaxies around real objects'
    objects_list = ['galaxy lens', 'galaxy']
    required_quantities = [('ra', 'dec'), ('ra', 'dec', 'g1', 'g2', 'w')]

    def __call__(self, stile_args, data, data2=None, random=None, random2=None, **kwargs):
        return self.getCF(stile_args, 'ng', data, data2, random, random2, **kwargs)

class BrightStarShearSysTest(CorrelationFunctionSysTest):
    """
    Compute the tangential and cross shear around a set of bright stars.
    """
    short_name = 'shear_around_bright_stars'
    long_name = 'Shear of galaxies around bright stars'
    objects_list = ['star bright', 'galaxy']
    required_quantities = [('ra', 'dec'), ('ra', 'dec', 'g1', 'g2', 'w')]

    def __call__(self, stile_args, data, data2=None, random=None, random2=None, **kwargs):
        return self.getCF(stile_args, 'ng', data, data2, random, random2, **kwargs)

class StarXGalaxyDensitySysTest(CorrelationFunctionSysTest):
    """
    Compute the number density of galaxies around stars.
    """
    short_name = 'star_x_galaxy_density'
    long_name = 'Density of galaxies around stars'
    objects_list = ['star', 'galaxy', 'star random', 'galaxy random']
    required_quantities = [('ra', 'dec'), ('ra', 'dec'), ('ra', 'dec'), ('ra', 'dec')]

    def __call__(self, stile_args, data, data2=None, random=None, random2=None, **kwargs):
        return self.getCF(stile_args, 'n2', data, data2, random, random2, **kwargs)

class StarXGalaxyShearSysTest(CorrelationFunctionSysTest):
    """
    Compute the cross-correlation of galaxy and star shapes.
    """
    short_name = 'star_x_galaxy_shear'
    long_name = 'Cross-correlation of galaxy and star shapes'
    objects_list = ['star', 'galaxy']
    required_quantities = [('ra', 'dec', 'g1', 'g2', 'w'), ('ra', 'dec', 'g1', 'g2', 'w')]

    def __call__(self, stile_args, data, data2=None, random=None, random2=None, **kwargs):
        return self.getCF(stile_args, 'g2', data, data2, random, random2, **kwargs)

class StarXStarShearSysTest(CorrelationFunctionSysTest):
    """
    Compute the auto-correlation of star shapes.
    """
    short_name = 'star_x_star_shear'
    long_name = 'Auto-correlation of star shapes'
    objects_list = ['star']
    required_quantities = [('ra', 'dec', 'g1', 'g2', 'w')]

    def __call__(self, stile_args, data, data2=None, random=None, random2=None, **kwargs):
        return self.getCF(stile_args, 'g2', data, data2, random, random2, **kwargs)

class GalaxyDensityCorrelationSysTest(CorrelationFunctionSysTest):
    """
    Compute the galaxy position autocorrelations.
    """
    short_name = 'galaxy_density'
    long_name = 'Galaxy position autocorrelation'
    objects_list = ['galaxy', 'galaxy random']
    required_quantities = [('ra', 'dec'), ('ra', 'dec')]

    def __call__(self, stile_args, data, data2=None, random=None, random2=None, **kwargs):
        return self.getCF(stile_args, 'n2', data, data2, random, random2, **kwargs)

class StarDensityCorrelationSysTest(CorrelationFunctionSysTest):
    """
    Compute the star position autocorrelations.
    """
    short_name = 'star_density'
    long_name = 'Star position autocorrelation'
    objects_list = ['star', 'star random']
    required_quantities = [('ra', 'dec'), ('ra', 'dec')]

    def __call__(self, stile_args, data, data2=None, random=None, random2=None, **kwargs):
        return self.getCF(stile_args, 'n2', data, data2, random, random2, **kwargs)


class StatSysTest(SysTest):
    """
    A class for the Stile systematics tests that use basic statistical quantities. It uses NumPy
    routines for all the innards, and saves the results in a stile.stile_utils.Stats object (see
    stile_utils.py) that can carry around the information, print the results in a useful format,
    write to file, or (eventually) become an argument to plotting routines that might output some of
    the results on plots.

    One of the calculations it does is find the percentiles of the given quantity.  The percentile
    levels to use can be set when the StatSysTest is initialized, or when it is called.  These
    percentiles must be provided as an iterable (list, tuple, or NumPy array).

    The objects on which this systematics test is used should be either (a) a simple iterable like a
    list, tuple, or NumPy array, or (b) a structured NumPy array with fields.  In case (a), the
    dimensionality of the NumPy array is ignored, and statistics are calculated over all
    dimensions.  In case (b), the user must give a field name using the `field` keyword argument,
    either at initialization or when calling the test.

    For both the `percentile` and `field` arguments, the behavior is different if the keyword
    argument is used at the time of initialization or calling.  When used at the time of
    initialization, that value will be used for all future calls unless called with another value
    for those arguments.  However, the value of `percentile` and `field` for calls after that will
    revert back to the original value from the time of initialization.

    By default, the systematics tester will simply return a Stats object for the user.  However,
    calling it with `verbose=True` will result in the statistics being printed directly using the
    Stats.prettyPrint() function.

    Ordinarily, a StatSysTest object will throw an exception if asked to run on an array that has
    any Nans or infinite values.  The `ignore_bad` keyword (at the time when the StatSytTest is
    called, not initialized) changes this behavior so these bad values are quietly ignored.

    Options to consider adding in future: weighted sums and other weighted statistics; outlier
    rejection.
    """
    short_name = 'stats'
    long_name = 'Calculate basic statistics of a given quantity'

    def __init__(self, percentiles=[2.2, 16., 50., 84., 97.8], field=None):
        """Function to initialize a StatSysTest object.

        @param percentiles     The percentile levels at which to find the value of the input array
                               when called.  [default: [2.2, 16., 50., 84., 97.8].]
        @param field           The name of the field to use in a NumPy structured array / catalog.
                               [default: None, meaning we're using a simple array without field
                               names.]

        @returns the requested StatSysTest object.
        """
        self.percentiles = percentiles
        self.field = field

    def __call__(self, array, percentiles=None, field=None, verbose=False, ignore_bad=False):
        """Calling a StatSysTest with a given array argument as `array` will cause it to carry out
        all the statistics tests and populate a stile.Stats object with the results, which it returns
        to the user.

        @param array           The tuple, list, NumPy array, or structured NumPy array/catalog on
                               which to carry out the calculations.
        @param percentiles     The percentile levels to use for this particular calculation.
                               [default: None, meaning use whatever levels were defined when
                               initializing this StatSysTest object]
        @param field           The name of the field to use in a NumPy structured array / catalog.
                               [default: None, meaning use whatever field was defined when
                               initializing this StatSysTest object]
        @param verbose         If True, print the calculated statistics of the input `array` to
                               screen.  If False, silently return the Stats object. [default:
                               False.]
        @param ignore_bad      If True, search for values that are NaN or Inf, and remove them
                               before doing calculations.  [default: False.]

        @returns a stile.stile_utils.Stats object
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
            if use_field not in use_array.dtype.fields.keys():
                raise RuntimeError('Field %s is not in this catalog, which contains %s!'%
                                   (use_field, use_array.dtype.fields.keys()))
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
            simple_stats=['min', 'max', 'median', 'mad', 'mean', 'stddev', 'variance', 'N',
                          'skew', 'kurtosis']
        except ImportError:
            simple_stats=['min', 'max', 'median', 'mad', 'mean', 'stddev', 'variance', 'N']
            
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
            print result.__str__()

        # Return.
        return result
