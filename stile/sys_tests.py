"""@file sys_tests.py
Contains the class definitions of the Stile systematics tests.
"""
import numpy
import stile
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
               

class CorrelationFunctionSysTest(SysTest):
    """
    A base class for the Stile systematics tests that use correlation functions. This implements the
    class method getCF(), which runs corr2 (via a call to the subprocess module) on a given set of 
    data.  Exact arguments to this method should be created by child classes of
    CorrelationFunctionSysTest; see the docstring for CorrelationFunctionSysTest.getCF() for 
    information on how to write further tests using it.
    """
    short_name = 'corrfunc'

    def getCF(self, stile_args, correlation_function_type, data=None, data2=None,
                                     random=None, random2=None, save_config=False, **kwargs):
        """
        Sets up and calls corr2 on the given set of data.  The data files and random files can
        be contained already in stile_args['corr2_kwargs'] or **kwargs, in which case passing None
        to the `data` and `random` kwargs is fine; otherwise they should be properly populated.
        
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
        
        Note: by default, the corr2 configuration files are written to the temp directory called by 
        tempfile.mkstemp().  If you need to examine the corr2 config files, you can pass 
        `save_config=True` and they will be written (as temp files probably beginning with "tmp") 
        to your working directory, which shouldn't be automatically cleaned up.  
        
        @param stile_args    The dict containing the parameters that control Stile's behavior
        @param correlation_function_type The type of correlation function ('n2','ng','g2','nk','k2',
                             'kg','m2','nm','norm') to request from corr2--see above.
        @param data, data2, random, random2: data sets in the format requested by 
                             corr2_utils.MakeCorr2FileKwargs().
        @param kwargs        Any other corr2 parameters to be written to the corr2 param file (will
                             silently supercede anything in stile_args).
        @returns             a numpy array of the corr2 outputs.
        """
        #TODO: know what kinds of data this needs and make sure it has it
        import tempfile
        import subprocess
        import os
        import copy
        handles = []

        # First, pull out the corr2-relevant parameters from the stile_args dict, and add anything
        # passed as a kwarg to that dict.
        corr2_kwargs = stile.corr2_utils.AddCorr2Dict(stile_args)
        corr2_kwargs.update(kwargs)
        # Now, pass the data and random arguments to MakeCorr2FileKwargs.  This will write to disk
        # any data that's currently contained in memory for Stile, as well as making sure that all
        # the files are in the same format--corr2 expects ra (etc) to be in the same column in
        # every file. Then it returns a bunch of (key,value) pairs that we can use to write a corr2
        # config file: the file names plus the format parameters (such as `ra_col`, `dec_col`,
        # etc).  Empty data sets return nothing, and if all data sets are empty, the return value
        # is an empty dict.  It's possible the user already ran MakeCorr2FileKwargs and the results
        # have been passed as kwargs to this function.  We don't explicitly check for that, but as
        # long as the user doesn't pass anything to `data`, `data2`, `random`, or `random2`, no
        # conflicts will arise.
        corr2_file_kwargs = stile.MakeCorr2FileKwargs(data, data2, random, random2)
        corr2_kwargs.update(corr2_file_kwargs)

        # make sure the set of non-None data sets makes sense
        if not ('file_list' in corr2_kwargs or 'file_name' in corr2_kwargs):
            raise ValueError("stile_args['corr2_kwargs'] or **kwargs must contain a file kwarg")
        if ('rand_list2' in corr2_kwargs or 'rand_name2' in corr2_kwargs) and not (
            'file_name2' in corr2_kwargs or 'file_list2' in corr2_kwargs):
            raise ValueError('Given random file for file 2, but there is no file 2')

        if save_config:
            handle, config_file = tempfile.mkstemp(dir='.')
        else:
            handle, config_file = tempfile.mkstemp()
        handles.append(handle)
        handle, output_file = tempfile.mkstemp()
        handles.append(handle)

        corr2_kwargs[correlation_function_type+'_file_name'] = output_file
       
        stile.WriteCorr2ConfigurationFile(config_file, corr2_kwargs)

        #TODO: don't hard-code the name of corr2!
        subprocess.check_call(['corr2', config_file])

        return_value = stile.ReadCorr2ResultsFile(output_file)
        for handle in handles:
            os.close(handle)
        return return_value
    
    # Set the details (such as field names and titles) for all the possible plots generated by corr2
    plot_details = [PlotDetails(t_field='omega', t_title='$\omega$', 
                                sigma_field='sig_omega', y_title="$\omega$"),  # n2
        PlotDetails(t_field='<gamT>', t_title=r'$\langle \gamma_T \rangle$', 
                    b_field='<gamX>', b_title=r'$\langle \gamma_X \rangle$',
                    datarandom_t_field='gamT_', datarandom_t_title='$\gamma_{T',
                    datarandom_b_field='gamX_', datarandom_b_title='$\gamma_{X',
                    sigma_field='sig', y_title="$\gamma$"),  #ng
        PlotDetails(t_field='xi+', t_title=r'$\xi_+$', b_field='xi-', b_title=r'$\xi_-$',
                    t_im_field='xi+_im', t_im_title=r'$\xi_{+,im}$', 
                    b_im_field='xi-_im', b_im_title=r'$\xi_{-,im}$', 
                    sigma_field='sig_xi', y_title=r"$\xi$"),  #g2
        PlotDetails(t_field='<kappa>', t_title=r'$\langle \kappa \rangle$', 
                    datarandom_t_field='kappa_', datarandom_t_title='$kappa_{',
                    sigma_field='sig', y_title="$\kappa$"),  # nk 
        PlotDetails(t_field='xi', t_title=r'$\xi$', sigma_field='sig_xi', y_title=r"$\xi$"),  # k2 
        PlotDetails(t_field='<kgamT>', t_title=r'$\langle \kappa \gamma_T\rangle$',
                    b_field='<kgamX>', b_title=r'$\langle \kappa \gamma_X\rangle$',
                    datarandom_t_field='kgamT_', datarandom_t_title=r'$\kappa \gamma_{T', 
                    datarandom_b_field='kgamX_', datarandom_b_title=r'$\kappa \gamma_{X',
                    sigma_field='sig', y_title="$\kappa\gamma$"),  # kg 
        PlotDetails(t_field='<Map^2>', t_title=r'$\langle M_{ap}^2 \rangle$', 
                    b_field='<Mx^2>', b_title=r'$\langle M_x^2\rangle$',
                    t_im_field='<MMx>(a)', t_im_title=r'$\langle MM_x \rangle(a)$', 
                    b_im_field='<Mmx>(b)', b_im_title=r'$\langle MM_x \rangle(b)$',
                    sigma_field='sig_map', y_title="$M_{ap}^2$"),  # m2
        PlotDetails(t_field='<NMap>', t_title=r'$\langle NM_{ap} \rangle$', 
                    b_field='<NMx>', b_title=r'$\langle NM_{x} \rangle$',
                    sigma_field='sig_nmap', y_title="$NM_{ap}$")  # nm or norm
        ]
        
    def plot(self, data, colors=['r', 'b'], log_yscale=False,
                   plot_bmode=True, plot_data_only=True, plot_random_only=True):
        """
        Plot the data returned from a CorrelationFunctionSysTest object.  This chooses some 
        sensible defaults, but much of its behavior can be changed.
        
        @param data       The data returned from a CorrelationFunctionSysTest, as-is.
        @param colors     A tuple of 2 colors, used for the first and second lines on any given plot
        @param log_yscale Whether to use a logarithmic y-scale [default: False]
        @param plot_bmode Whether to plot the b-mode signal, if there is one [default: True]
        @param plot_data_only   Whether to plot the data-only correlation functions, if present
                                [default: True]
        @param plot_random_only Whether to plot the random-only correlation functions, if present
                                [default: True]
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
        if pd.datarandom_t_field:
            plot_data_only &= pd.datarandom_t_field+'d' in fields
            plot_random_only &= pd.datarandom_t_field+'r' in fields
        if plot_bmode and pd.b_field and pd.t_im_field:
            nrows = 2
        elif pd.datarandom_t_field:
            nrows = 1 + plot_data_only + plot_random_only
        else:
            nrows = 1

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

    def __call__(self, stile_args, data=None, data2=None, random=None, random2=None, **kwargs):
        return self.getCF(stile_args, 'ng', data, data2, random, random2, **kwargs)

class BrightStarShearSysTest(CorrelationFunctionSysTest):
    """
    Compute the tangential and cross shear around a set of bright stars.
    """
    short_name = 'shear_around_bright_stars'
    long_name = 'Shear of galaxies around bright stars'
    objects_list = ['star bright', 'galaxy']
    required_quantities = [('ra', 'dec'), ('ra', 'dec', 'g1', 'g2', 'w')]

    def __call__(self, stile_args, data=None, data2=None, random=None, random2=None, **kwargs):
        return self.getCF(stile_args, 'ng', data, data2, random, random2, **kwargs)

class StarXGalaxyDensitySysTest(CorrelationFunctionSysTest):
    """
    Compute the number density of galaxies around stars.
    """
    short_name = 'star_x_galaxy_density'
    long_name = 'Density of galaxies around stars'
    objects_list = ['star', 'galaxy', 'star random', 'galaxy random']
    required_quantities = [('ra', 'dec'), ('ra', 'dec'), ('ra', 'dec'), ('ra', 'dec')]

    def __call__(self, stile_args, data=None, data2=None, random=None, random2=None, **kwargs):
        return self.getCF(stile_args, 'n2', data, data2, random, random2, **kwargs)

class StarXGalaxyShearSysTest(CorrelationFunctionSysTest):
    """
    Compute the cross-correlation of galaxy and star shapes.
    """
    short_name = 'star_x_galaxy_shear'
    long_name = 'Cross-correlation of galaxy and star shapes'
    objects_list = ['star', 'galaxy']
    required_quantities = [('ra', 'dec', 'g1', 'g2', 'w'), ('ra', 'dec', 'g1', 'g2', 'w')]

    def __call__(self, stile_args, data=None, data2=None, random=None, random2=None, **kwargs):
        return self.getCF(stile_args, 'g2', data, data2, random, random2, **kwargs)

class StarXStarShearSysTest(CorrelationFunctionSysTest):
    """
    Compute the auto-correlation of star shapes.
    """
    short_name = 'star_x_star_shear'
    long_name = 'Auto-correlation of star shapes'
    objects_list = ['star']
    required_quantities = [('ra', 'dec', 'g1', 'g2', 'w')]

    def __call__(self, stile_args, data=None, data2=None, random=None, random2=None, **kwargs):
        return self.getCF(stile_args, 'g2', data, data2, random, random2, **kwargs)

class GalaxyDensityCorrelationSysTest(CorrelationFunctionSysTest):
    """
    Compute the galaxy position autocorrelations.
    """
    short_name = 'galaxy_density'
    long_name = 'Galaxy position autocorrelation'
    objects_list = ['galaxy', 'galaxy random']
    required_quantities = [('ra', 'dec'), ('ra', 'dec')]

    def __call__(self, stile_args, data=None, data2=None, random=None, random2=None, **kwargs):
        return self.getCF(stile_args, 'n2', data, data2, random, random2, **kwargs)

class StarDensityCorrelationSysTest(CorrelationFunctionSysTest):
    """
    Compute the star position autocorrelations.
    """
    short_name = 'star_density'
    long_name = 'Star position autocorrelation'
    objects_list = ['star', 'star random']
    required_quantities = [('ra', 'dec'), ('ra', 'dec')]

    def __call__(self, stile_args, data=None, data2=None, random=None, random2=None, **kwargs):
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

class WhiskerPlotSysTest(SysTest):
    short_name = 'whiskerplot'
    """
    A base class for Stile systematics tests that generate whisker plots. This implements the class
    method whiskerPlot. Every child class of WhiskerPlotSysTest should use
    WhiskerPlotSysTest.whiskerPlot through __call__. See the docstring for
    WhiskerPlotSysTest.whiskerPlot for information on how to write further tests using it.
    """
    def whiskerPlot(self, x, y, g1, g2, size = None, linewidth = 0.01, scale = None,
                    keylength = 0.05, figsize = None, xlabel = None, ylabel = None,
                    size_label = None, xlim = None, ylim = None, equal_axis = False):
        """
        Draw a whisker plot and return a `matplotlib.figure.Figure` object.
        This method has a bunch of options for controlling the appearance of a plot, which are
        explained below. To implement a child class of WhiskerPlotSysTest, call whiskerPlot within
        __call__ of the child class and return the `matplotlib.figure.Figure` that whiskerPlot 
        returns.
        @param x               The tuple, list, or NumPy array for the x-position of objects.
        @param y               The tuple, list, or NumPy array for the y-position of objects.
        @param g1              The tuple, list, or Numpy array for the 1st ellipticity component
                               of objects.
        @param g2              The tuple, list, or Numpy array for the 2nd ellipticity component
                               of objects.
        @param size            The tuple, list, or Numpy array for the size of objects. The size
                               information is shown as color gradation.
                               [default: None, meaning do not show the size information]
        @param linewidth       Width of whiskers in units of inches.
                               [default: 0.01]
        @param scale           Length of whisker per inch.
                               [default: None, meaning follow the default autoscaling algorithm from
                               matplotlib]
        @param keylength       Length of a key.
                               [default: 0.05]
        @param figsize         Size of a figure (x, y) in units of inches.
                               [default: None, meaning use the default value of matplotlib]
        @param xlabel          The x-axis label.
                               [default: None, meaning do not show a label for the x-axis]
        @param ylabel          The y-axis label.
                               [default: None, meaning do not show a label for the y-axis]
        @param size_label      The label for `size`, which is shown at the right of the color bar.
                               [default: None, meaning do not show a size label]
        @param xlim            Limits of x-axis (min, max). 
                               [default: None, meaning do not set any limits for x]
        @param ylim            Limits of y-axis (min, max). 
                               [default: None, meaning do not set any limits for y]
        @equal_axis            If True, force equal scaling for the x and y axes (distance between
                               ticks of the same numerical values are equal on the x and y axes).
                               [default: False]
        @returns a matplotlib.figure.Figure object.
        """
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1,1,1)

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
        theta = numpy.arctan2(g2,g1)/2
        gx = g * numpy.cos(theta)
        gy = g * numpy.sin(theta)
        if size is None:
            q = ax.quiver(x, y, gx, gy, units = 'inches',
                          headwidth = 0., headlength = 0., headaxislength = 0.,
                          pivot = 'middle', width = linewidth, 
                          scale = scale)
        else:
            q = ax.quiver(x, y, gx, gy, size, units = 'inches',
                          headwidth = 0., headlength = 0., headaxislength = 0.,
                          pivot = 'middle', width = linewidth,
                          scale = scale)
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

class WhiskerPlotStarSysTest(WhiskerPlotSysTest):
    short_name = 'whiskerplot_star'
    long_name = 'Make a Whisker plot of stars'
    objects_list = ['star PSF']
    required_quantities = [('x','y','g1','g2','sigma')]

    def __call__(self, array, linewidth = 0.01, scale = None, figsize = None,
                 xlim = None, ylim = None):
        return self.whiskerPlot(array['x'], array['y'], array['g1'], array['g2'], array['sigma'],
                                linewidth = linewidth, scale = scale, figsize = figsize,
                                xlabel = r'$x$ [arcsec]', ylabel = r'$y$ [arcsec]',
                                size_label = r'$\sigma$ [arcsec]',
                                xlim = xlim, ylim = ylim, equal_axis = True)

class WhiskerPlotPSFSysTest(WhiskerPlotSysTest):
    short_name = 'whiskerplot_psf'
    long_name = 'Make a Whisker plot of PSFs'
    objects_list = ['star PSF']
    required_quantities = [('x','y','psf_g1','psf_g2','psf_sigma')]

    def __call__(self, array, linewidth = 0.01, scale = None, figsize = None,
                 xlim = None, ylim = None):
        return self.whiskerPlot(array['x'], array['y'], array['psf_g1'], array['psf_g2'],
                                array['psf_sigma'], linewidth = linewidth, scale = scale,
                                figsize = figsize, xlabel = r'$x$ [arcsec]', ylabel = r'$y$ [arcsec]',
                                size_label = r'$\sigma$ [arcsec]', 
                                xlim = xlim, ylim = ylim, equal_axis = True)
    
class WhiskerPlotResidualSysTest(WhiskerPlotSysTest):
    short_name = 'whiskerplot_residual'
    long_name = 'Make a Whisker plot of residuals'
    objects_list = ['star PSF']
    required_quantities = [('x','y', 'g1','g2','sigma', 'psf_g1','psf_g2','psf_sigma')]

    def __call__(self, array, linewidth = 0.01, scale = None, figsize = None,
                 xlim = None, ylim = None):
        return self.whiskerPlot(array['x'], array['y'], array['g1'] - array['psf_g1'],
                                array['g2'] - array['psf_g2'], array['sigma'] - array['psf_sigma'],
                                linewidth = linewidth, scale = scale,
                                figsize = figsize, xlabel = r'$x$ [arcsec]', ylabel = r'$y$ [arcsec]',
                                size_label = r'$\sigma$ [arcsec]', 
                                xlim = xlim, ylim = ylim, equal_axis = True)

class ScatterPlotSysTest(SysTest):
    short_name = 'scatterplot'
    """
    A base class for Stile systematics tests that generate scatter plots. This implements the class 
    method scatterPlot. Every child class of ScatterPlotSysTest should use
    ScatterPlotSysTest.scatterPlot through __call__. See the docstring for
    ScatterPlotSysTest.scatterPlot for information on how to write further tests using it.
    """
    def scatterPlot(self, x, y, yerr=None, z=None, xlabel=None, ylabel=None, zlabel=None, color = ""
                    , lim=None, equal_axis=False, linear_regression=False, reference_line = None):
        """
        Draw a scatter plot and return a `matplotlib.figure.Figure` object.
        This method has a bunch of options for controlling appearance of a plot, which is
        explained below. To implement a child class of ScatterPlotSysTest, call scatterPlot within
        __call__ of the child class and return `matplotlib.figure.Figure` that scatterPlot returns.
        @param x               The tuple, list, or NumPy array for x-axis.
        @param y               The tuple, list, or NumPy array for y-axis.
        @param yerr            The tuple, list, or Numpy array for error of the y values.
                               [default: None, meaning do not plot an error]
        @param z               The tuple, list, or Numpy array for an additional quantitiy
                               which appears as colors of scattered points.
                               [default: None, meaning there is no additional quantity]
        @param xlabel          The label of x-axis.
                               [default: None, meaning do not show a label of x-axis]
        @param ylabel          The label of y-axis.
                               [default: None, meaning do not show a label of y-axis]
        @param zlabel          The label of z values which appears at the side of color bar.
                               [default: None, meaning do not show a label of z values]
        @param color           The color of scattered points. This color is also applied to linear
                               regression if argument `linear_regression` is True. This parameter is 
                               ignored when z is not None. In this case, the color of linear 
                               regression is set to blue.
                               [default: None, meaning follow a matplotlib's default color]
        @param lim             The limit of axis. This can be specified explicitly by
                               using tuples such as ((xmin, xmax), (ymin, ymax)).
                               If one passes float p, it calculate p%-percentile around median
                               for each of x-axis and y-axis.
                               [default: None, meaning do not set any limits]
        @equal_axis            If True, force ticks of x-axis and y-axis equal to each other.
                               [default: False]
        @linear_regression     If True, perform linear regression for x and y and plot a regression
                               line. If yerr is not None, perform the linear regression with
                               incorporating the error into the standard chi^2 and plot
                               a regression line with a 1-sigma allowed region.
                               [default: False]
        @reference_line        Draw a reference line. If reference_line == 'one-to-one', x=y is
                               drawn. If reference_line == 'zero', y=0 id drawn. A user-specific
                               function can be used by passing an object which has an attribute
                               '__call__' and returns a 1-d Numpy array.
        @returns                a matplotlib.figure.Figure object
        """
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)

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
            # Even if lim = None, we want to set limits. Limits set by matplotlib looks uneven probably because it seems to pick round numbers for the endpoints (eg -0.2 and 0.2).
            xlim = (numpy.min(x)-0.05*(numpy.max(x)-numpy.min(x)),
                    numpy.max(x)+0.05*(numpy.max(x)-numpy.min(x)))
            # We apply the same thing to y. However, when y has error, setting the limit may cut out error, so we just leave it.
            if yerr is None:
                ylim = (numpy.min(y)-0.05*(numpy.max(y)-numpy.min(y)),
                        numpy.max(y)+0.05*(numpy.max(y)-numpy.min(y)))
            else:
                ylim = None

        # plot
        if z is None:
            if yerr is None:
                p = ax.plot(x, y, ".%s" % color)
            else:
                p = ax.errorbar(x, y, yerr, fmt=".%s" % color)
            # store color for latter use
            used_color = p[0].get_color()
        else:
            if yerr is not None:
                plt.errorbar(x, y, yerr=yerr, linestyle="None", color = "k", zorder=0)
            plt.scatter(x, y, c=z, zorder=1)
            cb = plt.colorbar()
            used_color = "b"

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
                m, c, cov_m, cov_c, cov_mc = self.linearRegression(x, y, err = yerr)
                ax.plot(xtmp, m*xtmp+c, "--%s" % used_color)
                y = m*xtmp+c
                # calculate yerr using the covariance
                yerr = numpy.sqrt(xtmp**2*cov_m + 2.*xtmp*cov_mc + cov_c)
                ax.fill_between(xtmp, y-yerr, y+yerr, facecolor = used_color,
                                edgecolor = used_color, alpha =0.5)
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
        if zlabel is not None:
            cb.set_label(zlabel)

        fig.tight_layout()

        return fig

    def linearRegression(self, x, y, err = None):
        """
        Perform linear regression (y=mx+c). If error is given, it returns covariance.
        @param x               NumPy array for x.
        @param y               NumPy array for y.
        @param err             Numpy array for y error.
                               [default: None, meaning do not consider y error]
        @returns               m, c. If err is not None, m, c, cov_m, cov_c, cov_mc.
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

    def getStatisticsPerCCD(self, ccds, x, y, yerr = None, stat = "mean"):
        """
        Calculate average for x and y for each CCD.
        @param ccd             NumPy array for CCD, an array in which each element indicates CCD ID
                               of each data point.
        @param x               NumPy array for x.
        @param y               NumPy array for y.
        @param err             Numpy array for y error.
                               [default: None, meaning do not consider y error]
        @returns               x_ave, y_ave, y_ave_std.
        """
        if stat == "mean":
            x_ave = numpy.array([numpy.average(x[ccds == ccd]) for ccd in set(ccds)])
            if yerr is None:
                y_ave = numpy.array([numpy.average(y[ccds == ccd]) for ccd in set(ccds)])
                y_ave_std = numpy.array([numpy.std(y[ccds == ccd])/numpy.sqrt(len(y[ccds == ccd]))
                                         for ccd in set(ccds)])
                return x_ave, y_ave, y_ave_std
            # calculate y and its std under the inverse variance weight if yerr is given
            else:
                y_ave = numpy.array([numpy.sum(y[ccds == ccd]/yerr[ccds == ccd]**2)/
                                     numpy.sum(1./yerr[ccds == ccd]**2) for ccd in set(ccds)])
                y_ave_std = numpy.array([numpy.sqrt(1./numpy.sum(1./yerr[ccds == ccd]**2))
                                         for ccd in set(ccds)])
                return x_ave, y_ave, y_ave_std
        elif stat == "median":
            x_med = numpy.array([numpy.median(x[ccds == ccd]) for ccd in set(ccds)])
            y_med = numpy.array([numpy.median(y[ccds == ccd]) for ccd in set(ccds)])
            y_med_std = numpy.array([numpy.sqrt(numpy.pi/2.)*numpy.std(y[ccds == ccd])/numpy.sqrt(len(y[ccds == ccd]))
                                     for ccd in set(ccds)])
        else:
            raise ValueError('stat should be mean or median.')
            return x_med, y_med, y_med_std

class ScatterPlotStarVsPSFG1SysTest(ScatterPlotSysTest):
    short_name = 'scatterplot_star_vs_psf_g1'
    long_name = 'Make a scatter plot of star g1 vs psf g1'
    objects_list = ['star PSF']
    required_quantities = [('g1', 'g1_err', 'psf_g1')]

    def __call__(self, array, per_ccd_stat = 'None', color = '', lim=None):
        per_ccd_stat = None if per_ccd_stat == 'None' else per_ccd_stat
        if per_ccd_stat:
            psf_g1, g1, g1_err = self.getStatisticsPerCCD(array['CCD'], array['psf_g1'],
                                                          array['g1'], yerr = array['g1_err'],
                                                          stat = per_ccd_stat)
        else:
            psf_g1, g1, g1_err = array['psf_g1'], array['g1'], array['g1_err']
        return self.scatterPlot(psf_g1, g1, yerr=g1_err,
                                xlabel=r'$g^{\rm PSF}_1$', ylabel=r'$g^{\rm star}_1$',
                                color=color, lim=lim, equal_axis=False,
                                linear_regression=True, reference_line='one-to-one')

class ScatterPlotStarVsPSFG2SysTest(ScatterPlotSysTest):
    short_name = 'scatterplot_star_vs_psf_g2'
    long_name = 'Make a scatter plot of star g2 vs psf g2'
    objects_list = ['star PSF']
    required_quantities = [('g2', 'g2_err', 'psf_g2')]

    def __call__(self, array, per_ccd_stat = 'None', color = '', lim=None):
        per_ccd_stat = None if per_ccd_stat == 'None' else per_ccd_stat
        if per_ccd_stat:
            psf_g2, g2, g2_err = self.getStatisticsPerCCD(array['CCD'], array['psf_g2'],
                                                          array['g2'], yerr = array['g2_err'],
                                                          stat = per_ccd_stat)
        else:
            psf_g2, g2, g2_err = array['psf_g2'], array['g2'], array['g2_err']
        return self.scatterPlot(psf_g2, g2, yerr=g2_err,
                                xlabel=r'$g^{\rm PSF}_2$', ylabel=r'$g^{\rm star}_2$',
                                color=color, lim=lim, equal_axis=False,
                                linear_regression=True, reference_line='one-to-one')

class ScatterPlotStarVsPSFSigmaSysTest(ScatterPlotSysTest):
    short_name = 'scatterplot_star_vs_psf_sigma'
    long_name = 'Make a scatter plot of star sigma vs psf sigma'
    objects_list = ['star PSF']
    required_quantities = [('sigma', 'sigma_err', 'psf_sigma')]

    def __call__(self, array, per_ccd_stat = 'None', color = '', lim=None):
        per_ccd_stat = None if per_ccd_stat == 'None' else per_ccd_stat
        if per_ccd_stat:
            psf_sigma, sigma, sigma_err = self.getStatisticsPerCCD(array['CCD'], array['psf_sigma'],
                                                                   array['sigma'],
                                                                   yerr = array['sigma_err'],
                                                                   stat = per_ccd_stat)
        else:
            psf_sigma, sigma, sigma_err = array['psf_sigma'], array['sigma'], array['sigma_err']
        return self.scatterPlot(psf_sigma, sigma, yerr=sigma_err,
                                xlabel=r'$\sigma^{\rm PSF}$ [arcsec]',
                                ylabel=r'$\sigma^{\rm star}$ [arcsec]',
                                color=color, lim=lim, equal_axis=False,
                                linear_regression=True, reference_line='one-to-one')

class ScatterPlotResidualVsPSFG1SysTest(ScatterPlotSysTest):
    short_name = 'scatterplot_residual_vs_psf_g1'
    long_name = 'Make a scatter plot of residual g1 vs psf g1'
    objects_list = ['star PSF']
    required_quantities = [('g1', 'g1_err', 'psf_g1')]

    def __call__(self, array, per_ccd_stat = 'None', color = '', lim=None):
        per_ccd_stat = None if per_ccd_stat == 'None' else per_ccd_stat
        if per_ccd_stat:
            psf_g1, g1, g1_err = self.getStatisticsPerCCD(array['CCD'], array['psf_g1'],
                                                          array['g1'], yerr = array['g1_err'],
                                                          stat = per_ccd_stat)
        else:
            psf_g1, g1, g1_err = array['psf_g1'], array['g1'], array['g1_err']
        return self.scatterPlot(psf_g1, g1-psf_g1, yerr=g1_err,
                                xlabel=r'$g^{\rm PSF}_1$',
                                ylabel=r'$g^{\rm star}_1 - g^{\rm PSF}_1$',
                                color=color, lim=lim, equal_axis=False,
                                linear_regression=True, reference_line='zero')

class ScatterPlotResidualVsPSFG2SysTest(ScatterPlotSysTest):
    short_name = 'scatterplot_residual_vs_psf_g2'
    long_name = 'Make a scatter plot of residual g2 vs psf g2'
    objects_list = ['star PSF']
    required_quantities = [('g2', 'g2_err', 'psf_g2')]

    def __call__(self, array, per_ccd_stat = 'None', color = '', lim=None):
        per_ccd_stat = None if per_ccd_stat == 'None' else per_ccd_stat
        if per_ccd_stat:
            psf_g2, g2, g2_err = self.getStatisticsPerCCD(array['CCD'], array['psf_g2'],
                                                          array['g2'], yerr = array['g2_err'],
                                                          stat = per_ccd_stat)
        else:
            psf_g2, g2, g2_err = array['psf_g2'], array['g2'], array['g2_err']
        return self.scatterPlot(psf_g2, g2-psf_g2, yerr=g2_err,
                                xlabel=r'$g^{\rm PSF}_2$',
                                ylabel=r'$g^{\rm star}_2 - g^{\rm PSF}_2$',
                                color=color, lim=lim, equal_axis=False,
                                linear_regression=True, reference_line='zero')

class ScatterPlotResidualVsPSFSigmaSysTest(ScatterPlotSysTest):
    short_name = 'scatterplot_residual_vs_psf_sigma'
    long_name = 'Make a scatter plot of residual sigma vs psf sigma'
    objects_list = ['star PSF']
    required_quantities = [('sigma', 'sigma_err', 'psf_sigma')]

    def __call__(self, array, per_ccd_stat = 'None', color = '', lim=None):
        per_ccd_stat = None if per_ccd_stat == 'None' else per_ccd_stat
        if per_ccd_stat:
            psf_sigma, sigma, sigma_err = self.getStatisticsPerCCD(array['CCD'], array['psf_sigma'],
                                                                   array['sigma'],
                                                                   yerr = array['sigma_err'],
                                                                   stat = per_ccd_stat)
        else:
            psf_sigma, sigma, sigma_err = array['psf_sigma'], array['sigma'], array['sigma_err']
        return self.scatterPlot(psf_sigma, sigma-psf_sigma, yerr=sigma_err,
                                xlabel=r'$\sigma^{\rm PSF}$  [arcsec]',
                                ylabel=r'$\sigma^{\rm star} - \sigma^{\rm PSF}$  [arcsec]',
                                color=color, lim=lim, equal_axis=False, 
                                linear_regression=True, reference_line='zero')

