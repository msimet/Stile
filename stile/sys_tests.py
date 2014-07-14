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

class SysTest:
    """
    A SysTest is a lensing systematics test of some sort.  It should define the following 
    attributes:
        short_name = a string that can be used in filenames to denote this systematics test
        long_name = a string to denote this systematics test within program text outputs
    
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
        
class CorrelationFunctionSysTest(SysTest):
    short_name = 'corrfunc'
    """
    A base class for the Stile systematics tests that use correlation functions. This implements the
    class method getCorrelationFunction, which runs corr2 (via a call to the subprocess module) on
    a given set of data.  Exact arguments to this method should be created by child classes of
    CorrelationFunctionSysTest; see the docstring for 
    CorrelationFunctionSysTest.getCorrelationFunction for information on how to write further 
    tests using it.
    """
    def getCorrelationFunction(self, stile_args, correlation_function_type, data=None, data2=None,
                                     random=None, random2=None, config_here=False, **kwargs):
        """
        Sets up and calls corr2 on the given set of data.  The data files and random files can
        be contained already in stile_args['corr2_kwargs'] or **kwargs, in which case passing None
        to the `data` and `random` kwargs is fine; otherwise they should be properly populated.
        
        Note: by default, the corr2 configuration files are written to the temp directory called by 
        tempfile.mkstemp().  If you need to example the corr2 config files, you can pass 
        `config_here=True` and they will be written (as temp files probably beginning with "tmp") 
        to your working directory, which shouldn't be automatically cleaned up.  
        
        @param stile_args    The dict containing the parameters that control Stile's behavior
        @param correlation_function_type The type of correlation function ('n2','ng','g2','nk','k2',
                             'kg','m2','nm','norm') to request from corr2.
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
        
        # nab the corr2 params, and make the files if needed
        if not 'corr2_kwargs' in stile_args:
            stile_args = stile.corr2_utils.AddCorr2Dict(stile_args)
        corr2_kwargs = copy.deepcopy(stile_args['corr2_kwargs'])
        corr2_kwargs.update(kwargs)
        corr2_file_kwargs = stile.MakeCorr2FileKwargs(data,data2,random,random2)
        corr2_kwargs.update(corr2_file_kwargs)

        # make sure the set of non-None data sets makes sense
        if not ('file_list' in corr2_kwargs or 'file_name' in corr2_kwargs):
            raise ValueError("stile_args['corr2_kwargs'] or **kwargs must contain a file kwarg")
        if ('rand_list' in corr2_kwargs or 'rand_name' in corr2_kwargs):
            if ('file_name2' in corr2_kwargs or 'file_list2' in corr2_kwargs) :
                if not ('rand_list2' in corr2_kwargs or 'rand_name2' in corr2_kwargs):
                    raise ValueError('Given random file for file 1 but not file 2')
            elif ('rand_list2' in corr2_kwargs or 'rand_name2' in corr2_kwargs):
                raise ValueError('Given random file for file 2, but there is no file 2')
        elif ('rand_list2' in corr2_kwargs or 'rand_name2' in corr2_kwargs):
            raise ValueError('Given random file for file 2, but not file 1')
            
        if config_here:
            handle, config_file = tempfile.mkstemp(dir='.')
        else:
            handle, config_file = tempfile.mkstemp()
        handles.append(handle)
        handle, output_file = tempfile.mkstemp()
        handles.append(handle)

        corr2_kwargs[correlation_function_type+'_file_name'] = output_file
        
        stile.WriteCorr2ConfigurationFile(config_file,corr2_kwargs)
        
        #TODO: don't hard-code the name of corr2!
        subprocess.check_call(['corr2', config_file])

        return_value = stile.ReadCorr2ResultsFile(output_file)
        for handle in handles:  
            os.close(handle)
        return return_value
    
    def plot(self,data,colors=['r','b'],log_yscale=False,
                  plot_bmode=True,plot_data_only=True,plot_random_only=True):
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
        for t_r in ['<R>','R_nominal','R']:
            if t_r in fields:
                r = t_r
                break
        else:
            raise ValueError('No radius parameter found in data')
        
        # Logarithmic x-axes have stupid default ranges: fix this.
        rstep = data[r][1]/data[r][0]
        xlim = [min(data[r])/rstep,max(data[r])*rstep]    
        # Check what kind of data is in the array that .plot() received.  This annoyingly large list
        # contains all the possible sets of data, in tuples with the array field name and the
        # corresponding legend labels, plus error bars (w) and y-axis titles.  In order, the
        # elements of each tuple are:
        # a T-mode Y and a B-mode Y [or xi+ and xi-];
        # an imaginary xi+ and xi-;
        # a  separate t-mode Y and b-mode Y for the data and randoms alone rather
        # than together, in which case it's the keys t_dr_y + 'd' or 'r' for data and randoms, 
        # respectively, or + 'd}$' or 'r}$' for the legend labels;
        # error bar and y-axis title.
        # Each type of data may have only some of those elements--if not the item is None.
        for t_y, t_yb, t_y_im, t_yb_im, t_dr_y, t_dr_yb, t_w, t_ytitle in [
            (('omega','$\omega$'),None,None,None,None,None,'sig_omega',"$\omega$"), # n2
            (('<gamT>',r'$\langle \gamma_T \rangle$'),
             ('<gamX>',r'$\langle \gamma_X \rangle$'),None,None,
             ('gamT_','$\gamma_{T'),('gamX_','$\gamma_{X'),'sig',"$\gamma$"), # ng
            (('xi+',r'$\xi_+$'),('xi-',r'$\xi_-$'),
             ('xi+_im',r'$\xi_{+,im}$'),('xi-_im',r'$\xi_{-,im}$'),None,None,'sig_xi',r"$\xi$"), #g2
            (('<kappa>',r'$\langle \kappa \rangle$'),None,None,None,
             ('kappa_','$kappa_{'),None,'sig',"$\kappa$"), # nk 
            (('xi',r'$\xi$'),None,None,None,None,None,'sig_xi',r"$\xi$"), # k2 
            (('<kgamT>',r'$\langle \kappa \gamma_T\rangle$'),
             ('<kgamX>',r'$\langle \kappa \gamma_X\rangle$'),None,None,
             ('kgamT_',r'$\kappa \gamma_{T'),('kgamX_',r'$\kappa \gamma_{X'),
             'sig',"$\kappa\gamma$"), # kg 
            (('<Map^2>',r'$\langle M_{ap}^2 \rangle$'),('<Mx^2>',r'$\langle M_x^2\rangle$'),
             ('<MMx>(a)',r'$\langle MM_x \rangle(a)$'),('<Mmx>(b)',r'$\langle MM_x \rangle(b)$'),
             None,None,'sig_map', "$M_{ap}^2$"), # m2 
            (('<NMap>',r'$\langle NM_{ap} \rangle$'),('<NMx>',r'$\langle NM_{x} \rangle$'),
             None,None,None,None,'sig_nmap',"$NM_{ap}$") # nm or norm
            ]:
            # Pick the one the data contains and use it; break before trying the others.
            if t_y[0] in fields:
                y = t_y
                y_im = t_y_im if t_y_im and t_y_im[0] in fields else None
                w = t_w
                ytitle = t_ytitle
                if plot_bmode:
                    yb = t_yb if t_yb and t_yb[0] in fields else None
                    yb_im = t_yb_im if t_yb_im and t_yb_im[0] in fields else None
                else:
                    yb = None
                    yb_im = None
                if plot_data_only or plot_random_only:
                    dr_y = t_dr_y if t_dr_y and t_dr_y[0] in fields else None
                    dr_yb = t_dr_yb if t_dr_yb and t_dr_yb[0] in fields else None
                else:
                    dr_y = None
                    dr_yb = None
                break
        else:
            raise ValueError("No valid y-values found in data")
        if log_yscale:
            yscale = 'log'
        else:
            yscale = 'linear'
        fig = plt.figure()
        # Figure out how many plots you'll need--never more than 3, so we just use a stacked column.
        if (y_im and yb):  
            nrows = 2
        elif dr_y:
            nrows = 1 + plot_data_only + plot_random_only
        else:
            nrows = 1

        # Plot the first thing
        ax = fig.add_subplot(nrows,1,1)
        ax.errorbar(data[r],data[y[0]],yerr=data[w],color=colors[0],label=y[1])
        if yb:
            ax.errorbar(data[r],data[yb[0]],yerr=data[w],color=colors[1],label=yb[1])
        elif y_im: # Plot y and y_im if you're not plotting yb (else it goes on a separate plot)
            ax.errorbar(data[r],data[y_im[0]],yerr=data[w],color=colors[1],label=y_im[1])
        ax.set_xscale('log')
        ax.set_yscale(yscale)
        ax.set_xlim(xlim)
        ax.set_xlabel(r)
        ax.set_ylabel(ytitle)
        ax.legend()
        if yb and y_im: # Both yb and y_im: plot (y,yb) on one plot and (y_im,yb_im) on the other.
            ax = fig.add_subplot(nrows,1,2)
            ax.errorbar(data[r],data[y_im[0]],yerr=data[w],color=colors[0],label=y_im[1])
            ax.errorbar(data[r],data[yb_im[0]],yerr=data[w],color=colors[1],label=yb_im[1])
            ax.set_xscale('log')
            ax.set_yscale(yscale)
            ax.set_xlim(xlim)
            ax.set_xlabel(r)
            ax.set_ylabel(ytitle)
            ax.legend()
        if plot_data_only and dr_y: # Plot the data-only measurements if requested
            curr_plot+=1
            ax = fig.add_subplot(nrows,1,2)
            ax.errorbar(data[r],data[dr_y[0]+'d'],yerr=data[w],color=colors[0],
                        label=dr_y[1]+'d}$')
            if dr_yb:
                ax.errorbar(data[r],data[dr_yb[0]+'d'],yerr=data[w],color=colors[1],
                        label=dr_yb[1]+'d}$')
            ax.set_xscale('log')
            ax.set_yscale(yscale)
            ax.set_xlim(xlim)
            ax.set_xlabel(r)
            ax.set_ylabel(ytitle)
            ax.legend()
        if plot_random_only and dr_y: # Plot the randoms-only measurements if requested
            ax = fig.add_subplot(nrows,1,nrows)
            ax.errorbar(data[r],data[dr_y[0]+'r'],yerr=data[w],color=colors[0],
                        label=dr_y[1]+'r}$')
            if dr_yb:
                ax.errorbar(data[r],data[dr_yb[0]+'r'],yerr=data[w],color=colors[1],
                        label=dr_yb[1]+'r}$')
            ax.set_xscale('log')
            ax.set_yscale(yscale)
            ax.set_xlim(xlim)
            ax.set_xlabel(r)
            ax.set_ylabel(ytitle)
            ax.legend()
        return fig 
        
        
class RealShearSysTest(CorrelationFunctionSysTest):
    """
    Compute the tangential and cross shear around a set of real objects.
    """
    short_name = 'real_shear'
    long_name = 'Shear of galaxies around real objects'
    objects_list = ['galaxy lens','galaxy']
    required_quantities = [('ra','dec'),('ra','dec','g1','g2','w')]

    def __call__(self,stile_args,data=None,data2=None,random=None,random2=None,**kwargs):
        return self.getCorrelationFunction(stile_args,'ng',data,data2,random,random2,**kwargs)

class BrightStarShearSysTest(CorrelationFunctionSysTest):
    """
    Compute the tangential and cross shear around a set of bright stars.
    """
    short_name = 'star_shear'
    long_name = 'Shear of galaxies around bright stars'
    objects_list = ['star bright','galaxy']
    required_quantities = [('ra','dec'),('ra','dec','g1','g2','w')]

    def __call__(self,stile_args,data=None,data2=None,random=None,random2=None,**kwargs):
        return self.getCorrelationFunction(stile_args,'ng',data,data2,random,random2,**kwargs)

class StarXGalaxyDensitySysTest(CorrelationFunctionSysTest):
    """
    Compute the number density of galaxies around stars.
    """
    short_name = 'star_x_galaxy_density'
    long_name = 'Density of galaxies around stars'
    objects_list = ['star','galaxy']
    required_quantities = [('ra','dec'),('ra','dec')]

    def __call__(self,stile_args,data=None,data2=None,random=None,random2=None,**kwargs):
        return self.getCorrelationFunction(stile_args,'n2',data,data2,random,random2,**kwargs)

class StarXGalaxyShearSysTest(CorrelationFunctionSysTest):
    """
    Compute the cross-correlation of galaxy and star shapes.
    """
    short_name = 'star_x_galaxy_shear'
    long_name = 'Cross-correlation of galaxy and star shapes'
    objects_list = ['star','galaxy']
    required_quantities = [('ra','dec','g1','g2','w'),('ra','dec','g1','g2','w')]

    def __call__(self,stile_args,data=None,data2=None,random=None,random2=None,**kwargs):
        return self.getCorrelationFunction(stile_args,'g2',data,data2,random,random2,**kwargs)

class StarAutoShearSysTest(CorrelationFunctionSysTest):
    """
    Compute the auto-correlation of star shapes.
    """
    short_name = 'star_auto_shear'
    long_name = 'Auto-correlation of star shapes'
    objects_list = ['star']
    required_quantities = [('ra','dec','g1','g2','w')]

    def __call__(self,stile_args,data=None,data2=None,random=None,random2=None,**kwargs):
        return self.getCorrelationFunction(stile_args,'g2',data,data2,random,random2,**kwargs)

class RoweISysTest(CorrelationFunctionSysTest):
    """
    Compute the auto-correlation of (star-PSF model) shapes.
    """
    short_name = 'psf_residual_auto_shear'
    long_name = 'Auto-correlation of (star-PSF model) shapes'
    objects_list = ['star']
    required_quantities = [('ra','dec','g1_residual','g2_residual','w')]

    def __call__(self,stile_args,data=None,data2=None,random=None,random2=None,**kwargs):
        return self.getCorrelationFunction(stile_args,'g2',data,data2,random,random2,**kwargs)

class RoweIISysTest(CorrelationFunctionSysTest):
    """
    Compute the cross-correlation of (star-PSF model) residuals with star shapes.
    """
    short_name = 'psf_residual_x_star_shear'
    long_name = 'Cross-correlation of (star-PSF model) residuals with star shapes'
    objects_list = ['star','star']
    required_quantities = [('ra','dec','g1_residual','g2_residual','w'),('ra','dec','g1','g2','w')]

    def __call__(self,stile_args,data=None,data2=None,random=None,random2=None,**kwargs):
        raise NotImplementedError("Need to figure out how to tell corr2 to use the residuals!")
        return self.getCorrelationFunction(stile_args,'g2',data,data2,random,random2,**kwargs)

class GalaxyDensityCorrelationSysTest(CorrelationFunctionSysTest):
    """
    Compute the galaxy position autocorrelations.
    """
    short_name = 'galaxy_density'
    long_name = 'Galaxy position autocorrelation'
    objects_list = ['galaxy']
    required_quantities = [('ra','dec')]

    def __call__(self,stile_args,data=None,data2=None,random=None,random2=None,**kwargs):
        return self.getCorrelationFunction(stile_args,'n2',data,data2,random,random2,**kwargs)

class StarDensityCorrelationSysTest(CorrelationFunctionSysTest):
    """
    Compute the star position autocorrelations.
    """
    short_name = 'star_density'
    long_name = 'Star position autocorrelation'
    objects_list = ['star']
    required_quantities = [('ra','dec')]

    def __call__(self,stile_args,data=None,data2=None,random=None,random2=None,**kwargs):
        return self.getCorrelationFunction(stile_args,'n2',data,data2,random,random2,**kwargs)


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
                                   (use_field,use_array.dtype.fields.keys()))
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

    def WhiskerPlot(self, x, y, g1, g2):
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)

        # mask data with nan
        sel = numpy.logical_and.reduce(
            [numpy.isnan(x) == False, numpy.isnan(y) == False,
             numpy.isnan(g1) == False, numpy.isnan(g2) == False])
        x = x[sel]
        y = y[sel]
        g1 = g1[sel]
        g2 = g2[sel]

        # plot
        g = numpy.sqrt(g1*g1+g2*g2)
        theta = numpy.arctan2(g2,g1)/2
        gx = g * numpy.cos(theta)
        gy = g * numpy.sin(theta)
        q = ax.quiver(x, y, gx, gy, headwidth = 0., headlength = 0., headaxislength = 0.,
                      pivot = 'middle', width = 0.0005, scale = 1.)
        ruler = 0.05
        qk = plt.quiverkey(q, 0.5, 0.92, ruler, 'ellipticity: %s' % str(ruler), labelpos='W')
        plt.xlabel("x")
        plt.ylabel("y")

        return fig

class WhiskerPlotStarSysTest(WhiskerPlotSysTest):
    short_name = 'whiskerplot_star'
    long_name = 'Make a Whisker plot of stars'
    objects_list = ['star PSF']
    required_quantities = [('x','y','g1','g2')]

    def __call__(self, array):
        return self.WhiskerPlot(array['x'], array['y'], array['g1'], array['g2'])
    
