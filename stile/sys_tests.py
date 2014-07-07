"""@file sys_tests.py
Contains the class definitions of the Stile systematics tests.
"""
import numpy
import stile

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
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return None
        fields = data.dtype.names
        w = None
        for poss_r in ['<R>','R_nominal','R']:
            if poss_r in fields:
                r = poss_r
                break
        else:
            raise ValueError('No radius parameter found in data')
        # Could have: a T-mode Y and a B-mode Y; a real xi+ and xi-, then imaginary;
        # and possibly some separate t-mode Y and b-mode Y for the data and randoms alone rather
        # than together, in which case it's the keys poss_dr_y + 'd' or 'r' for data and randoms, 
        # respectively; plus an error bar.
        for poss_y, poss_yb, poss_y_im, poss_yb_im, poss_dr_y, poss_dr_yb, poss_w in [
            ('omega',None,None,None,None,None,'sig_omega'), # n2 style
            ('<gamT>','<gamX>',None,None,'gamT_','gamX_','sig'), # ng style
            ('xi+','xi-','xi+_im','xi-_im',None,None,'sig_xi'), # g2 style
            ('<kappa>',None,None,None,'kappa_',None,'sig'), # nk style
            ('xi',None,None,None,None,None,'sig_xi'), # k2 style
            ('<kgamT>','<kgamX>',None,None,'kgamT_','kgamX_','sig'), # kg style
            ('<Map^2>','<Mx^2>','<MMx>(a)','<Mmx>(b)',None,None,'sig_map'), # m2 style
            ('<NMap>','<NMx>',None,None,None,None,'sig_nmap') # nm style or norm style
            ]:
            if poss_y in fields:
                y = poss_y
                y_im = poss_y_im
                w = poss_w
                if plot_bmode:
                    yb = poss_yb
                    yb_im = poss_yb_im
                else:
                    yb = None
                    yb_im = None
                if plot_data_only or plot_random_only:
                    dr_y = poss_dr_y
                    dr_yb = poss_dr_yb
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
        if (y_im and y_b):  
            nrows = 2
        elif dr_y:
            nrows = 1 + plot_data_only + plot_random_only
        
        ax = fig.add_suplot(nrows,1,1)
        ax.plot(data[r],data[y],data[w],color=colors[0],title=y)
        if yb:
            ax.plot(data[r],data[yb],data[w],color=colors[1],title=yb)
        elif y_im:
            ax.plot(data[r],data[y_im],data[w],color=colors[1],title=y_im)
        ax.set_xscale('log')
        ax.set_yscale(yscale)
        ax.legend()
        if y_b and y_im:
            ax = fig.add_subplot(nrows,1,2)
            ax.plot(data[r],data[y_im],data[w],color=colors[0],title=y_im)
            ax.plot(data[r],data[yb_im],data[w],color=colors[1],title=yb_im)
        curr_plot = 1
        if plot_data_only and dr_y:
            curr_plot+=1
            ax = fig.add_subplot(nrows,1,curr_plot)
            ax.plot(data[r],data[dr_y+'d'],data[w],color=colors[0],title=y_im)
            if dr_yb:
                ax.plot(data[r],data[dr_yb+'d'],data[w],color=colors[1],title=yb_im)
            ax.set_xscale('log')
            ax.set_yscale(yscale)
            ax.legend()
        if plot_random_only and dr_y:
            curr_plot+=1
            ax = fig.add_subplot(nrows,1,curr_plot)
            ax.plot(data[r],data[dr_y+'r'],data[w],color=colors[0],title=y_im)
            if dr_yb:
                ax.plot(data[r],data[dr_yb+'r'],data[w],color=colors[1],title=yb_im)
            ax.set_xscale('log')
            ax.set_yscale(yscale)
            ax.legend()
        return fig
        
        
class RealShearSysTest(CorrelationFunctionSysTest):
    """
    Compute the tangential and cross shear around a set of real objects.
    """
    short_name = 'realshear'
    long_name = 'Shear of galaxies around real objects'

    def __call__(self,stile_args,data=None,data2=None,random=None,random2=None,**kwargs):
        return self.getCorrelationFunction(stile_args,'ng',data,data2,random,random2,**kwargs)

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
