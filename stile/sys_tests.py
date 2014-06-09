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
    __call__(self, stile_args, data_handler, data, **kwargs) = run the SysTest.  **kwargs may 
    include data2 (source data set for lens-source pairs), random and random2 (random data sets 
    corresponding to data and data2), bin_list (list of SingleBins already applied to the data).
    """
    short_name = ''
    long_name = ''
    def __init__(self):
        pass
    def __call__(self):
        raise NotImplementedError()
        
class CorrelationFunctionSysTest(SysTest):
    """
    A base class for the Stile systematics tests that use correlation functions. This implements the
    class method get_correlation_function, which runs corr2 (via a call to the subprocess module) on
    a given set of data.  Exact arguments to this method should be created by child classes of
    CorrelationFunctionSysTest; see the docstring for 
    CorrelationFunctionSysTest.get_correlation_function for information on how to write further 
    tests using it.
    """
    def getCorrelationFunction(self, stile_args, dh, correlation_function_type, data, data2=None, 
                                 random=None, random2=None, **kwargs):
        """
        Sets up and calls corr2 on the given set of data.
        @param stile_args    The dict containing the parameters that control Stile's behavior
        @param correlation_function_type The type of correlation function ('n2','ng','g2','nk','k2',
                             'kg','m2','nm','norm') to request from corr2.
        @param dh            A DataHandler object describing the data set given in the data lists
                             below.
        @param data          A tuple whose first element is a string "name" or "list", corresponding
                             to the corr2 arg to write to, and whose second element is the name of a
                             file that exists in the filesystem.
        @param data2         If this is a cross-correlation, two sets of data are required; this 
                             kwarg should contain the second set in the same format as data. 
                             (default: None)
        @param random        A random data set corresponding to the contents of data, in the same 
                             format. (default: None)
        @param random2       A random data set corresponding to the contents of data2, in the same
                             format. (default: None)
        @param kwargs        Any other corr2 parameters to be written to the corr2 param file.
        @returns             a numpy array of the corr2 outputs.
        """
        #TODO: know what kinds of data this needs and make sure it has it
        import tempfile
        import subprocess
        import os
        
        file_handles = []
        delete_files = []
        
        corr2_kwargs = stile_args['corr2_kwargs']
        corr2_kwargs.update(kwargs) # TODO: Don't know if this will work if we actually pass kwargs
        corr2_kwargs['file_'+data[0]] = data[1]
        if data2:
            corr2_kwargs['file_'+data2[0]+'2'] = data2[1]
        if random:
            corr2_kwargs['rand_'+random[0]] = random[1]
        if random2:
            if data:
                corr2_kwargs['rand_'+random2[0]+'2'] = random2[1]
            else:
                raise ValueError("random2 data set passed without corresponding data2 data set!")

        handle, config_file = tempfile.mkstemp(dir=dh.temp_dir)
        file_handles.append(handle)
        delete_files.append(config_file)
        if 'bins_name' in stile_args:
            output_file = dh.getOutputPath(self.short_name+stile_args['bins_name'])
        else:
            output_file = dh.getOutputPath(self.short_name)
        corr2_kwargs[correlation_function_type+'_file_name'] = output_file
        stile.WriteCorr2ConfigurationFile(config_file,corr2_kwargs)
        
        #TODO: don't hard-code the name of corr2!
        subprocess.check_call(['corr2', config_file])

        return_value  = stile.ReadCorr2ResultsFile(output_file)
        for handle in file_handles:
            os.close(handle)
        for file_name in delete_files:
            os.remove(file_name)
        return return_value
        
class RealShearSysTest(CorrelationFunctionSysTest):
    short_name = 'realshear'
    long_name = 'Shear of galaxies around real objects'

    def __call__(self,stile_args,dh,data,data2,random=None,random2=None):
        corr2_kwargs = stile_args['corr2_kwargs']
        return self.getCorrelationFunction(stile_args,dh,'ng',data,data2,random,random2,
                                              **corr2_kwargs)

class StatsSysTest(SysTest):
    """
    A class for the Stile systematics tests that use basic statistical quantities. It uses NumPy
    routines for all the innards, and saves the results in a stile.Stats object (basically a dict;
    see stile_utils.py) that can carry around the information, print the results in a useful format,
    write to file, or (eventually) become an argument to plotting routines that might output some of
    the results on plots.

    One of the calculations it does is find the percentiles of the given quantity.  The percentile
    levels to use can be set when the StatsSysTest is initialized, or when it is called.  These
    percentiles must be provided as an iterable (list, tuple, or NumPy array).
    """
    short_name = 'stats'
    long_name = 'Calculate basic statistics of a given quantity'

    def __init__(self, percentiles=[2.2, 16., 50., 84., 97.8]):
        self.percentiles = percentiles

    def __call__(self, array, percentiles=None):
        """Calling a StatsSysTest with a given array argument as `array` will cause it to carry out
        all the statistics test and populate a stile.Stats object with the results, which it returns
        to the user.  The `array` can in principle be multi-dimensional, but that detail is ignored;
        if you want to get statistics of one column in a 2d array, for example, then for `array` you
        should pass in that particular column.

        The user can optionally pass in percentiles at which it wants to get values; if None, then
        it uses the values that were used when initializing the StatsSysTest object.
        """
        # Set the percentile levels, if the user provided them.  Otherwise use what was set up at
        # the time of initialization.
        if percentiles is not None: self.percentiles = percentiles

        # Check to make sure that percentiles is iterable (list, numpy array, tuple, ...)
        if not hasattr(self.percentiles, '__iter__'):
            raise RuntimeError('List of percentiles is not an iterable (list, tuple, NumPy array)!')

        # Collapse multi-dimensional arrays to 1d

        # Create the output object, a stile.Stats() object.

        # Populate the basic entries, like median, mean, standard deviation.
        # Populate things like skewness, kurtosis.  (Check list of NumPy options for more things
        # that could be useful.)
        # Populate the percentiles and values.
        # Return.

