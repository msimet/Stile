"""@file tests.py
Contains the class definitions of the Stile systematics tests.
"""
import numpy
import stile

class Test:
    """
    A Test is a lensing systematics test of some sort.  It should define the following attributes:
    short_name = a string that can be used in filenames to denote this test
    long_name = a string to denote this test within program text outputs
    
    It should define the following methods:
    __call__(self, stile_args, data_handler, data, **kwargs) = run the test.  **kwargs may include 
        data2 (source data set for lens-source pairs), random and random2 (random data sets 
        corresponding to data and data2), bin_list (list of SingleBins already applied to the data).
    """
    short_name = ''
    long_name = ''
    def __init__(self):
        pass
    def __call__(self):
        raise NotImplementedError()
        
class CorrelationFunctionTest(Test):
    """
    A base class for the Stile tests that use correlation functions.  This implements the class 
    method get_correlation_function, which runs corr2 (via a call to the subprocess module) on a 
    given set of data.  Exact arguments to this method should be created by child classes of
    CorrelationFunctionTest; see the docstring for CorrelationFunctionTest.get_correlation_function
    for information on how to write further tests using it.
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
        @returns             a numpy.recarray of the corr2 outputs.
        """
        #TODO: know what kinds of data this needs and make sure it has it
        import tempfile
        import subprocess
        import os
        
        file_handles = []
        delete_files = []
        
        corr2_options = stile_args['corr2_options']
        corr2_options.update(kwargs) # TODO: Don't know if this will work if we actually pass kwargs
        corr2_options['file_'+data[0]] = data[1]
        if data2:
            corr2_options['file_'+data2[0]+'2'] = data2[1]
        if random:
            corr2_options['rand_'+random[0]] = random[1]
        if random2:
            if data:
                corr2_options['rand_'+random2[0]+'2'] = random2[1]
            else:
                raise ValueError("random2 data set passed without corresponding data2 data set!")

        handle, param_file = tempfile.mkstemp(dir=dh.temp_dir)
        file_handles.append(handle)
        delete_files.append(param_file)
        if 'bins_name' in stile_args:
            output_file = dh.getOutputPath(self.short_name+stile_args['bins_name'])
        else:
            output_file = dh.getOutputPath(self.short_name)
        corr2_options[correlation_function_type+'_file_name'] = output_file
        stile.WriteCorr2ParamFile(param_file,corr2_options)
        
        #TODO: don't hard-code the name of corr2!
        subprocess.check_call(['corr2', param_file])

        return_value  = stile.ReadCorr2ResultsFile(output_file)
        for handle in file_handles:
            os.close(handle)
        for file_name in delete_files:
            os.remove(file_name)
        return return_value
        
class TestXShear(CorrelationFunctionTest):
    short_name = 'realshear'
    long_name = 'Shear of galaxies around real objects'

    def __call__(self,stile_args,dh,data,data2,random=None,random2=None):
        corr2_options = stile_args['corr2_options']
        return self.getCorrelationFunction(stile_args,dh,'ng',data,data2,random,random2,
                                              **corr2_options)

