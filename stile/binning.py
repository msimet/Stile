"""@file binning.py
Contains definitions of Bin* classes that generate binning schemes and the SingleBins
objects they create which can be applied to data to limit it to the bin in question.
"""

import numpy

class BinList:
    """
    An object which returns bin definitions (a list of SingleBins) following the bin edge
    definitions given as the input bin_list.  
    
    @param field     Which data field to apply the binning system to
    @param bin_list  A list of bin endpoints such that bin_list[0] <= (bin 0 data) < bin_list[1],
                     bin_list[1] <= (bin 1 data) < bin_list[2], etc; each interval is assumed to be
                     [low,high) and the list must be monotonic.
    """
    def __init__(self,field,bin_list):
        if not isinstance(field,str):
            raise TypeError('Field description must be a string. Passed value: '+str(field))
        if not bin_list:
            raise TypeError('Must pass a non-empty bin_list')
        self.field = field
        monotonic = numpy.array(bin_list[1:])-numpy.array(bin_list[:-1])
        if numpy.all(monotonic>0):
            self.reverse = False
        elif numpy.all(monotonic<0):
            self.reverse = True
            bin_list.reverse()
        else:
            raise ValueError('bin_list must be monotonically increasing or decreasing. Passed '+
                             'list: %s'%bin_list)
        self.bin_list = bin_list
    def __call__(self):
        return_list = [SingleBin(field=self.field,low=low,high=high,short_name=str(i)) 
                        for i, (low, high) in  enumerate(zip(self.bin_list[:-1],self.bin_list[1:]))]
        if self.reverse:
            return_list.reverse()
        return return_list

class BinStep:
    """
    An object which returns bin definitions (a list of SingleBins) following the simple 
    constant-step bins described by the input arguments. Can handle linear-space and log-space 
    binning (default is linear). AT LEAST THREE of the arguments (low,high,step,n_bins) must be 
    passed; if all four are passed they will be checked for consistency.  If low, high, and step are
    passed, high may be slightly increased to ensure an integer number of bins, so users who need
    a hard cutoff at high are suggested to use n_bins instead.
    
    @param field     Which field of data to apply the binning system to
    @param low       The low edge of the lowest bin, inclusive; should be in linear space regardless
                     of use_log (default: None)
    @param high      The high edge of the highest bin, exclusive; should be in linear space
                     regardless of use_log (default: None)
    @param step      The width of each bin (in linear space if use_log=False, in log space if 
                     use_log=True) (default: None)
    @param n_bins    The total number of bins requested; if float, will be converted to the next
                     largest integer (default: None)
    @param use_log   If True, bin in log space; else bin in linear space. Even when use_log=True,
                     all arguments except step should be given in linear space, and the returned
                     bin edges will also be in linear space. (default: False)
    """
    def __init__(self,field,low=None,high=None,step=None,n_bins=None,use_log=False):
        if not isinstance(field,str):
            raise TypeError('Field description must be a string. Passed value: '+str(field))
        self.field = field
        n_none = (low is None) + (high is None) + (step is None) + (n_bins is None)
        if n_none>1:
            raise TypeError('Must pass at least three of low, high, step, n_bins')
        if high==low:
            raise ValueError('High must be != low. Given arguments: %f %f'%(high,low))
        if step is not None and step==0:
            raise ValueError('Step must be nonzero. Given argument: %f'%step)
        if n_bins is not None and n_bins<=0:
            raise ValueError('n_bins must be positive. Given argument: %i'%n_bins)
        if n_bins and not isinstance(n_bins,int):
            if int(n_bins)==n_bins:
                n_bins==int(n_bins)
            else:
                n_bins = int(numpy.ceil(n_bins))
        if use_log:
            if (low is not None and low<=0) or (high is not None and high<=0):
                raise ValueError('Only positive arguments accepted for low and high if use_log. '+
                                 'Given arguments: %f, %f'%(low,high))
            if low:
                low = numpy.log(low)
            if high:
                high = numpy.log(high)
        self.use_log = use_log
        if low is not None:
            self.low = low
            if high is not None:
                if step:
                    if (high-low)*step<0:
                        raise ValueError('Argument step must have the same sign as (high-low). '+
                                        'Given arguments: high %f, low %f, step %f'%(high,low,step))
                    self.step = step
                    self.n_bins = int(numpy.ceil((high-low)/step))
                    if n_bins:
                        if n_bins!=self.n_bins:
                            raise ValueError('Cannot form a consistent binning with low %f, high '
                                             '%f, step %f, and n_bins %i--derived n_bins is %i.'
                                             %(low,high,step,n_bins,self.n_bins))
                else:
                    self.n_bins = n_bins
                    self.step = float(high-low)/(n_bins)     
            else:
                self.step = step
                self.n_bins = n_bins
        else:
            self.step = step
            self.n_bins = n_bins
            self.low = high-n_bins*step
        if self.step<0:
            self.low = self.low+self.n_bins*self.step
            self.step*=-1
            self.reverse = True
        else:
            self.reverse = False
    def __call__(self):
        if self.use_log:
            return_list = [SingleBin(field=self.field,low=numpy.exp(self.low+i*self.step),
                                     high=numpy.exp(self.low+(i+1)*self.step),
                                     short_name=str(i)) for i in range(self.n_bins)]
        else:
            return_list = [SingleBin(field=self.field,low=self.low+i*self.step,
                                     high=self.low+(i+1)*self.step,short_name=str(i)) 
                                     for i in range(self.n_bins)]
        if self.reverse:
            return_list.reverse()
        return return_list

        
class SingleBin:
    """
    A class that contains the information for one particular bin generated from one of the Bin* 
    classes. The attributes can be accessed directly for DataHandlers that read in the data 
    selectively. The class can also be called with a data array to generate an array of bools 
    such that array[SingleBin()] gives only the data within the bounds of the particular instance
    of the class.  The endpoints are assumed to be [low,high), that is, low <= data < high, with
    defined relational operators.  
    
    @param field    The index of the field containing the data to be binned (must be str)
    @param low      The lower edge of the bin (inclusive)
    @param high     The upper edge of the bin (exclusive)
    @param shorname A string denoting this bin in filenames
    @param long_name A string denoting this bin in program text outputs/plots (default: "low-high")  
    """
    def __init__(self,field,low,high,short_name,long_name=None):
        if not isinstance(field,str):
            raise TypeError('Field description must be a string. Passed value: '+str(field))
        if high < low:
            raise ValueError("High ("+str(high)+") must be greater than low ("+str(low)+")")
        if not isinstance(short_name,str) or (long_name and not isinstance(long_name,str)):
            raise TypeError("Short_name and long_name must be strings")
        self.field = field
        self.low = low
        self.high = high
        self.short_name = short_name
        if long_name:
            self.long_name = long_name
        else:
            self.long_name = str(low)+'-'+str(high)
    def __call__(self,data):
        """
        Given data, returns an array of bools such that array[SingleBin()] gives only the data 
        within the bounds [self.low,self.high).
        @param data   An array of data which can be indexed by self.field
        @returns      An array of bools indicating which of the data points are in the given range
        """
        return (data[self.field]>=self.low) & (data[self.field]<self.high)
    
class BinFunction:
    """
    An object which returns bin definitions (a list of SingleFunctionBins) following the definitions
    given in initialization.  Note that unlike other SingleBins, the SingleFunctionBins created
    by BinFunction do not have public field, low, or high attributes, since the function is assumed
    to be too complex for such parameterization.
    
    @param function       The function to be applied to the data (an entire array, not just a 
                          field).  The function should return an array of ints corresponding to the
                          bin numbers of the data rows (unless return_bools is set, see below).  The
                          function should take a data array as its only argument, unless
                          return_bools is set to True, in which case it should take a bin number as
                          the second argument.
    @param n_bins         The maximum number of bins returned by the input function.  If None, the
                          function will be checked for an n_bins attribute; if it does not exist
                          an error will be raised.
    @param returns_bools  True if the function will return an array of bools corresponding to a
                          mask to the bin number in question; false otherwise.  (default: False) 
    """
    def __init__(self, function, n_bins=None, returns_bools=False):
        self.function = function
        
        if n_bins is None: # Dunno why somebody'd do 0 bins, but it's fine, I guess...
            try:
                self.n_bins = function.n_bins
            except:
                raise TypeError("Argument n_bins must be passed directly or as the attribute "+
                                 "function.n_bins!")
        else:
            self.n_bins = n_bins
        self.returns_bools = returns_bools
    def __call__(self):
        return [SingleFunctionBin(self.function,i,self.returns_bools) for i in range(self.n_bins)]

class SingleFunctionBin(SingleBin):
    """
    A class that contains the information for one particular bin generated from a function. The 
    class can also be called with a data array to generate an array of bools such that 
    array[SingleBin()] gives only the data within the bounds of the particular instance of the 
    class.  Unlike SingleBins, there are no public field, low, or high attributes, as these are
    assumed to be insufficient to describe the behavior of the binning scheme.    
    
    @param function       The function that returns the bin information
    @param n              Which bin this SingleFunctionBin considers
    @param returns_bools  True if the function returns bools, else False (default: False)
    @param short_name     A string denoting this bin in filenames (default: str(n))
    @param long_name       A string denoting this bin in program outputs/plots (default: short_name)  
    """
    def __init__(self,function,n,returns_bools=False, short_name=None, long_name=None):
        if (short_name and not isinstance(short_name,str)) or (
                long_name and not isinstance(long_name,str)):
            raise TypeError("short_name and long_name must be strings")
        if short_name is not None:
            self.short_name = short_name
        else:
            self.short_name = str(n)
        if long_name is not None:
            self.long_name = long_name
        else:
            self.long_name = self.short_name
        self.function=function
        self.n=n
        if returns_bools:
            self.__call__=self._call_bool
        else:
            self.__call__=self._call_int
    def _call_int(self,data):
        return self.function(data)==self.n
    def _call_bool(self,data):
        return self.function(data,self.n)    
        
