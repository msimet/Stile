"""
Contains definitions of Bin* classes that generate binning schemes and the SingleBin
objects they create which can be applied to data to limit it to the bin in question.
"""
#TODO: binning for images
import numpy


class BinList:
    """
    An object which, when called, returns bin definitions (a list of :class:`SingleBin`\s) following
    the bin edge definitions given as the input ``bin_list``.  The bins are defined as an input
    iterable, which is assumed to contain the bin edges in order.

    :param field:     Data field to which the binning system should be applied, such as ``ra`` or
                      ``z``.
    :param bin_list:  A list of bin endpoints such that
                      ``bin_list[0] <= (bin 0 data) < bin_list[1]``,
                      ``bin_list[1] <= (bin 1 data) < bin_list[2]`` if the ``bin_list`` is
                      monotonically increasing, or ``bin_list[0] > (bin 0 data) >= bin_list[1]``,
                      ``bin_list[1] > (bin 1 data) >= bin_list[2]`` if the ``bin_list`` is
                      monotonically decreasing.  Nonmonotonic `bin_list`\s will raise an error.
    :returns:         A list of :class:`SingleBin` objects determined by the input criteria.
    """

    def __init__(self, bin_list, field=None):
        if not isinstance(field, str) and field is not None:
            raise TypeError('Field description must be a string or None. Passed value: '+str(field)+
                              ' of type'+str(type(field)))
        if not bin_list:
            raise TypeError('Must pass a non-empty bin_list')
        self.field = field
        monotonic = numpy.array(bin_list[1:])-numpy.array(bin_list[:-1])
        if numpy.all(monotonic > 0):
            self.reverse = False
        elif numpy.all(monotonic < 0):
            self.reverse = True
            bin_list.reverse()
        else:
            raise ValueError('bin_list must be monotonically increasing or decreasing. Passed '+
                             'list: %s'%bin_list)
        self.bin_list = bin_list

    def __call__(self):
        """
        Returns a list of :class:`SingleBin` objects following the definitions provided when the
        class was created.
        """
        if self.field is not None:
            bin_type = SingleBin
        else:
            bin_type = SingleBinFieldless
        return_list = [bin_type(field=self.field, low=low, high=high, short_name=str(i))
                       for i, (low, high) in  enumerate(zip(self.bin_list[:-1], self.bin_list[1:]))]
        if self.reverse:
            return_list.reverse()
        return return_list


class BinStep:
    """
    An object which, when called, returns bin definitions (a list of :class:`SingleBin`\s) following
    the simple constant-step bins described by the input arguments. Can handle linear-space and log-
    space binning (default is linear). AT LEAST THREE of the arguments (``low``, ``high``, ``step``,
    ``n_bins``) must be passed; if all four are passed they will be checked for consistency.  If
    ``low``, ``high``, and ``step`` are passed, ``high`` may be slightly increased to ensure an
    integer number of bins, so users who need a hard cutoff at ``high`` should use ``n_bins``
    instead.

    :param field:     Data field to which the binning system should be applied.
    :param low:       The low edge of the lowest bin, inclusive; should be in linear space
                      regardless of ``use_log`` [default: None].
    :param high:      The high edge of the highest bin, exclusive; should be in linear space
                      regardless of ``use_log`` [default: None].
    :param step:      The width of each bin (in linear space if ``use_log=False``, in base-e log
                      space if ``use_log=True``) [default: None].
    :param n_bins:    The total number of bins requested; if float, will be converted to an
                      integer with ``numpy.ceil()``.  [default: None].
    :param use_log:   If True, bin in log space; else bin in linear space. Even when
                      ``use_log=True``, all arguments except ``step`` should be given in linear
                      space, and the returned bin edges will also be in linear space.
                      [default: False]
    :returns:         A list of :class:`SingleBin` objects determined by the input criteria.
    """

    def __init__(self, field=None, low=None, high=None, step=None, n_bins=None, use_log=False):
        if not isinstance(field, str):
            raise TypeError('Field description must be a string. Passed value: '+str(field)+
                              ' of type'+str(type(field)))
        self.field = field
        n_none = (low is None) + (high is None) + (step is None) + (n_bins is None)
        if n_none > 1:
            raise TypeError('Must pass at least three of low, high, step, n_bins')
        if high == low:
            raise ValueError('High must be != low. Given arguments: (high, low) = (%f, %f)'%
                             (high, low))
        if step is not None and step == 0:
            raise ValueError('Step must be nonzero. Given argument: %f'%step)
        if n_bins is not None and n_bins <= 0:
            raise ValueError('n_bins must be positive. Given argument: %i'%n_bins)
        if n_bins and not isinstance(n_bins, int):
            if int(n_bins) == n_bins:
                n_bins = int(n_bins)
            else:
                n_bins = int(numpy.ceil(n_bins))
        if use_log:
            if (low is not None and low <= 0) or (high is not None and high <= 0):
                raise ValueError('Only positive arguments accepted for low and high if '+
                                 'use_log. Given arguments: (low, high) = (%f, %f)'%(low, high))
            if low:
                low = numpy.log(low)
            if high:
                high = numpy.log(high)
        self.use_log = use_log
        if low is not None:
            self.low = low
            if high is not None:
                if step:
                    if (high-low)*step < 0:
                        raise ValueError('Argument step must have the same sign as (high-low). '+
                                         'Given arguments: high %f, low %f, step %f'%
                                         (high, low, step))
                    self.step = step
                    self.n_bins = int(numpy.ceil((high-low)/step))
                    if n_bins:
                        if n_bins != self.n_bins:
                            raise ValueError('Cannot form a consistent binning with low %f, high '
                                             '%f, step %f, and n_bins %i--derived n_bins is %i.'
                                             %(low, high, step, n_bins, self.n_bins))
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
        if self.step < 0:
            # We want to store parameters such that bin_edge[0] <= bin 0 < bin_edge[1], even if
            # the step is negative, so we keep track of that here.
            self.low = self.low+self.n_bins*self.step
            self.step *= -1
            self.reverse = True
        else:
            self.reverse = False

    def __call__(self):
        if self.field is not None:
            bin_type = SingleBin
        else:
            bin_type = SingleBinFieldless
        if self.use_log:
            return_list = [bin_type(field=self.field, low=numpy.exp(self.low+i*self.step),
                                     high=numpy.exp(self.low+(i+1)*self.step),
                                     short_name=str(i)) for i in range(self.n_bins)]
        else:
            return_list = [bin_type(field=self.field, low=self.low+i*self.step,
                                     high=self.low+(i+1)*self.step, short_name=str(i))
                                     for i in range(self.n_bins)]
        if self.reverse:
            return_list.reverse()
        return return_list


class SingleBinFieldless(object):
    """
    A class that contains the information for one particular bin generated from one of
    the :class:`Bin*` classes. The attributes can be accessed directly for :class:`DataHandler`\s
    that read in the data selectively. The class can also be called with a data array to bin it to
    the correct data range: ``SingleBin(array)`` will return only the data within the bounds of the
    particular instance of the class.  The endpoints are assumed to be ``[low,high)``, that is,
    ``low <= data < high``\.

    :param field:      The index of the field containing the data to be binned (must be a string).
    :param low:        The lower edge of the bin (inclusive).
    :param high:       The upper edge of the bin (exclusive).
    :param short_name: A string denoting this bin in filenames.
    :param long_name:  A string denoting this bin in program text outputs/plots
                       [default: ``"low-high"``].
    """

    def __init__(self, low, high, short_name, long_name=None):
        if high <= low:
            raise ValueError("High ("+str(high)+") must be greater than low ("+str(low)+")")
        if not isinstance(short_name, str) or (long_name and not isinstance(long_name, str)):
            raise TypeError("Short_name and long_name must be strings")
        self.low = low
        self.high = high
        self.short_name = short_name
        if long_name:
            self.long_name = long_name
        else:
            self.long_name = str(low)+'-'+str(high)

    def __call__(self, data):
        """
        Given data, returns only the data with ``data[self.field]`` within the bounds
        ``[self.low, self.high)``.

        :param data:   A NumPy array of data that can be indexed by ``self.field``.
        :returns:      A NumPy array corresponding to the input data, restricted to the bin
                       described by this object.
        """
        return data[numpy.logical_and(data >= self.low, data < self.high)]


class SingleBin(SingleBinFieldless):
    """
    A class that contains the information for one particular bin generated from one of the Bin*
    classes. The attributes can be accessed directly for DataHandlers that read in the data
    selectively. The class can also be called with a data array to bin it to the correct data
    range: SingleBin(array) will return only the data within the bounds of the particular instance
    of the class.  The endpoints are assumed to be [low,high), that is, low <= data < high, with
    defined relational operators.
    
    This version is for data arrays with defined fields such as numpy.recarrays; see 
    :class:`SingleBinFieldless` for the version that operates on raw arrays.

    @param field      The index of the field containing the data to be binned (must be a string).
    @param low        The lower edge of the bin (inclusive).
    @param high       The upper edge of the bin (exclusive).
    @param short_name A string denoting this bin in filenames.
    @param long_name  A string denoting this bin in program text outputs/plots.
                      [default: "low-high"]
    """
    def __init__(self, field, low, high, short_name, long_name=None):
        if not isinstance(field, str):
            raise TypeError('Field description must be a string. Passed value: '+str(field)+
                              ' of type'+str(type(field)))
        self.field = field
        super(SingleBin, self).__init__(low, high, short_name, long_name)

    def __call__(self, data):
        """
        Given data, returns only the data with data[self.field] within the bounds
        [self.low, self.high).

        @param data   A NumPy array of data that can be indexed by self.field.
        @returns      A NumPy array corresponding to the input data, restricted to the bin
                      described by this object.
        """
        return super(SingleBin, self).__call__(data[self.field])



class BinFunction:
    """
    An object which returns bin definitions (a list of :class:`SingleFunctionBin`\s) following the
    definitions given in initialization.  Note that unlike other :class:`SingleBin`\s, the
    :class:`SingleFunctionBin`\s created by :class:`BinFunction` do not have public ``field``,
    ``low``, or ``high`` attributes, since the function is assumed to be too complex for such
    parameterization.

    :param function:       The function to be applied to the data (an entire array, not just a
                           field).  The function should return an array of ``int`` s corresponding
                           to the bin numbers of the data rows (unless ``returns_bools`` is set, see
                           below). The function should take a data array as its only argument,
                           unless ``return_bools`` is set to True, in which case it should take a
                           bin number as its second argument.
    :param n_bins:         The maximum number of bins returned by the input function.  If None,
                           the function will be checked for an ``n_bins`` attribute; if it does not
                           exist an error will be raised.
    :param returns_bools:  True if the function will return an array of bools corresponding to a
                           mask to the bin number in question; False otherwise.
                           [default: False]
    :returns:              A list of :class:`SingleFunctionBin` objects determined by the input
                           criteria.
    """

    def __init__(self, function, n_bins=None, returns_bools=False):
        self.function = function

        if n_bins is None:
            try:
                self.n_bins = function.n_bins
            except:
                raise TypeError("Argument n_bins must be passed directly or as the attribute "+
                                "function.n_bins!")
        else:
            self.n_bins = n_bins
        self.returns_bools = returns_bools

    def __call__(self):
        return [SingleFunctionBin(self.function, i, self.returns_bools) for i in range(self.n_bins)]


class SingleFunctionBin(SingleBin):
    """
    A class that contains the information for one particular bin generated from a function. The
    class can be called with a data array to return only the data within the bounds of the
    particular instance of the class.  Unlike :class:`SingleBin`\s, there are no public ``field``,
    ``low``, or ``high`` attributes, as these are assumed to be insufficient to describe the
    behavior of the binning scheme.

    :param function:       The function that returns the bin information.
    :param n:              Which bin this :class:`SingleFunctionBin` considers.
    :param returns_bools:  True if the function returns bools, False if it returns bin
                           numbers [default: False].
    :param short_name:     A string denoting this bin in filenames [default: ``str(n)``].
    :param long_name:      A string denoting this bin in program outputs/plots
                           [default: ``short_name``].
    """

    def __init__(self, function, n, returns_bools=False, short_name=None, long_name=None):
        if ((short_name and not isinstance(short_name, str)) or
            (long_name and not isinstance(long_name, str))):
            raise TypeError("short_name and long_name must be strings")
        if short_name is not None:
            self.short_name = short_name
        else:
            self.short_name = str(n)
        if long_name is not None:
            self.long_name = long_name
        else:
            self.long_name = self.short_name
        self.function = function
        self.n = n
        if returns_bools:
            self.__call__ = self._call_bool
        else:
            self.__call__ = self._call_int

    def _call_int(self, data):
        """
        Given ``data``, returns only the data with ``self.bin_function==self.n``, as defined when
        the class was created.

        :param data:   Data which can be interpreted by ``self.function``.
        :returns:      A NumPy array corresponding to the input data, restricted to the bin
                       described by this object.
        """
        return data[self.function(data) == self.n]

    def _call_bool(self, data):
        """
        Given ``data``, returns only the data where ``self.bin_function(data, self.n)==True``, as
        defined when the class was created.

        :param data:   Data which can be interpreted by ``self.function``.
        :returns:      A NumPy array corresponding to the input data, restricted to the bin
                       described by this object.
        """
        return data[self.function(data, self.n)]


def ExpandBinList(bin_list):
    """
    If the user has indicated more than one :class:`Bin*` object, we assume that they want to do the
    intersection of each of those bins for the data set (for example, if they pass a magnitude bin
    scheme and a color bin scheme, we want to do magnitude bin 0 with color bin 0, then magnitude
    bin 0 with color bin 1, etc).  This function takes the :class:`Bin*` objects, generates their
    lists of :class:`SingleBin`\s, and then nests them in a way that makes sure the final list of
    lists of :class:`SingleBin`\s contains every possible set of (:class:`SingleBin`
    from :class:`Bin*` object 0, :class:`SingleBin` from :class:`Bin*` object 1, ...).  In
    particular, we make sure that the first :class:`SingleBin` corresponds to the
    first :class:`Bin*` object, the second to the second, etc, and that the
    first :class:`SingleBin` changes most slowly, then the second, etc.

    This function also works to expand a single :class:`Bin*` object, so it's fine for different
    modules to call this function with whatever they receive as a binning specification without
    typechecking first.

    Or in short, take a list of :class:`Bin*` objects, and expand them to return a list
    of :class:`SingleBin`\s to step through.  E.g., if the user passes a list:

        >>> bin_list = [BinList0, BinStep1]

    where ``BinList0.n_bins = 2`` and ``BinStep1.n_bins = 3``, then calling this function will
    return

        >>> [[SingleBinObject_0_0, SingleBinObject_1_0],
             [SingleBinObject_0_0, SingleBinObject_1_1],
             [SingleBinObject_0_0, SingleBinObject_1_2],
             [SingleBinObject_0_1, SingleBinObject_1_0],
             [SingleBinObject_0_1, SingleBinObject_1_1],
             [SingleBinObject_0_1, SingleBinObject_1_2]].

    :param bin_list:  A list of ``Bin``-type objects, such as the ones in ``binning.py``, which when
                      called return a list of items which behave like :class:`SingleBin`\s.  (No
                      checks are made in this function that these are valid :class:`SingleBin`\s.)
    :returns:         A list of lists of :class:`SingleBin`\s (or other items returned by calling
                      the input items in ``bin_list``), properly nested.
    """
    if not bin_list:
        return []
    if not isinstance(bin_list, (list, tuple)):
        bin_list = [bin_list]
    # A copy of bin_list that we can alter without altering the parent list, but which doesn't
    # duplicate the objects in bin_list. (I think.)
    bl = bin_list[:]
    data_bins = [[]]
    while bl:
        this_bin = bl.pop()
        data_bins = [[bin]+d for bin in this_bin() for d in data_bins]
    return data_bins

