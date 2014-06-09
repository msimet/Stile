"""@file stile_utils.py
Various utilities for the Stile pipeline.  Includes input parsing and some numerical helper 
functions.
"""

import numpy

def Parser():
    """
    Returns an argparse Parser object with input args used by Stile and corr2.
    """
    import corr2_utils
    import argparse
    p = argparse.Parser(parent=corr2_utils.Parser())
    #TODO: add, obviously, EVERYTHING ELSE
    return p

def ExpandBinList(bin_list):
    """
    Take a list of Bin* objects, and expand them to return a list of SimpleBins to step through.  
    E.g., if the user passes a list :
        >>> bin_list = [ListBin0, ListBin1]
    where ListBin1.n_bins = 2 and ListBin2.n_bins = 3, then calling this function will return
        >>> [[SimpleBinObject_0_0, SimpleBinObject_1_0],
             [SimpleBinObject_0_0, SimpleBinObject_1_1],
             [SimpleBinObject_0_0, SimpleBinObject_1_2],
             [SimpleBinObject_0_1, SimpleBinObject_1_0],
             [SimpleBinObject_0_1, SimpleBinObject_1_1],
             [SimpleBinObject_0_1, SimpleBinObject_1_2]].
    The first item in the list changes most slowly, then the second item, etc.
             
    @param bin_list  A list of Bin-type objects, such as the ones in binning.py, which when called
                     return a list of items which behave like SimpleBins.  (No checks are made in
                     this function that these are valid items.)
    @returns         A list of lists of SimpleBins (or other items returned by calling the input
                     items in bin_list), properly nested.
    """
    if not bin_list:
        return []
    # A copy of bin_list that we can alter without altering the parent list, but which doesn't 
    # duplicate the objects in bin_list. (I think.)
    bl = bin_list[:]
    data_bins = [[]]
    while bl:
        this_bin = bl.pop()
        data_bins = [[bin]+d for bin in this_bin() for d in data_bins]
    return data_bins


def GetVectorType(x):
    """
    Figure out the least-precise type that can represent an entire vector of data.

    @param x An iterable of data (a single scalar value will likely result in errors)
    @returns a string usable in a NumPy dtype declaration for x
    """
    if not hasattr(x,'astype'):
        x = numpy.array(x)
    #TODO: better data types here
    for var_type, type_name in [(int,'l'),(float,'d')]:
        try:
            x.astype(var_type)
            return type_name
        except:
            pass
    return 'S'+str(max([len(xx) for xx in x]))

def FormatArray(d,fields=None,only_floats=False):
    """
    Turn a regular NumPy array of arbitrary types into a formatted array, with optional field name 
    description.

    @param d      A NumPy array (or other iterable which satisfies hasattr(d,'shape')).
    @param fields A dictionary whose keys are the names of the fields you'd like for the output 
                  array, and whose values are field numbers (starting with 0) whose names those keys 
                  should replace. (default: None)
    @param only_floats All fields are floats, don't check for data type (default: False)
    @returns      A formatted numpy array with the same shape as d except that the innermost 
                  dimension has turned into a record field, optionally with field names 
                  appropriately replaced.
    """
    if hasattr(d,'dtype') and hasattr(d.dtype,'names'):
        pass
    else:
        d_shape = d.shape
        new_d = d.reshape(-1,d_shape[-1])
        new_d = numpy.array(d)
        if only_floats:
            types = ','.join(['d']*len(new_d[0]))
        else:
            types = ','.join([get_vector_type(new_d[:,i]) for i in range(len(new_d[0]))])
        d = numpy.array([tuple(nd) for nd in new_d],dtype=types)
        if len(d_shape)>1:
            d = d.reshape(d_shape[:-1])
    if fields:
        if isinstance(fields,dict):
            names = list(d.dtype.names)
            for key in fields:
                names[fields[key]]=key
            d.dtype.names = names
        elif len(fields)==len(d.dtype.names):
            d.dtype.names = fields
        else:
            raise RuntimeError('Cannot use given fields: '+str(fields))
    return d
    
        
def MakeFiles(dh, data, data2=None, random=None, random2=None):
    """
    Pick which files need to be written to a file for corr2, and which can be passed simply as a
    filename. This takes care of making temporary files, checking that the field schema is
    consistent in any existing files and rewrites the ones that do not match the dominant field 
    schema if necessary, and figuring out the corr2 column arguments (eg ra_col).
    
    @param dh      A DataHandler instance
    @param data    The data that will be passed to the Stile tests. Can be a 
                   (file_name,field_schema) tuple, a NumPy array, or a list of one or the 
                   other of those options (but NOT both!)
    @param data2   The second set of data that will be passed for cross-correlations
    @param random  The random data set corresponding to data
    @param random2 The random data set corresponding to data2
    @returns       A 7-item tuple with the following items: 
                     new_data, new_data2, new_random, new_random2, - with arrays replaced by files
                     corr2_kwargs, - dictionary of kwargs to be passed to corr2
                     handles,      - open-file handles to be closed later (AFTER file use, as
                                     some OSes will delete these temporary files if they're closed!)
                     deletes       - names of files to be deleted after use.
    """
    
    #TODO: do this in a smarter way that only cares about the fields we'll be using
    #TODO: check FITS/ASCII
    #TODO: proper corr2 kwargs for FITS columns
    #TODO: think about how this works if we rewrite a data set we want to come back to
    import os
    import corr2_utils
    import file_io
    import tempfile
    already_written = []
    aw_files = []
    to_write = []
    # First check for already-written files, and grab their field schema
    for data_list in [data, data2, random, random2]:
        if data_list is None or len(data_list)==0:
            continue
        elif isinstance(data_list,tuple):
            if os.path.isfile(data_list[0]):
                already_written.append(data_list[1])
                aw_files.append(data_list[0])
            else:
                raise RuntimeError("Data tuple appears to point to an existing file %s, but that "+
                                   "file is not found according to os.path.isfile()"%data_list[0])
        elif hasattr(data_list,'__getitem__'):
            # We don't want to cycle through a whole data array, so first check whether the first 
            # item points to a file..
            if isinstance(data_list[0],tuple):
                if os.path.isfile(data_list[0][0]):
                    for dl in data_list:
                        if os.path.isfile(dl[0]):
                            already_written.append(dl[1])
                            aw_files.append(dl[0])
                        else:
                            raise RuntimeError("Data tuple appears to point to an existing file "+
                                               "%s, but that file is not found according to "+
                                               "os.path.isfile()"%dl[0])
                else:
                    raise RuntimeError("Data tuple appears to point to an existing file %s, but "+
                                       "that file is not found according to "+
                                       "os.path.isfile()"%data_list[0][0])
    # Check field schema for consistency
    if already_written:
        while True:
            all_same = True
            for i in range(len(already_written)-1):
                for j in range(i,len(already_written)):
                    if not already_written[i]==already_written[j]:
                        all_same=False
            if all_same:
                break
            aw_keys = []
            for aw in already_written:
                aw_keys += aw.keys()
            aw_keys = set(aw_keys)
            all_same = True
            for key in aw_keys:
                n = [aw.get(key,None) for aw in already_written]
                n = set([nn for nn in n if nn is not None])
                if len(n)>1:
                    all_same = False
                    break
            if all_same:
                break
            else:
                # If they're inconsistent, remove the smallest file and repeat this loop
                sizes = [os.path.getsize(awf) for awf in aw_files]
                remove = sizes.index(min(sizes))
                to_write.append((aw_files[remove],already_written[remove]))
                del already_written[remove]
                del aw_files[remove]
        aw_set = already_written[0]
        for aw in already_written[1:]:
            aw_set.update(aw)
        fields = [0 for i in range(max([aw_set[key] for key in aw_set.keys()])+1)]
        for key in aw_set:
            fields[aw_set[key]] = key
    else:
        # need to fix this more completely/robustly, but to get things working for now...
        if hasattr(data,'dtype') and hasattr(data.dtype,'names'):
            fields = data.dtype.names
        elif hasattr(data[0],'dtype') and hasattr(data[0].dtype,'names'):
            fields = data[0].dtype.names
        else:
            fields = ['id','ra','dec','z','g1','g2']

    handles = []
    deletes = []
    new_data = []
    new_data2 = []
    new_random = []
    new_random2 = []
    # Now loop through again and write to a file any data arrays we need to.
    # NOTE: currently not checking again that file exists
    for data_list, new_data_list in [(data,new_data), (data2,new_data2), 
                                      (random,new_random), (random2, new_random2)]:
        if data_list is None or len(data_list)==0:
            continue
        elif isinstance(data_list,tuple):
            if os.path.isfile(data_list[0]):
                if data_list[0] in to_write:
                    data = dh.get_data(data_list[0],force=True)
                    handle, data_file = tempfile.mkstemp(dh.temp_dir)
                    handles.append(handle)
                    deletes.append(data_file)
                    file_io.write_ascii_table(data_file,data,fields=fields)
                    new_data_list.append(data_file)
                else:
                    new_data_list.append(data_list[0])
        elif hasattr(data_list,'__getitem__'):
            if isinstance(data_list[0],tuple):
                for dl in data_list:
                    if dl[0] in to_write:
                        data = dh.get_data(dl[0],force=True)
                        handle, data_file = tempfile.mkstemp(dh.temp_dir)
                        handles.append(handle)
                        deletes.append(data_file)
                        file_io.write_ascii_table(data_file,data,fields=fields)
                        new_data_list.append(data_file)
                    else:
                        new_data_list.append(dl[0])
            else:
                if hasattr(data_list,'dtype') and hasattr(data_list.dtype,'names'): 
                    handle, data_file = tempfile.mkstemp(dh.temp_dir)
                    handles.append(handle)
                    deletes.append(data_file)
                    file_io.WriteAsciiTable(data_file,data_list,fields=fields)
                    new_data_list.append(data_file)
                else: 
                    for dl in data_list:
                        if not hasattr(dl,'dtype') or not hasattr(dl.dtype,'names'):
                            raise RuntimeError("Cannot parse data: should be a tuple, "+
                                               "numpy array, or an unmixed list of one or the "+
                                               "other.  Given:"+str(data_list))
                        handle, data_file = tempfile.mkstemp(dh.temp_dir)
                        handles.append(handle)
                        deletes.append(data_file)
                        file_io.write_ascii_table(data_file,dl,fields=fields)
                        new_data_list.append(data_file)
    
    # Lists of files need to be written to a separate file to be read in; do that.
    file_args = []
    for file_list in [new_data, new_data2, new_random, new_random2]:
        if len(file_list)>1:
            handle, data_file = tempfile.mkstemp(dh.temp_dir)
            handles.append(handle)
            deletes.append(data_file)
            with open(data_file,'w') as d:
                for fl in file_list:
                    d.write(fl+'\n')
            file_args.append(('list',data_file))
        elif len(file_list)==1:
            file_args.append(('name',file_list[0]))
        else:
            file_args.append(None)
    new_data, new_data2, new_random, new_random2 = file_args
    
    corr2_kwargs = corr2_utils.MakeCorr2Cols(fields)
    return new_data, new_data2, new_random, new_random2, corr2_kwargs, handles, deletes

class Stats:
    """A Stats object can carry around and output the statistics of some array.

    Currently it can carry around two types of statistics:

    (1) Basic array statistics: min, max, median, mean, standard deviation, variance.

    (2) Percentiles: the value at a given percentile level.

    The StatsSysTest class in `sys_tests.py` can be used to create and populate values for one of
    these objects.  Presently it is necessary to update both the definition of the Stats class and
    the StatsSysTest class if you want to add / remove tests.  In future we might want to make
    changes necessary in only one place (e.g., by storing in the Stats object the name of the
    function to be used for populating that field, so that StatsSysTest objects just iterate over
    members of the Stats object and call the relevant function).

    Also note that if we want things like skewness and kurtosis, we either need to calculate them
    directly or use scipy, since numpy does not include those.
    """
    self.min = None
    self.max = None
    self.median = None
    self.mean = None
    self.stddev = None
    self.variance = None
    self.percentiles = None
    self.values = None

    def prettyPrint(self):
        """This routine will print the contents of the Stats object in a nice format."""
        # First check whether this has any values that are not None.  If not, then just return or
        # throw exception or something.
        # Loop over simple statistics and print them.
        # Loop over combinations of percentiles and values, and print them.
