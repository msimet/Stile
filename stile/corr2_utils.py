"""@file corr2_utils.py
Contains elements of Stile needed to interface with Mike Jarvis's corr2 program.
"""
import copy
import numpy
import os
import file_io
import tempfile

# A dictionary containing all corr2 command line arguments.  (At the moment we only support v 2.5+,
# so only one dict is here; later versions of Stile may need to implement if statements here for
# the corr2 versions.)  The arguments themselves are mapped onto dicts with the following keys: 
#    'type': a tuple of the allowed input types.  
#    'val' : if the value must be one of a limited set of options, they are given here; else None.
#    'status': whether or not this is a corr2 argument that Stile will pass through without 
#              altering.  The options are 'disallowed_computation' (Stile makes these choices),
#              'disallowed_file' (the DataHandler makes these choices), and 'allowed' (Stile 
#              should silently pass it through to corr2).
corr2_kwargs = {
    'file_name': 
        {'type': (str,),
         'val': None,
         'status': 'disallowed_file'},
    'do_auto_corr': 
        {'type': (bool,),
         'val': None,
         'status': 'disallowed_computation'},
    'do_cross_corr':
        {'type': (bool,),
         'val': None,
         'status': 'disallowed_computation'},
    'file_name2': 
        {'type': (str,),
         'val': None,
         'status': 'disallowed_file'},
    'rand_file_name': 
        {'type': (str,),
         'val': None,
         'status': 'disallowed_file'},
    'rand_file_name2': 
        {'type': (str,),
         'val': None,
         'status': 'disallowed_file'},
    'file_list': 
        {'type': (str,),
         'val': None,
         'status': 'disallowed_file'},
    'file_list2': 
        {'type': (str,),
         'val': None,
         'status': 'disallowed_file'},
    'rand_file_list': 
        {'type': (str,),
         'val': None,
         'status': 'disallowed_file'},
    'rand_file_list2': 
        {'type': (str,),
         'val': None,
         'status': 'disallowed_file'},
    'file_type': 
        {'type': (str,),
         'val': ("ASCII","FITS"),
         'status': 'allowed'},
    'delimiter': 
        {'type': (str,),
         'val': None,
         'status': 'allowed'},
    'comment_marker': 
        {'type': (str,),
         'val': None,
         'status': 'allowed'},
    'first_row':
        {'type': (int,),
         'val': None,
         'status': 'allowed'},
    'last_row':
        {'type': (int,),
         'val': None,
         'status': 'allowed'},
    'x_col':
        {'type': (int,str),
         'val': None,
         'status': 'allowed'},
    'y_col':
        {'type': (int,str),
         'val': None,
         'status': 'allowed'},
    'ra_col':
        {'type': (int,str),
         'val': None,
         'status': 'allowed'},
    'dec_col':
        {'type': (int,str),
         'val': None,
         'status': 'allowed'},
    'x_units':
        {'type': (str,),
         'val': ['radians', 'hours', 'degrees', 'arcmin', 'arcsec'],
         'status': 'allowed'},
    'y_units':
        {'type': (str,),
         'val': ['radians', 'hours', 'degrees', 'arcmin', 'arcsec'],
         'status': 'allowed'},
    'ra_units':
        {'type': (str,),
         'val': ['radians', 'hours', 'degrees', 'arcmin', 'arcsec'],
         'status': 'allowed'},
    'dec_units':
        {'type': (str,),
         'val': ['radians', 'hours', 'degrees', 'arcmin', 'arcsec'],
         'status': 'allowed'},
    'g1_col':
        {'type': (int,str),
         'val': None,
         'status': 'allowed'},
    'g2_col':
        {'type': (int,str),
         'val': None,
         'status': 'allowed'},
    'k_col':
        {'type': (int,str),
         'val': None,
         'status': 'allowed'},
    'w_col':
        {'type': (int,str),
         'val': None,
         'status': 'allowed'},
    'flip_g1':
        {'type': (bool,),
         'val': None,
         'status': 'allowed'},
    'flip_g2':
        {'type': (bool,),
         'val': None,
         'status': 'allowed'},
    'pairwise':
        {'type': (bool,),
         'val': None,
         'status': 'disallowed_computation'},
    'project':
        {'type': (bool,),
         'val': None,
         'status': 'allowed'},
    'project_ra':
        {'type': (float,),
         'val': None,
         'status': 'allowed'},
    'project_dec':
        {'type': (float,),
         'val': None,
         'status': 'allowed'},
    'min_sep':
        {'type': (float,),
         'val': None,
         'status': 'allowed'},
    'max_sep':
        {'type': (float,),
         'val': None,
         'status': 'allowed'},
    'nbins':
        {'type': (float,),
         'val': None,
         'status': 'allowed'},
    'bin_size':
        {'type': (float,),
         'val': None,
         'status': 'allowed'},
    'sep_units':
        {'type': (str,),
         'val': ['radians', 'hours', 'degrees', 'arcmin', 'arcsec'],
         'status': 'allowed'},
    'bin_slop':
        {'type': (float,),
         'val': None,
         'status': 'allowed'},
    'smooth_scale':
        {'type': (float,),
         'val': None,
         'status': 'disallowed_computation'},
    'n2_file_name':
        {'type': (str,),
         'val': None,
         'status': 'disallowed_file'},
    'n2_statistic':
        {'type': (str,),
         'val': ['compensated','simple'],
         'status': 'disallowed_computation'},
    'ng_file_name':
        {'type': (str,),
         'val': None,
         'status': 'disallowed_file'},
    'ng_statistic':
        {'type': (str,),
         'val': ['compensated','simple'],
         'status': 'disallowed_computation'},
    'g2_file_name':
        {'type': (str,),
         'val': None,
         'status': 'disallowed_file'},
    'nk_file_name':
        {'type': (str,),
         'val': None,
         'status': 'disallowed_file'},
    'nk_statistic':
        {'type': (str,),
         'val': ['compensated','simple'],
         'status': 'disallowed_computation'},
    'k2_file_name':
        {'type': (str,),
         'val': None,
         'status': 'disallowed_file'},
    'kg_file_name':
        {'type': (str,),
         'val': None,
         'status': 'disallowed_file'},
    'precision': 
        {'type': (int,),
         'val': None,
         'status': 'allowed'},
    'm2_file_name':
        {'type': (str,),
         'val': None,
         'status': 'disallowed_file'},
    'm2_uform':
        {'type': (str,),
         'val': ['Schneider','Crittenden'],
         'status': 'allowed'},
    'nm_file_name':
        {'type': (str,),
         'val': None,
         'status': 'disallowed_file'},
    'norm_file_name':
        {'type': (str,),
         'val': None,
         'status': 'disallowed_file'},
    'verbose': 
        {'type': (int,),
         'val': None,
         'status': 'allowed'},
    'num_threads':
        {'type': (int,),
         'val': None,
         'status': 'allowed'},
    'split_method':
        {'type': (str,),
         'val': ["mean","median","middle"],
         'status': 'allowed'}}

def Parser():
    import argparse
    p = argparse.Parser()
    p.add_argument('--file_type',
                   help="File type (ASCII or FITS) -- only allowed by certain DataHandlers",
                   dest='file_type')
    p.add_argument('--delimiter',
                   help="ASCII file column delimiter -- only allowed by certain DataHandlers",
                   dest='comment_marker')
    p.add_argument('--comment_marker',
                   help="ASCII file comment-line marker -- only allowed by certain DataHandlers",
                   dest='comment_marker')
    p.add_argument('--first_row',
                   help="First row of the file(s) to be considered -- only allowed by certain "+
                        "DataHandlers",
                   dest='first_row')
    p.add_argument('--last_row',
                   help="Last row of the file(s) to be considered -- only allowed by certain "+
                        "DataHandlers",
                   dest='last_row')
    p.add_argument('--comment_marker',
                   help="ASCII file comment-line marker -- only allowed by certain DataHandlers",
                   dest='comment_marker')
    p.add_argument('--x_col',
                   help="Number of the x-position column -- only allowed by certain DataHandlers",
                   dest='x_col')
    p.add_argument('--y_col',
                   help="Number of the y-position column -- only allowed by certain DataHandlers",
                   dest='y_col')
    p.add_argument('--ra_col',
                   help="Number of the ra column -- only allowed by certain DataHandlers",
                   dest='ra_col')
    p.add_argument('--dec_col',
                   help="Number of the dec column -- only allowed by certain DataHandlers",
                   dest='dec_col')
    p.add_argument('--x_units',
                   help="X-column units (radians, hours, degrees, arcmin, arcsec)  -- only allowed "+
                        "by certain DataHandlers",
                   dest='x_units')
    p.add_argument('--y_units',
                   help="Y-column units (radians, hours, degrees, arcmin, arcsec) -- only allowed "+
                        "by certain DataHandlers",
                   dest='y_units')
    p.add_argument('--ra_units',
                   help="RA-column units (radians, hours, degrees, arcmin, arcsec) -- only "+
                        "allowed by certain DataHandlers",
                   dest='ra_units')
    p.add_argument('--dec_units',
                   help="dec-column units (radians, hours, degrees, arcmin, arcsec) -- only "+
                        "allowed by certain DataHandlers",
                   dest='dec_units')
    p.add_argument('--g1_col',
                   help="Number of the g1 column -- only allowed by certain DataHandlers",
                   dest='g1_col')
    p.add_argument('--g2_col',
                   help="Number of the g2 column -- only allowed by certain DataHandlers",
                   dest='g2_col')
    p.add_argument('--k_col',
                   help="Number of the kappa [scalar] column -- only allowed by certain "+
                        "DataHandlers",
                   dest='k_col')
    p.add_argument('--w_col',
                   help="Number of the weight column -- only allowed by certain DataHandlers",
                   dest='w_col')
    p.add_argument('--flip_g1',
                   help="Flip the sign of g1 (default: False)",
                   dest='flip_g1',default=False)
    p.add_argument('--flip_g2',
                   help="Flip the sign of g2 (default: False)",
                   dest='flip_g2',default=False)
    p.add_argument('--project',
                   help="Corr2 argument: use a tangent-plane projection instead of curved-sky "+
                        "(this is a negligible performance improvement, and not recommended)",
                   dest='project')
    p.add_argument('--project_ra',
                   help="Corr2 argument: the RA of the tangent point for projection, used in "+
                        "conjunction with --project, and not recommended",
                   dest='project_ra')
    p.add_argument('--project_dec',
                   help="Corr2 argument: the dec of the tangent point for projection, used in "+
                        "conjunction with --project, and not recommended",
                   dest='project_dec')
    p.add_argument('--min_sep',
                   help="Minimum separation for the corr2 correlation functions",
                   dest='min_sep')
    p.add_argument('--max_sep',
                   help="Maximum separation for the corr2 correlation functions",
                   dest='max_sep')
    p.add_argument('--nbins',
                   help="Number of bins for the corr2 correlation functions",
                   dest='nbins')
    p.add_argument('--bin_size',
                   help="Bin width for the corr2 correlation functions",
                   dest='bin_size')
    p.add_argument('--sep_units',
                   help="Units for the max_sep/min_sep/bin_size arguments for the corr2 "+
                        "correlation functions",
                   dest='max_sep')
    p.add_argument('--bin_slop',
                   help="A parameter relating to accuracy of the corr2 bins--changing is not "+
                        "recommended",
                   dest='bin_slop')
    p.add_argument('--precision',
                   help="Number of digits after (scientific notation) decimal point in corr2 "+
                        "(default: 3)",
                   dest='precision')
    p.add_argument('--m2_uform',
                   help="Set to 'Schneider' to use the Schneider rather than the Crittenden forms "+
                        "of the aperture mass statistic in corr2 (see corr2 Read.me for more info)",
                   dest='m2_uform')
    p.add_argument('-v','--verbose',
                   help="Level of verbosity",
                   dest='verbose')
    p.add_argument('--num_threads',
                   help='Number of OpenMP threads (corr2) or multprocessing.Pool processors '+
                        '(Stile) to use; default is to automatically determine',
                   dest='num_threads')
    p.add_argument('--split_method',
                   help="One of 'mean', 'median', or 'middle', directing corr2 how to split the "
                        "tree into child nodes. (default: 'mean')",
                   dest='split_method')
    return p                   


def CheckArguments(input_dict, check_status=True):
    """
    A function that checks the (key,value) pairs of the dict passed to it against the corr2 
    arguments dict.  If the key is not understood, or if check_status is True and the key is not 
    allowed, an error is raised.  If the key is allowed, the type and/or values are checked 
    against the corr2 requirements.
    
    @param input_dict   A dict which will be used to write a corr2 configuration file
    @param check_status A flag indicating whether to check the status of the keys in the dict.  This
                        should be done when eg reading in arguments from the command line; later 
                        checks for type safety, after Stile has added its own kwargs, shouldn't
                        do it.  (default: True)
    @returns            The input dict, unchanged.            
    """
    #TODO: add check_required to make sure it has all necessary keys
    for key in input_dict:
        if key not in corr2_kwargs:
            raise ValueError('Argument %s not a recognized corr2 argument.  Please check syntax '                              'and try again.'%key)
        else:
            c2k = corr2_kwargs[key]
            if check_status:
                if c2k['status']=='disallowed_file':
                    raise ValueError('Argument %s for corr2 is forbidden by Stile, which may need '
                                     'to write multiple output files of this type.  Please remove '
                                     'this argument from your syntax, and check the documentation '
                                     'for where the relevant output files will be located.'%key)
                elif c2k['status']=='disallowed_computation':
                    raise ValueError('Argument %s for corr2 is forbidden by Stile, which controls '
                                     'the necessary correlation functions.  Depending on your '
                                     'needs, please either remove this argument from your syntax '
                                     'or consider running corr2 as a standalone program.'%key)
            if type(input_dict[key]) not in c2k['type']:
                # The arguments may be passed as strings. Since a string may not be the desired
                # argument type, try casting the value into the correct type or types and see if
                # it works or raises an error; if at least one works, pass, else raise an error.
                type_ok = False
                for arg_type in c2k['type']:
                    try:
                        arg_type(input_dict[key])
                        type_ok=True
                    except:
                        pass
                if not type_ok:
                    raise ValueError(("Argument %s is a corr2 argument, but the type of the given "
                                     "value %s does not match corr2's requirements.  Please "
                                     "check syntax and try again.")%(key,input_dict[key]))
            if c2k['val']:
                if input_dict[key] not in c2k['val']:
                    raise ValueError('Corr2 argument %s only accepts the values [%s].  Please '
                                     'check syntax and try again.'%(key,', '.join(c2k['val'])))
    return input_dict
    
def WriteCorr2ConfigurationFile(config_file_name,corr2_dict,**kwargs):
    """
    Write the given corr2 kwargs to a corr2 configuration file if they are in the arguments dict 
    above. 
    
    @param config_file_name May be a file name or any object with a .write(...) attribute.
    @param corr2_dict       A dict containing corr2 kwargs.
    @param kwargs           Any extra keys to be added to the given corr2_dict.  If they conflict,
                            the keys given in the kwargs will silently supercede the values in the
                            corr2_dict.
    """
    if hasattr(config_file_name,'write'):
        f=config_file_name
        close_file=False
    else:
        f=open(config_file_name,'w')
        close_file=True
    if kwargs:
        corr2_dict.update(kwargs)
    CheckArguments(corr2_dict,check_status=False)
    for key in corr2_dict:
        f.write(key+' = ' + str(corr2_dict[key])+'\n')
    if close_file:
        f.close()
        
def ReadCorr2ResultsFile(file_name):
    """
    Read in the given file_name of type file_type.  Cast it into a formatted numpy array with the
    appropriate fields and return it.
    
    @param file_name The location of an output file from corr2.
    @returns         A numpy array corresponding to the data in file_name.
    """    
    import stile_utils
    # Currently there is a bug in corr2 that puts some text output into results files, necessitating
    # the "comments='R'" line, plus the "skiprows" argument to skip the first (real) comment line.
    output = file_io.ReadASCIITable(file_name,comments='R',skiprows=1)
    
    if not len(output):
        raise RuntimeError('File %s (supposedly an output from corr2) is empty.'%file_name)
    # Now, the first line of the corr2 output file is of the form:
    # "# col1 . col2 . col3 [...]"
    # so we can get the proper field names by reading the first line of the file and processing it.
    with open(file_name) as f:
        fields = f.readline().split()
    fields = fields[1:]
    fields = [field for field in fields if field!='.']
    return stile_utils.FormatArray(output,fields=fields)

def AddCorr2Dict(input_dict):
    """
    Take an input_dict, harvest the kwargs you'll need for corr2, and create a new 'corr2_args'
    key in the input_dict containing these values (or update the existing 'corr2_args' key).  This
    is useful if you have a parameters dict that contains some things corr2 might want, but some
    other keys that shouldn't be written to the corr2 parameter file.
    
    @param input_dict A dict containing some (key,value) pairs that apply to corr2
    @returns          The input_dict with an added or updated key 'corr2_kwargs' whose value is a
                      dict containing the (key,value) pairs from input_dict that apply to corr2
    """    
    corr2_dict = {}
    new_dict = copy.deepcopy(input_dict)
    for key in corr2_kwargs:
        if key in input_dict:
            corr2_dict[key] = input_dict[key]
    if 'corr2_kwargs' in new_dict:
        new_dict['corr2_kwargs'].update(corr2_dict)
    else:
        new_dict['corr2_kwargs'] = corr2_dict
    return new_dict
    
def MakeCorr2Cols(cols,use_as_k=None):
    """
    Takes an input dict or list of columns and extracts the right variables for the column keys in a
    corr2 configuration file.  Note that we generally call these "fields" in Stile, but for 
    compatibility with corr2 config files they're called "cols" here.
    
    @param cols     A list of strings denoting the columns of a file (first column is first element
                    of list, etc), or a dict with the key-value pairs "string column name": column 
                    number
    @param use_as_k Which column to use as the "kappa" (scalar) column, if given (default: None).
                    Corr2 allows a correlation function between a scalar value such as the 
                    convergence and other quantities such as the shear; we might want to use another
                    parameter (such as star brightness) here, so setting use_as_k to that column
                    will tell corr2 to do a convergence-type correlation function with that column
                    as the "convergence" value.
    @returns        A dict containing the column key-value pairs for corr2
    """
    corr2_kwargs = {}
    col_args = ['x','y','ra','dec','g1','g2','k','w']
    if isinstance(cols,dict):
        for col in col_args:
            if col in cols:
                if isinstance(cols[col],int):
                    corr2_kwargs[col+'_col'] = cols[col]+1 # corr2 ordering starts at 1, Stile at 0
                elif isinstance(cols[col],str):
                    corr2_kwargs[col+'_col'] = cols[col]
        if use_as_k and use_as_k in cols:
            if isinstance(cols[use_as_k],int):
                corr2_kwargs['k_col'] = cols[use_as_k]+1
            elif isinstance(cols[col],str):
                corr2_kwargs['k_col'] = cols[use_as_k]
    elif hasattr(cols,'index'):
        for col in col_args:
            if col in cols:
                corr2_kwargs[col+'_col'] = cols.index(col)+1
        if use_as_k and use_as_k in cols:
            corr2_kwargs['k_col'] = cols.index(use_as_k)+1
    return corr2_kwargs
    

class OSFile:
    """
    A class that contains information about an array of data and writes it to a file on disk.  The
    class takes care of closing OS file handles and deleting the temporary file if necessary.  The
    string representation of this class is simply the name of the file it is keeping track of, so
    OSFiles may be used interchangeably with filenames in places where only the string 
    representation matters, such as writing corr2 parameter files.
    
    The data can be passed in two ways:
        - As a (data_handler `dh`, `data_id` pair, in which case dh.getData(data_id,force=True)
          is called
        - Directly as an array, in which case `is_array` should be set to True.
    In either case, "fields" may be set to control which fields of the data are printed to the
    temporary file.  "fields" should be either a list of fields in order, or a dict of 
    {'field_name': field_number/field_str} pairs, with "field_str" applying only if you have 
    pyfits/astropy installed to handle FITS files (to map field names onto Stile/Corr2 expected 
    field names). Further caveats about the use of the "fields" kwarg may be found in the 
    documentation for WriteTable.
    
    @param data      An array of data.
    @param fields    A description of the fields to be written out. See above or the documentation  
                     for WriteTable. (default: None)
    """
    
    def __init__(self,data,fields=None):
        # Do these first to protect against annoying errors during cleanup if init fails
        self.handle = -1
        self.file_name = ''
        if isinstance(data,OSFile) and (not fields or fields==data.fields):
            # Not sure how to do this better
            self.data = data.data
            self.fields = data.fields
            self.file_name = data.file_name
            self.handle = data.handle
        else:
            if isinstance(data,OSFile):
                self.data = data.data
            else:
                self.data = data
            self.fields = fields
            if not self.fields:
                try:
                    self.fields = data.dtype.names # So we can check against this later
                except:
                    pass
            if self.fields: 
                # This will be True if this is a formatted NumPy array.  (WriteFITSTable doesn't 
                # deal well with non-NumPy arrays, since it doesn't know what's a column and what's 
                # a row.)
                self.handle, self.file_name = tempfile.mkstemp(suffix=file_io.GetExtension())
                file_io.WriteTable(self.file_name,self.data,fields=self.fields)
            else: 
                self.handle, self.file_name = tempfile.mkstemp()
                file_io.WriteASCIITable(self.file_name,self.data)
    def __repr__(self):
        return self.file_name
    def __del__(self):
        try:
            os.close(self.handle)
        except OSError: # in case already closed 
            pass
        if os.path.isfile(self.file_name):
            os.remove(self.file_name)
    def __eq__(self,other):
        # Mostly necessary for testing purposes
        return (numpy.all(self.data==other.data) and self.fields==other.fields and 
                self.file_name==other.file_name and self.handle==other.handle)

def _merge_fields(has_fields,old_fields,new_fields):
    """ Get the intersection (not union!) of two field schemas. "has_fields" means the old_fields
    dict ever contained fields, even if the intersection is empty. """
    if not new_fields:
        return has_fields, old_fields
    if not has_fields:
        return True, _coerce_schema(new_fields)
    else:
        keys = old_fields.keys()
        for key in keys:
            if key not in new_fields:
                del old_fields[key]
        return True, old_fields
        
def _check_fields(has_fields,already_written_files,fields,data_list):
    """
    Run _merge_fields for all the various ways data could have been passed to MakeCorr2FileKwargs.
    """
    if isinstance(data_list,tuple):
        if data_list[0] in already_written_files:
            return has_fields, fields
        else:
            has_fields, fields = _merge_fields(has_fields,fields,data_list[1])
    elif isinstance(data_list,OSFile):
        if data_list.file_name in already_written_files:
            return has_fields, fields
        else:
            has_fields, fields = _merge_fields(has_fields,fields,data_list.fields)
    elif hasattr(data_list,'dtype') and data_list.dtype.names:
        has_fields, fields = _merge_fields(has_fields,fields,data_list.dtype.names)
    elif hasattr(data_list,'__iter__'):
        for dl in data_list:
            has_fields, fields = _check_fields(has_fields,already_written_files,fields,dl)
    elif data_list:
        raise ValueError("Cannot understand data type (should be a NumPy formatted array or a "+
                         "tuple (file_name, field_description): "+str(data_list))
    return has_fields, fields

def _coerce_schema(schema):
    """
    Turn the list-type field description into the dict-type field description, since it's easier
    to handle here.
    """
    if isinstance(schema,(list,tuple)):
        return dict([(schema[i],i) for i in range(len(schema))])
    elif isinstance(schema,dict):
        return schema
    else:
        raise ValueError("Schema must be a list or dict")
    
def MakeCorr2FileKwargs(data, data2=None, random=None, random2=None, use_as_k=None):
    """
    Pick which files need to be written to a file for corr2, and which can be passed simply as a
    filename. This takes care of making temporary files, checking that the field schema is
    consistent in any existing files and rewriting the ones that do not match the dominant field 
    schema if necessary, and figuring out the corr2 column arguments (eg ra_col).
    
    @param data     The data that will be passed to the Stile tests. Can be a 
                    (file_name,field_schema) tuple, a NumPy array, or a list of one or the 
                    other of those options.  The field_schema is the same kind of description used
                    in stile_utils.FormatArray and the table read functions in file_io.py: a 
                    dictionary whose keys are the names of the fields you'd like for the output 
                    array, and whose values are column numbers in the file (starting with 0) whose 
                    names those keys should replace (or, if it's a FITS file, the existing field 
                    names the keys should replace); alternately, a list with the same length as the 
                    rows of the file.  In the dict form, you don't need to specify every column, 
                    only the ones Stile will use.
    @param data2    The second set of data that will be passed for cross-correlations, with the same
                    format options as data.
    @param random   The random data set corresponding to data (ditto)
    @param random2  The random data set corresponding to data2 (ditto)
    @param use_as_k This is passed through to MakeCorr2Cols to designate a scalar field as the
                    "convergence" for a correlation function; see the documentation for 
                    MakeCorr2Cols for more information.
    @returns        A dict containing the file names and column descriptions for corr2.
    """
    #TODO: do this in a smarter way that only cares about the fields we'll be using
    already_written_schema = []
    already_written_files = []
    to_write = []

    # First check for already-written files, and grab their field schema
    for data_list in [data, data2, random, random2]:
        # len(None) fails, and so does "not numpy.ndarray", so check for existence separately
        if (not isinstance(data_list,numpy.ndarray) and not data_list) or (
            isinstance(data_list,numpy.ndarray) and len(data_list)==0):
            continue
        elif isinstance(data_list,tuple):
            if os.path.isfile(data_list[0]):
                already_written_schema.append(_coerce_schema(data_list[1]))
                already_written_files.append(data_list[0])
            else:
                raise RuntimeError(("Data tuple appears to point to an existing file %s, but that "+
                                   "file is not found according to os.path.isfile()")%data_list[0])
        elif isinstance(data_list,OSFile):
            if os.path.isfile(data_list.file_name):
                already_written_schema.append(_coerce_schema(data_list.fields))
                already_written_files.append(data_list.file_name)
            else:
                raise RuntimeError(("Data item appears to be an OSFile object, but does not point "+
                                   "to an existing object: %s")%data_list[0])
        elif hasattr(data_list,'__getitem__') and not hasattr(data_list,'dtype'):
            for dl in data_list:
                if isinstance(dl, tuple):
                    if os.path.isfile(dl[0]):
                        already_written_schema.append(_coerce_schema(dl[1]))
                        already_written_files.append(dl[0])
                    else:
                        raise RuntimeError(("Data tuple appears to point to an existing file %s, "+
                                            "but that file is not found according to "+
                                            "os.path.isfile()")%data_list[0])
                elif isinstance(dl,OSFile):
                    if os.path.isfile(dl.file_name):
                        already_written_schema.append(_coerce_schema(dl.fields))
                        already_written_files.append(dl.file_name)
                    else:
                        raise RuntimeError("Data item appears to be an OSFile object, but does "+
                                           "not point to an existing object: %s"%data_list[0])
                elif not hasattr(dl,'dtype'):
                    raise ValueError("Cannot understand data: "+str(data_list))
        elif not hasattr(data_list,'dtype'):
            raise ValueError("Cannot understand data: "+str(data_list))
    # Check existing field schema for consistency.  (Corr2 only allows one schema specification.)
    if already_written_schema:
        while True:
            all_same = True
            # First: straight-up equality
            for i in range(len(already_written_schema)-1):
                for j in range(i,len(already_written_schema)):
                    if not already_written_schema[i]==already_written_schema[j]:
                        all_same=False
            if all_same:
                break
            # Next: check the intersection of the schemas
            aw_keys = already_written_schema[0].keys()
            for aws in already_written_schema[1:]:
                for key in aw_keys:
                    if key not in aws and key in aw_keys:
                        del aw_keys[key]
            all_same = True
            for key in aw_keys:
                # Do they all point to the same column?
                n = set([aw[key] for aw in already_written_schema])
                if len(n)>1:
                    all_same = False
                    break
            if all_same:
                break
            else:
                # If they're inconsistent, remove the smallest file and repeat this loop
                sizes = [os.path.getsize(awf) for awf in already_written_files]
                remove = sizes.index(min(sizes))
                to_write.append(already_written_files[remove])
                del already_written_files[remove]
                del already_written_schema[remove]
        # Keep only the intersection of the already-written schemas!
        fields = already_written_schema[0]
        for aw in already_written_schema[1:]:
            keys = fields.keys()
            for key in keys:
                if key not in aw:
                    del fields[key]
    else:
        fields = []

    # Now make sure we have all the fields we need.
    has_fields = True if fields else False
    for data_list in [data, data2, random, random2]:
        has_fields, fields = _check_fields(has_fields,already_written_files,fields,data_list)
    if has_fields and not fields:
        raise RuntimeError('Intersection of field description for data files to write is empty.')
    new_data = []
    new_data2 = []
    new_random = []
    new_random2 = []
    # Now loop through again and write to a file any data arrays we need to.
    for data_list, new_data_list in [(data,new_data), (data2,new_data2), 
                                     (random,new_random), (random2, new_random2)]:
        if (not isinstance(data_list,numpy.ndarray) and not data_list) or (
                isinstance(data_list,numpy.ndarray) and len(data_list)==0):
            continue
        elif isinstance(data_list,tuple):
            if os.path.isfile(data_list[0]):
                data_fields = _coerce_schema(data_list[1])
                # The "any(...)" bit here is to protect against the case where you use two identical
                # filenames but different field descriptions, which would otherwise be caught by
                # this check and rewritten.  (You could picture doing this if you had a catalog with
                # two different shape definitions and wanted to correlate them, for example.)
                if data_list[0] in to_write and any(
                                          [data_fields[key]!=fields[key] for key in fields.keys()]):
                    data = file_io.ReadTable(data_list[0],fields=data_list[1])
                    new_data_list.append(OSFile(data,fields=fields))
                    to_write.remove(data_list[0])
                else:
                    new_data_list.append(data_list[0])
        elif isinstance(data_list,OSFile):
            new_data_list.append(data_list)
        elif hasattr(data_list,'__getitem__'):
            if isinstance(data_list[0],tuple):
                for dl in data_list:
                    data_fields = _coerce_schema(dl[1])
                    if dl[0] in to_write and any(
                                          [data_fields[key]!=fields[key] for key in fields.keys()]):
                        data = file_io.ReadTable(dl[0],fields=dl[1])
                        new_data_list.append(OSFile(data,fields=fields))
                    else:
                        new_data_list.append(dl[0])
            else:
                if hasattr(data_list,'dtype') and data_list.dtype.names: 
                    new_data_list.append(OSFile(data_list,fields=fields))
                else: 
                    for dl in data_list:
                        if not hasattr(dl,'dtype') or not hasattr(dl.dtype,'names'):
                            raise RuntimeError("Cannot parse data: should be a tuple, "+
                                               "numpy array, or an unmixed list of one or the "+
                                               "other.  Given:"+str(data_list))
                        new_data_list.append(OSFile(dl,fields=fields))
        
    
    # Lists of files need to be written to a separate file; do that.
    file_args = []
    for file_list in [new_data, new_data2, new_random, new_random2]:
        if len(file_list)>1:
            file_args.append(('list',OSFile(file_list)))
        elif len(file_list)==1:
            file_args.append(('name',file_list[0]))
        else:
            file_args.append(None)
    
    corr2_kwargs = MakeCorr2Cols(fields,use_as_k=use_as_k)
    if file_args[0]:
        corr2_kwargs['file_'+file_args[0][0]] = file_args[0][1]
    if file_args[1]:
        corr2_kwargs['file_'+file_args[1][0]+'2'] = file_args[1][1]
    if file_args[2]:
        corr2_kwargs['random_'+file_args[2][0]] = file_args[2][1]
    if file_args[3]:
        corr2_kwargs['random_'+file_args[3][0]+'2'] = file_args[3][1]
    return corr2_kwargs

    
