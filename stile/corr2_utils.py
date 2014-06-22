"""@file corr2_utils.py
Contains elements of Stile needed to interface with Mike Jarvis's corr2 program.
"""
import copy
import numpy

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
                   help="RA-column units (radians, hours, degrees, arcmin, arcsec) -- only allowed "+
                        "by certain DataHandlers",
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
            raise ValueError('Argument %s not understood by Stile and not a recognized corr2 '
                               'argument.  Please check syntax and try again.'%key)                         
        else:
            c2k = corr2_kwargs[key]
            if check_status:
                if c2k['status']=='disallowed_file':
                    raise ValueError('Argument %s for corr2 is forbidden by Stile, which may need '+
                                     'to write multiple output files of this type.  Please remove '+
                                     'this argument from your syntax, and check the documentation '+
                                     'for where the relevant output files will be located.'%key)
                elif c2k['status']=='disallowed_computation':
                    raise ValueError('Argument %s for corr2 is forbidden by Stile, which controls '+
                                     'the necessary correlation functions.  Depending on your '+
                                     'needs, please either remove this argument from your syntax '+
                                     'or consider running corr2 as a standalone program.'%key)
            if type(input_dict[key]) not in c2k['type']:
                # The unknown arguments are passed as strings.  Since the string may not be the
                # desired argument, try casting the value into the correct type or types and see if
                # it works or raises an error; if at least one works, pass, else raise an error.
                type_ok = False
                for arg_type in c2k['type']:
                    try:
                        arg_type(input_dict[key])
                        type_ok=True
                    except:
                        pass
                if not type_ok:
                    raise ValueError(("Argument %s is a corr2 argument, but the type of the given "+
                                     "value %s does not match corr2's requirements.  Please "+
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
    for key in corr2_dict:
        if key in corr2_kwargs:
            f.write(key+' = ' + str(corr2_dict[key])+'\n')
        else:
            raise ValueError("Unknown corr2 key %s."%key)
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
    import file_io
    #output = numpy.loadtxt(file_name)
    # Currently there is a bug in corr2 that puts some text output into results files...
    output = file_io.ReadAsciiTable(file_name,comment='R',start_line=2)
    
    if not len(output):
        raise RuntimeError('File %s (supposedly an output from corr2) is empty.'%file_name)
    with open(file_name) as f:
        fields = f.readline().split()
    fields = fields[1:]
    fields = [field for field in fields if field!='.']
    return stile_utils.FormatArray(output,fields=fields,only_floats=True)

def AddCorr2Dict(input_dict):
    """
    Take an input_dict, harvest the kwargs you'll need for corr2, and create a new 'corr2_args'
    key in the input_dict containing these values (or update the existing 'corr2_args' key).
    
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
    @param use_as_k Which column to use as the "kappa" (scalar) column, if given (default: None)
    @returns        A dict containing the column key-value pairs for corr2
    """
    corr2_kwargs = {}
    col_args = ['x','y','ra','dec','g1','g2','k','w']
    if isinstance(cols,dict):
        for col in col_args:
            if col in cols and isinstance(cols[col],int):
                corr2_kwargs[col+'_col'] = cols[col]+1 # corr2 ordering starts at 1, Stile at 0
        if use_as_k and use_as_k in cols and isinstance(cols[use_as_k],int):
            corr2_kwargs['k_col'] = cols[use_as_k]+1
    elif hasattr(cols,'__getitem__'):
        for cp in col_args:
            if cp in cols:
                corr2_kwargs[cp+'_col'] = cols.index(cp)+1
        if use_as_k and use_as_k in cols:
            corr2_kwargs['k_col'] = cols.index(use_as_k)+1
    return corr2_kwargs

