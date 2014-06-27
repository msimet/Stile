"""@file file_io.py
Simple file input/output.
"""

try:
    import pyfits as fits_handler
    has_fits=True
except ImportError:
    try:
        import astropy.io.fits as fits_handler
        has_fits=True
    except ImportError:
        has_fits=False
import numpy
import os

def ReadFITSImage(file_name,hdu=0):
    """
    Return the data from a single HDU extension of the FITS file file_name.  Technically this
    doesn't have to be an image as the method is the same for table data; it's called "image" 
    because of the default hdu=0 which can only be image data.
    
    Header information is not preserved, as it is not used by Stile.  Any scaling of the data
    will be done automatically when the data is accessed.
    
    @param file_name A path leading to a valid FITS file
    @param hdu       The HDU in which the requested data is located (default: 0)
    @returns         The contents of the requested HDU
    """
    if has_fits:
        fits_file = fits_handler.open(file_name)
        data = fits_file[hdu].data
        fits_file.close()
        return data
    else:
        raise ImportError('No FITS handler found!')
    
def ReadFITSTable(file_name,hdu=1):
    """
    This function exists so you can read_fits_table(file_name) rather than remembering that
    table data is usually in extension 1.
    
    @param file_name A path leading to a valid FITS file
    @param hdu       The HDU in which the requested data is located (default: 1)
    @returns         The contents of the requested HDU
    """
    return ReadFITSImage(file_name,hdu)

def ReadASCIITable(file_name, **kwargs):
    """
    Read an ASCII table from disk.  This is a small wrapper for numpy.genfromtxt() that returns the
    kind of array we expect.  **kwargs should be suitable kwargs from numpy.genfromtxt().
    """
    import stile_utils
    d = numpy.genfromtxt(file_name,dtype=None,**kwargs)
    return stile_utils.FormatArray(d)

    
# numpy.savetxt uses a completely different format specification language than the dtypes, so
# this dict and the function _format_str take a formatted NumPy array and return something
# that savetxt understands.  I've left the default field width (18 characters) for all
# fields except the string-like ones, which have their own default widths, and the object-like
# ones, which tend to have width ~1 when their string representations are longer--right now I set 
# that to 60.
_fmt_dict = {'?': 'u', 'B': 'c', 'I': 'u', 'H': 'u', 'L': 'u', 'Q': 'u', 'b': 'c', 'd': 'g', 'g': 'g', 'f': 'g', 'i': 'd', 'h': 'd', 'l': 'd', 'q': 'd'}

def _format_str(dtype):
    if dtype.names:
        return [_format_str(dtype[i]) for i in range(len(dtype))]
    else:
        char = dtype.char
        if char=='S' or char=='V' or char=='U':
            width = dtype.str.split(char)[-1]
            return '%'+str(width)+'s'
        elif char=='O':
            # Objects tend to have width-1 even if their string representations don't.
            return '%-60s'
        elif char=='B' or char=='G' or char=='F':
            return '%18g %18g'
        else:
            return '%18'+_fmt_dict[char]

def _handleFields(data_array,fields):
    """
    Rearrange the data according to the fields specification.
    """
    data = numpy.array(data_array)
    if not fields:
        pass
    elif not data.dtype.names:
        raise ValueError('Fields kwarg only usable if data is a formatted NumPy array')
    elif isinstance(fields,(tuple,list)):
        if not len(set(fields))==len(fields):
            raise RuntimeError('Field description list has duplicate elements')
        if isinstance(fields,tuple):
            fields = list(fields)
        data = data[fields]
    elif isinstance(fields,dict):
        # Make a list that's only as long as it needs to be to cover the fields dict; populate it 
        # with the keys of fields, and then fill in any blank spaces with the unused fields from
        # the original column descriptors.
        old_fields = [name for name in data.dtype.names if name not in fields]
        new_fields = ['']*(max(fields.values())+1)
        for key in fields:
            new_fields[fields[key]] = key
        for i in range(len(new_fields)):
            if not new_fields[i]:
                new_fields[i] = old_fields.pop(0)
        data = data[new_fields]
    else:
        raise ValueError("Fields description not understood: "+str(fields))
    return data
            
def WriteASCIITable(file_name,data_array,fields=None):
    """
    Given a file_name and a data_array, write the data_array to the file_name as an ASCII file.
    If fields is not None, this will rearrange a NumPy formatted array to the field 
    specification (must be either a list of field names, or a dict of the form 
    'field_name': field_position.  Note that in the second case, columns not indicated by the dict
    are moved around to fill in any gaps, so if you specify, say, columns 0, 1, and 3, you may be
    surprised by what is in column 2!
    
    At the moment, if your maximum column number in the fields dict is greater than the number of
    fields in the data_array, an error will occur.  Also, if you send an object in the array whose 
    string representation is >60 characters, it will be truncated to 60.  If you have strings which
    contain spaces, the column descriptions won't hold properly, and you should probably use a 
    FITS file writer.
    """
    data = _handleFields(data_array,fields)
    numpy.savetxt(file_name,data,fmt=_format_str(data.dtype))

_fits_dict = {'f': 'D', 'i': 'K'}
def _coerceFitsFormat(fmt):
    if 'S' in fmt.str:
        return 'A'+fmt.str.split('S')[1]
    for key in _fits_dict:
        if key in fmt.str:
            return _fits_dict[key]
    return fmt

def WriteFITSTable(file_name,data_array,fields=None):
    """
    Given a file_name and a data_array, write the data_array to the file_name as a FITS file if
    there is an available module to do so (pyfits or astropy.io.fits).  Otherwise, raise an error.
    If fields is not None, this will rearrange a NumPy formatted array to the field specification 
    (must be either a list of field names, or a dict of the form 'field_name': field_position.  
    Note that in the second case, columns not indicated by the dict are moved around to fill in 
    any gaps, so if you specify, say, columns 0, 1, and 3, you may be surprised by what is in 
    column 2!
    
    At the moment, if your maximum column number in the fields dict is greater than the number of
    fields in the data_array, an error will occur.
    """
    if not has_fits:
        raise ImportError('FITS-type table requested, but no FITS handler found')
    data = _handleFields(data_array,fields)
    cols = [fits_handler.Column(name=data.dtype.names[i],format=_coerceFitsFormat(data.dtype[i]))
               for i in range(len(data.dtype))]
    table = fits_handler.new_table(cols)
    table.data = data
    # Next line is a kludge for some versions of PyFITS which will yell if you replace table data
    # without doing this (fix from https://github.com/spacetelescope/PyFITS/issues/31)
    table.data = numpy.array(table.data).view(table._data_type)
    hdulist = fits_handler.HDUList([fits_handler.PrimaryHDU(),table])
    hdulist.verify()
    hdulist.writeto(file_name)
    
def WriteTable(file_name,data_array,fields=None):
    """
    Pick a type of file (ASCII or FITS) and write to it.  If the file_name has an extention, it will
    be used to determine the file type ('.fit' or '.fits' in any capitalization will be FITS, else
    ASCII); if no extension, it will write a FITS file if a fits handler is found, else an ASCII 
    file.  If you know which kind of file you want to write, you should use WriteFITSTable or
    WriteASCIITable directly.
    
    If fields is not None, these functions will rearrange a NumPy formatted array to the field 
    specification (must be either a list of field names, or a dict of the form 'field_name': 
    field_position.  Note that in the second case, columns not indicated by the dict are moved 
    around to fill in any gaps, so if you specify, say, columns 0, 1, and 3, you may be surprised 
    by what is in column 2!
    
    At the moment, if your maximum column number in the fields dict is greater than the number of
    fields in the data_array, an error will occur.
    """
    ext = os.path.splitext(file_name)[1]
    if not ext:
        if has_fits:
            WriteFITSTable(file_name,data_array,fields)
        else:
            WriteASCIITable(file_name,data_array,fields)
    ext = ext.lower()
    if ext=='.fit' or ext=='.fits':
        WriteFITSTable(file_name,data_array,fields)
    else:
        WriteASCIITable(file_name,data_array,fields)

def ReadTable(file_name,**kwargs):
    """
    Pick a proper (FITS or ASCII) reading function for a file containing a table and read the file
    in.  If the file_name has an extention, it will be used to determine the file type ('.fit' or 
    '.fits' in any capitalization will be FITS, else ASCII); if no extension, it will try reading
    it as a FITS file, then as an ASCII file.  If you know which kind of file you want to read, 
    you should use WriteFITSTable or WriteASCIITable directly.
    """
    ext = os.path.splitext(file_name)[1]
    if not ext:
        try:
            return ReadFITSTable(file_name,**kwargs)
        except:
            return ReadASCIITable(file_name,**kwargs)
    ext = ext.lower()
    if ext=='.fit' or ext=='.fits':
        return ReadFITSTable(file_name,**kwargs)
    else:
        return ReadASCIITable(file_name,**kwargs)
