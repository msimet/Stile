"""@file file_io.py
Simple file input/output.
"""

try:
    import pyfits as fits_handler
    has_fits=True
except:
    try:
        import astropy.io.fits as fits_handler
        has_fits=True
    except:
        has_fits=False
import numpy

def read_fits_image(file_name,hdu=0):
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
    
def read_fits_table(file_name,hdu=1):
    """
    This function exists so you can read_fits_table(file_name) rather than remembering that
    table data is usually in extension 1.
    
    @param file_name A path leading to a valid FITS file
    @param hdu       The HDU in which the requested data is located (default: 1)
    @returns         The contents of the requested HDU
    """
    return read_fits_image(file_name,hdu=1)

def read_ascii_table(file_name, startline=None, comment=None):
    """
    Read in an ASCII table, and represent it via the simplest possible form for each column.  
    
    If you know your data contains only numbers (no strings), type safety and proper column
    alignment can be better handled by the functions numpy.loadtxt() [complete table] or
    numpy.genfromtxt() [missing columns].  This function is mostly useful if you have
    string data and don't want to skip those columns or turn the file into a FITS table.
    
    @param file_name The name of the file containing the ASCII table
    @param startline The number of the line to start on. startline=1 skips the first line of the
                     file, startline=2 skips the first two lines, etc. (default: None)
    @param comment   Ignore any lines whose first non-whitespace characters are this string.
                     (default: None)
    @returns         a numpy array containing the data in the file file_name
    """
    from stile_utils import get_vector_type

    f=open(file_name,'r')
    if startline:
        for i in range(startline-1):
            f.readline()
    d = [line.split() for line in f.readlines()]
    f.close()
    if len(d[-1])==0: # In case of trailing newline
        d = d[:-1]
    if comment:
        lenc = len(comment)
        d = [tuple(dd) for dd in d if dd[0].strip()[:lenc]!=comment]
    else:
        d = [tuple(dd) for dd in d]
    if len(d)==0:
        return []
    d_arr = numpy.array(d)
    types = ','.join([get_vector_type(d_arr[:,i]) for i in range(len(d_arr[0]))])
    return numpy.array(d,dtype=types)

def write_point(f,line,pos):
    if pos>=0:
        f.write(str(line[pos])+' ')
    else:
        f.write('0 ')
    
def write_ascii_table(file_name,data_array,cols=None):
    if not cols:
        cols = [i for i in range(len(data_array.dtypes.names))]
    else:
        tcols = [i for i in range(len(cols))]
        names = data_array.dtype.names
        for i,col in enumerate(cols):
            if col in names:        
                tcols[i] = names.index(col)
            else:
                tcols[i] = -1
    with open(file_name,'w') as f:
        for line in data_array:
            [write_point(f,line,pos) for pos in cols]
            f.write('\n')    
    
