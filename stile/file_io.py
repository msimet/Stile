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

def ReadFitsImage(file_name,hdu=0):
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
    
def ReadFitsTable(file_name,hdu=1):
    """
    This function exists so you can read_fits_table(file_name) rather than remembering that
    table data is usually in extension 1.
    
    @param file_name A path leading to a valid FITS file
    @param hdu       The HDU in which the requested data is located (default: 1)
    @returns         The contents of the requested HDU
    """
    return read_fits_image(file_name,hdu=1)

def ReadAsciiTable(file_name, start_line=None, comment=None):
    """
    Read in an ASCII table, and represent it via the simplest possible form for each field.  
    
    If you know your data contains only numbers (no strings), type safety and proper field
    alignment can be better handled by the functions numpy.loadtxt() [complete table] or
    numpy.genfromtxt() [missing fields].  This function is mostly useful if you have
    string data and don't want to skip those fields or turn the file into a FITS table.
    
    @param file_name The name of the file containing the ASCII table
    @param startline The number of the line to start on. startline=1 skips the first line of the
                     file, startline=2 skips the first two lines, etc. (default: None)
    @param comment   Ignore any lines whose first non-whitespace characters are this string.
                     (default: None)
    @returns         a numpy array containing the data in the file file_name
    """
    from stile_utils import GetVectorType

    f=open(file_name,'r')
    if start_line:
        for i in range(start_line-1):
            f.readline()
    d = [line.split() for line in f.readlines()]
    f.close()
    if not d:
        return numpy.array([])
    if comment:
        lenc = len(comment)
        d = [tuple(dd) for dd in d if dd and dd[0].strip()[:lenc]!=comment]
    else:
        d = [tuple(dd) for dd in d if dd]
    if len(d)==0:
        return []
    d_arr = numpy.array(d)
    types = ','.join([GetVectorType(d_arr[:,i]) for i in range(len(d_arr[0]))])
    return numpy.array(d,dtype=types)

def WritePoint(f,line,pos):
    if pos>=0:
        f.write(str(line[pos])+' ')
    else:
        f.write('0 ')
    
def WriteAsciiTable(file_name,data_array,fields=None):
    if not fields:
        fields = [i for i in range(len(data_array.dtype.names))]
    else:
        tfields = [i for i in range(len(fields))]
        names = data_array.dtype.names
        for i,field in enumerate(fields):
            if field in names:        
                tfields[i] = names.index(field)
            else:
                tfields[i] = -1
    with open(file_name,'w') as f:
        for line in data_array:
            [WritePoint(f,line,pos) for pos in fields]
            f.write('\n')    
    
