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

def ReadAsciiTable(file_name, **kwargs):
    import stile_utils
    d = numpy.genfromtxt(file_name,dtype=None,**kwargs)
    return stile_utils.FormatArray(d)
    
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
    
