Data structure
==============

Stile requires that data be in a format that can be indexed by a column name. That means you should be able to do

>>> ra = data['ra']

to get the right ascension of your data.  There is a standard set of Stile column names:

- **dec**, the declination of the object
- **ra**, the RA of the object
- **x**, the x coordinate of the object
- **y**, the y coordinate of the object
- **g1**, a shear component in the ra or x direction
- **g2**, a shear component 45 degrees from the ra or x direction
- **sigma**, a size parameter for objects with dimension [length] in arbitrary units
- **psf_g1**, the g1 of the psf at the location of this object
- **psf_g2**, the g2 of the psf at the location of this object
- **psf_sigma**, the sigma of the psf at the location of this object
- **w**, the weight to apply per object
- **z**, the redshift of the object

Of course, not every data array needs to include all of these columns!

For some tests, a dict would be okay. However, we usually assume that the data is in a contiguous array so we can mask it:

>>> masked_data = data[mask]

which does not work on a dict.  The data types that can handle both masking and column calls via names are: the FITS catalogs of ``pyfits`` or ``astropy``; NumPy formatted arrays; and NumPy record arrays.  The difference between a formatted array and a recarray is that a recarray has an extra layer of Python code when calling columns by strings.  That could cause some slowdown if you're doing a large number of indexing calls.  On the other hand, they're a lot easier to make.

To make a recarray from vectors ``ra``, ``dec``, ``g1``, ``g2``, and ``w``:

>>> numpy.rec.fromarrays([ra, dec, g1, g2, w], names=['ra', 'dec', 'g1', 'g2', 'w'])

Note that this will cause all columns to be in the same format--so if any of those are a string, all columns will appear as strings.  You can get around this with more complicated data types, as explained in the documentation for :func:`numpy.core.records.fromarrays`.

To make a formatted array from an existing array, we have a Stile helper function called :func:`stile.FormatArray <stile.stile_utils.FormatArray>`.  From a data array ``arr`` with ``ra`` in the 0th column, ``dec`` in the 1st, ``g1`` in the 2nd, ``g2`` in the 3rd, and ``w`` in the 4th, you can do either:

>>> stile.FormatArray(arr, fields=['ra', 'dec', 'g1', 'g2', 'w'])

or

>>> stile.FormatArray(arr, fields={'ra': 0, 'dec': 1, 'g1': 2, 'g2': 3, 'w': 4})

The dict form doesn't need to give names to all the columns, but the list form does.  The dict form can also have strings as the values if you're rewriting column names from an existing formatted array/recarray/FITS catalog.  If the array was not previously a formatted or structured array of some kind, all columns will be cast to the most complex column type; if the array `was` already a formatted or structured array, only the names of the columns will change.

Once you have this array, it can be reused for all tests on that data.
