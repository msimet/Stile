In the rest of these examples, we assume that you have installed PyYAML, which allows the reading of YAML configuration files.  Everything that follows will also work with the builtin JSON, once the required format changes to the configuration file have been made.

RUNNING STILE FROM A CONFIGURATION FILE

To start running Stile from a configuration file, you only need to type:
/path/to/Stile.py config.yaml
Stile.py is found in the bin/ directory of the Stile package you downloaded from Github.

CONTENTS OF A CONFIGURATION FILE

THE "FILE" ITEM(S)

You need to tell Stile where your data is, how to read it, and what kind of data it is.  We classify our data four ways:
- EPOCH: You can have "single" or "multiepoch" data.  This is a statement about data *structure* and not data *provenance*: if you have a coadded catalog, based on many images taken at multiple times, but containing only one (summarized) measurement per object, that is a "single" data set.  It is only "multiepoch" if you have multiple measurements of the same quantity for each object.  The Stile tests assume different inputs for the two cases, so you shouldn't mix them.
- EXTENT: We have four levels of spatial extent: "CCD", "field", "patch" and "tract".  The terminology comes from the LSST pipeline, but you don't need to use it exactly; the spatial extent is merely a guide for the sorts of tests you'd be interested in running.  A "CCD" is a single CCD, a "field" a single pointing/field-of-view, a "patch" a somewhat larger area, and a "tract" the largest area you're interested in doing systematics tests on.
- DATA FORMAT: "Catalog" or "image."  We do not currently implement any image-level tests, but we retain this keyword for future development.
- OBJECT TYPE: What kind of object is in your catalog.  Roughly, we expect "star" and "galaxy" object types, with some subclasses that use "star" or "galaxy" as their first word: "star PSF", the stars used in PSF determination; "star not PSF", the stars not used in PSF determination; "star bright", an especially bright star sample (your definition of "especially bright"); "galaxy lens", objects you'd like to compute lensing signals etc around (for systematics only--not for science!), "galaxy random", random points distributed in the same way as your galaxies; etc.

We expect a list of files in a configuration file with appropriate descriptions of what's in them.  You can specify this multiple times to describe different data sets, with the following caveats:
    - You can't use the same key in the config file more than once, so you should number your file descriptions: "file0", "file1", etc.  Just make sure the first four characters are "file" and Stile will interpret it for you (except "file_reader", which has its own meaning--see below).
    - Stile will NOT collapse multiple definitions of the same file into one definition unless all of its associated parameters--fields specification, which function to use as a file reader, etc--are the same.  If you include a file twice and those parameters are different, it will get analyzed twice.

There are two main ways you can specify your file information:

AS LIST OF DICTS:
```
 - epoch: 'single', extent: 'CCD', data_format: 'catalog', object_type: 'star', name: 'star0.fits'
 - epoch: 'single', extent: 'CCD', data_format: 'catalog', object_type: 'galaxy', name: 'galaxy0.fits'
 - epoch: 'single', extent: 'CCD', data_format: 'catalog', object_type: 'star', name: 'star1.fits'
 - epoch: 'single', extent: 'CCD', data_format: 'catalog', object_type: 'galaxy', name: 'galaxy1.fits'
 - epoch: 'single', extent: 'CCD', data_format: 'catalog', object_type: 'star', name: 'star2.fits'
 - epoch: 'single', extent: 'CCD', data_format: 'catalog', object_type: 'galaxy', name: 'galaxy2.fits'
 - epoch: 'single', extent: 'field', data_format: 'catalog', object_type: 'star', name: 'star3.fits'
 - epoch: 'single', extent: 'field', data_format: 'catalog', object_type: 'galaxy', 
   name: 'galaxy3.fits'
```
A little verbose, but clear.  

Some tests require two object types from the same region of sky: say, a B-mode signal around random points, for which you need both "galaxy random" and "galaxy" types.   In that case, you can add a "group" keyword to the dicts, matching up different object types:
```
 - epoch: 'single', extent: 'CCD', data_format: 'catalog', object_type: 'star', name: 'star0.fits',
   group: 0
 - epoch: 'single', extent: 'CCD', data_format: 'catalog', object_type: 'galaxy', 
   name: 'galaxy0.fits', group: 0
 - epoch: 'single', extent: 'CCD', data_format: 'catalog', object_type: 'star', name: 'star1.fits'
 - epoch: 'single', extent: 'CCD', data_format: 'catalog', object_type: 'galaxy', name: 'galaxy1.fits'
 - epoch: 'single', extent: 'CCD', data_format: 'catalog', object_type: 'star', name: 'star2.fits'
 - epoch: 'single', extent: 'CCD', data_format: 'catalog', object_type: 'galaxy', name: 'galaxy2.fits'
 - epoch: 'single', extent: 'field', data_format: 'catalog', object_type: 'star', name: 'star3.fits'
 - epoch: 'single', extent: 'field', data_format: 'catalog', object_type: 'galaxy', 
   name: 'galaxy3.fits'
 ```
Now, Stile will assume that a test that needs both a galaxy and a star sample can use 'star0.fits' and 'galaxy0.fits'.  It doesn't matter what you call the "groups" as long as they're unique.  (Also, passing more than one file of the same object_type to a group will result in an error.)
   
If you want, you can pass a list of filenames instead of a single filename:   
```
 - epoch: 'single', extent: 'CCD', data_format: 'catalog', object_type: 'star', 
   name: 'star0.fits', 'star1.fits', 'star2.fits'
 - epoch: 'single', extent: 'CCD', data_format: 'catalog', object_type: 'galaxy', 
   name: 'galaxy0.fits', 'galaxy1.fits', 'galaxy2.fits'
```
   
If most of your data is in a group, or if you just want to type less, you can pass your data as a nested dict, with the innermost bit being a list of files:
```
'single':
    'CCD':
        'catalog':
            'star': star0.fits, star1.fits, star2.fits
            'galaxy': galaxy0.fits, galaxy1.fits, galaxy2.fits
    'field':
        'catalog':
            'star': star3.fits
            'galaxy': galaxy3.fits
```
Each file in the list is assumed to be a separate data set.  In this format, if all the object types with the same extent, epoch and data format have a list of the same length, Stile will assume that each POSITION in that list is a group.  So in this previous example, 'star0.fits' and 'galaxy0.fits' would be assumed to go together, as would 'galaxy1.fits' and 'star1.fits', and as would 'star3.fits' and 'galaxy3.fits'.

The order of the nesting isn't important: the following is just fine.
```
 'catalog':
   'single':
     'CCD':
       'star': star0.fits, star1.fits, star2.fits
       'galaxy': galaxy0.fits, galaxy1.fits, galaxy2.fits
     'field':
       'star': star3.fits
       'galaxy': galaxy3.fits
```
And the order doesn't need to be the same at each level, as long as it's consistent within each key.  So this is also fine:
```
 'catalog':
   'single':
     'CCD':  
       'star': star0.fits, star1.fits, star2.fits
       'galaxy': galaxy0.fits, galaxy1.fits, galaxy2.fits
   'multiepoch"
     'star':
        'CCD': star3.fits
     'galaxy':
        'CCD': galaxy3.fits
```
       
Note that if the lists of files aren't all the same length, the grouping assumption goes away.  So the following dict:
```
 'catalog':
   'single':
     'CCD':
       'star': star0.fits, star1.fits
       'galaxy': galaxy0.fits, galaxy1.fits, galaxy2.fits
     'field':
       'star': star3.fits
       'galaxy': galaxy3.fits
```
would consider "star3.fits" and "galaxy3.fits" to be a grouping, but wouldn't group "star0.fits" and "galaxy0.fits" because it doesn't know how to align the star and galaxy lists properly.  You could pass the "spare" galaxy catalog as an individual dict to force the automatic grouping for the other two. 

Finally, if you need more information in the inner level, you can make those items a dict:
```
'single':
    'CCD':
        'catalog':
            'star': star0.fits, star1.fits, {'name': star2.dat, 'file_reader': ASCII}
            'galaxy': galaxy0.fits, galaxy1.fits, galaxy2.fits
```
Just make sure the filename is indicated with "name".  The keywords in that innermost dict will override any higher-level keys in the dict except the format keys (re-defining the format keys will result in an error).

You can also turn off the manual grouping by adding "group: False", either to the top level of your config file (in which case all grouping is turned off unless overridden by an explicit group= keyword), or at any level of a nested dict, which will turn off grouping for that level and all its sub-levels.  In addition, manually adding a group number will cause Stile to try to group only the items that don't already have a group number; the following dict:
```
 'catalog':
   'single':
     'CCD':
       'star': star0.fits, star1.fits
       'galaxy': galaxy0.fits, galaxy1.fits, {name: galaxy2.fits, group: False}
     'field':
       'star': star3.fits
       'galaxy': galaxy3.fits
```
would match star0 with galaxy0, star1 with galaxy1, and star3 with galaxy3.

Finally, you can use a "wildcard: True" flag at any level of your dicts, which will pass the filename you specify through the Python module "glob".  "Glob" works roughly like Bash wildcarding: asterisks are any number of characters including 0 characters, question marks are single characters, etc.  The list of files will then be sorted by Python string sort.  The grouping still applies, too.  So, for example, if you had a directory with:
star000.dat, star001.dat, ..., star999.dat, galaxy000.dat, galaxy001.dat, ..., galaxy999.dat
then you could pass a dict:
```
'wildcard': True,
'catalog':
    'single':
        'field':
            'star': 'star*.dat',
            'galaxy': 'galaxy*.dat'
```
and the wildcards would be expanded (yielding all 2000 files) and then the resulting lists would pair star000.dat with galaxy000.dat, star001.dat with galaxy001.dat, etc.

MULTIEPOCH DATA SETS
Since multiepoch data sets contain multiple files, the processing works a little differently.  Instead of a single file, Stile expects a list.  So the following config item:
```
 'catalog':
   'CCD':
     'multiepoch':
       'star': star0.fits, star1.fits
```
would only give you one data set: a multiepoch-CCD-catalog format with a "star" object type that would cycle through [star0.fits, star1.fits] .  If you instead want to specify multiple sets, you should specify a list of lists:
```
 'catalog':
   'CCD':
     'multiepoch':
       'star': [star0-0.fits, star0-1.fits], [star1-0.fits, star1-1.fits]
```
Whatever is the innermost list level cannot contain the dict-type descriptions that are used in other file specifications.  This would fail:
```
 'catalog':
   'CCD':
     'multiepoch':
       'star': [star0-0.fits, {'name': star0-0?.fits, 'wildcard': True}], [star1-0.fits, star1-1.fits]
```
       
SPECIFYING FIELD SCHEMAS (COLUMN NAMES)

If your files are ASCII tables, or if they're FITS files with column names different from what you expect, Stile needs to know what all those columns should be called.  You can specify this with a "fields" keyword.  The easiest thing to do is probably a list:
`fields = ['ra','dec','id','z','g1','g2','comment']`
Certain columns need to be named what Stile expects: `ra, dec, g1, g2, w` [for weight] should all have those specific names.  But it's not a problem if you want to specify a bunch of other columns.

You can also specify via a dict, if you don't need to identify all the columns:
`fields = {'ra': 0, 'dec': 1, 'z': 15, 'g1': 25, 'g2': 26}`
Note that in this form, column numbers start from 0!  You can also use FITS field names instead of column numbers, if your data is in a FITS table.  (You don't need to specifically tell Stile any FITS field names that already have the right names--it will figure it out.)

Field schemas can be defined in the top level of your file, in which case they'll be used for all the files that don't have a more specific field schema defined.  You can also define the schema at any level of your dicts:
```
 'fields': ['ra','dec','g1','g2']
 'catalog':
   'single':
     'CCD':
       'fields': ['ra','dec','x','y','g1','g2']
       'star': star0.fits, star1.fits, star2.fits
       'galaxy': galaxy0.fits, galaxy1.fits, 
                 {'name': galaxy2.fits, 'fields': ['ra','dec','z','g1','g2']}
     'field':
       'star': star3.fits
       'galaxy': galaxy3.fits
```       

       
If you want, you can also define the fields by format, like you defined the files:
```
 'catalog':
    'single':
       'fields': ['ra','dec','x','y','g1','g2']
    'multiepoch':
       'fields': ['ra','dec','g1','g2']
```
This is ignored anywhere you gave the `fields` argument directly in the original files dict.

SPECIFYING OBJECT FLAGS

What if you have a single file containing your stars, but you want to be able to select only the stars used as PSFs for certain tests?  Then you can pass a "flag_field" keyword as well.  When Stile reads the data in, it will only keep the data with data['flag_field'] == True.  You can also pass a list of flag fields; then it only keeps the data where data[flag_field_1] == True AND data[flag_field_2] == True, etc.  The fields should be the same names you passed in the field schema--not the column numbers or the FITS fields (unless the FITS fields have the same name you want).

SPECIFYING FILE READERS

By default, Stile will assume your files are pure ASCII or FITS tables--the latter if the filename ends in '.fit' or '.fits' in any capitalization.  But you can also tell it specifically which one to use.  You can also specify "unpickle" if your data was made in Python and then pickled.

If you need to change something about the way Stile reads things by default--if your comments start with "%" instead of the default "#", for example, you can specify file reader keywords by making your file reader specification a dict:

file_reader: {'name': ASCII, comment='%')

The ASCII keyword arguments are the arguments for numpy.genfromtxt(); the FITS argument keywords are for pyfits.open() [or astropy.io.fits.open()].  Note that by default the data is read from hdu=1, the first extension, where table data is normally found.

If you're using Stile directly from Python, instead of the command line, you can also define the file reader as a function of your own.  It should take as its argument only the filename you specified in the config file/config dict, and also any keywords you pass in the file_reader key.

THE "TESTS" ITEM

By default, Stile will run a set of tests described in the documentation.  If you want to change this, you can pass a "tests" key in your config file, listing the names of the different tests as described in the documentation.  You can do this either as a straight-up list, in which case all tests will be run on all data; or you can pass it as a nested dict just like the data.

Since the "Stat" item requires a field to be run on, you should specify "stat-<fieldname>" to run the Stat test on the given field name.  You can pass multiple stat-<fieldname> objects for different field names.

CORR2 KWARGS

Finally, if you are doing any correlation function tests, you can pass certain arguments from Corr2 to change its behavior.  Generally you can't pass anything that has to do with input or output files, since Stile controls that; don't worry, though, it will complain if you pass a keyword that's not allowed.
