# How to use the Stile config interface #

Stile has an automatic driver for files and systematics tests defined using a config file.  You can
run this driver via the executable bin/StileConfig.py, passing one or more config files as arguments
on the command line.  In the rest of these examples, we assume that you have installed PyYAML, which
allows the reading of YAML configuration files.  Everything that follows will also work with the
builtin JSON, once the required format changes to the configuration file have been made.

## Table of contents ## 
I. Running Stile from a configuration file
II. Contents of a configuration file
    A. The 'file' Items
        1. As a list of dicts
        2. As a nested dict
        3. Other file specification keywords
            a. Field schemas (column names)
            b. File readers
            c. Flags
            d. Binning
    B. The 'sys_test' Items
    C. Global arguments for config files
III. Lists of options
    A. Field names
    B. Extra arguments for file specification
    C. Formats
    D. Object types

## Running STile from a configuration file ##

To start running Stile from a configuration file, you only need to type:
/path/to/StileConfig.py config.yaml [config2.yaml config3.yaml...]
StileConfig.py is found in the bin/ directory of the Stile package you downloaded from Github.
Config files will be combined into one large dictionary before being processed; later files will
override the keys of earlier files.

Configuration files contain specifiers for the *files* you would like to include, the *tests* you
would like to run on those files, and *optional other arguments* to control how processing proceeds.

## CONTENTS OF A CONFIGURATION FILE ##

### THE "FILE" ITEM(S) ###

You need to tell Stile where your data is, how to read it, and what kind of data it is.  We classify
our data four ways:
- *epoch*: You can have "single", "coadd", or "multiepoch" data.  This is a statement about data
*structure* and not data *provenance*: if you have a coadded catalog, based on many images taken at
multiple times, but containing only one (summarized) measurement per object, that is a "coadd" data
set.  It is only "multiepoch" if you have multiple measurements of the same quantity for each object.
The Stile tests assume different inputs for the two cases, so you shouldn't mix them.
- *extent*: We have three levels of spatial extent: "CCD", "field", and "survey".  You don't need to
use these terms in an exact sense; the spatial extent is merely a guide for the sorts of tests you'd
be interested in running.  A "CCD" is a single CCD, a "field" a single pointing/field-of-view, and a
"survey" a larger area.
- *data_format*: "Catalog" or "image."  We do not currently implement any image-level tests, but we
retain this keyword for future development.
- *object_type*: What kind of object is in your catalog.  Roughly, we expect "star" and "galaxy"
object types, with some subclasses that use "star" or "galaxy" as their first word: "star PSF", the
stars used in PSF determination; "star not PSF", the stars not used in PSF determination; "star
bright", an especially bright star sample (your definition of "especially bright"); "galaxy lens",
objects you'd like to compute lensing signals etc around (for systematics only--not for science!),
"galaxy random", random points distributed in the same way as your galaxies; etc.

We call the first three things (epoch, extent, and data_format) a "format".

We expect a list of files in a configuration file with appropriate descriptions of what's in them.
You can specify this multiple times to describe different data sets, with the following caveats:
    - You can't use the same key in the config file more than once, so you should number your file
      descriptions: "file0", "file1", etc, if you need to use more than one input key.  Just make
      sure the first four characters are "file" and Stile will interpret it for you (except 
      "file_reader", which has its own meaning--see below).
    - Stile will NOT collapse multiple definitions of the same file into one definition unless all
      of its associated parameters--fields specification, which function to use as a file reader,
      etc--are the same.  If you include a file twice and those parameters are different, it will
      get analyzed twice.

There are two main ways you can specify your file information:

#### As a list of dicts ####
```
 - epoch: single, extent: CCD, data_format: catalog, object_type: star, name: star0.fits
 - epoch: single, extent: CCD, data_format: catalog, object_type: galaxy, name: galaxy0.fits
 - epoch: single, extent: CCD, data_format: catalog, object_type: star, name: star1.fits
 - epoch: single, extent: CCD, data_format: catalog, object_type: galaxy, name: galaxy1.fits
 - epoch: single, extent: CCD, data_format: catalog, object_type: star, name: star2.fits
 - epoch: single, extent: CCD, data_format: catalog, object_type: galaxy, name:  galaxy2.fits
 - epoch: single, extent: field, data_format: catalog, object_type: star, name: star3.fits
 - epoch: single, extent: field, data_format: catalog, object_type: galaxy, name: galaxy3.fits
```
A little verbose, but clear.  

Some tests require two object types from the same region of sky: say, a B-mode signal around random
points, for which you need both "galaxy random" and "galaxy" types.   In that case, you can add a
"group" keyword to the dicts, matching up different object types:
```
 - epoch: single, extent: CCD, data_format: catalog, object_type: star, name: star0.fits, group: 0
 - epoch: single, extent: CCD, data_format: catalog, object_type: galaxy, name: galaxy0.fits,
   group: 0
 - epoch: single, extent: CCD, data_format: catalog, object_type: star, name: star1.fits
 - epoch: single, extent: CCD, data_format: catalog, object_type: galaxy, name: galaxy1.fits
 - epoch: single, extent: CCD, data_format: catalog, object_type: star, name: star2.fits
 - epoch: single, extent: CCD, data_format: catalog, object_type: galaxy, name:  galaxy2.fits
 - epoch: single, extent: field, data_format: catalog, object_type: star, name: star3.fits
 - epoch: single, extent: field, data_format: catalog, object_type: galaxy, name: galaxy3.fits
 ```
Now, Stile will assume that a test that needs both a galaxy and a star sample can use 'star0.fits'
and 'galaxy0.fits'.  It doesn't matter what you call the 'groups' as long as they're unique.  (Also,
passing more than one file of the same format and object_type to a group will result in an error.)
   
If you want, you can pass a list of filenames instead of a single filename:   
```
 - epoch: single, extent: CCD, data_format: catalog, object_type: star, 
   name: [star0.fits, star1.fits, star2.fits]
 - epoch: single, extent: CCD, data_format: catalog, object_type: galaxy, 
   name: [galaxy0.fits, galaxy1.fits, galaxy2.fits]
```
   
#### As a nested dict ####
   
If most of your data is in a group, or if you just want to type less, you can pass your data as a
nested dict, with the innermost bit being a list of files:
```
single:
    CCD:
        catalog:
            star: star0.fits, star1.fits, star2.fits
            galaxy: galaxy0.fits, galaxy1.fits, galaxy2.fits
    field:
        catalog:
            star: star3.fits
            galaxy: galaxy3.fits
```
Each file in the list is assumed to be a separate data set.  In this format, if all the object types
with the same extent, epoch and data format have a list of the same length, Stile will assume that
each POSITION in that list is a group.  So in this previous example, 'star0.fits' and 'galaxy0.fits'
would be assumed to go together, as would 'galaxy1.fits' and 'star1.fits', and as would 'star3.fits'
and 'galaxy3.fits'.  

The order of the nesting isn't important: the following is just fine.
```
 catalog:
   single:
     CCD:
       star: star0.fits, star1.fits, star2.fits
       galaxy: galaxy0.fits, galaxy1.fits, galaxy2.fits
     field:
       star: star3.fits
       galaxy: galaxy3.fits
```
And the order doesn't need to be the same at each level, as long as it's consistent within each key.
So this is also fine:
```
 catalog:
   single:
     CCD:  
       star: star0.fits, star1.fits, star2.fits
       galaxy: galaxy0.fits, galaxy1.fits, galaxy2.fits
   multiepoch:
     star:
        CCD: star3.fits
     galaxy:
        CCD: galaxy3.fits
```
But this is not:
```
 catalog:
   single:
     CCD:  
       star: star0.fits, star1.fits, star2.fits
       galaxy: galaxy0.fits, galaxy1.fits, galaxy2.fits
   CCD:
     star:
        multiepoch: star3.fits
     galaxy:
        multiepoch: galaxy3.fits
```
since it mixes keys of different types at the same level of the dict ("CCD" and "single").
       
Note that if the lists of files aren't all the same length, the grouping assumption goes away.  So
the following dict:
```
 catalog:
   single:
     CCD:
       star: star0.fits, star1.fits
       galaxy: galaxy0.fits, galaxy1.fits, galaxy2.fits
     field:
       star: star3.fits
       galaxy: galaxy3.fits
```
would consider "star3.fits" and "galaxy3.fits" to be a grouping, but wouldn't group "star0.fits" and
"galaxy0.fits" because it doesn't know how to align the star and galaxy lists properly.  You could
pass the "spare" galaxy catalog as an individual dict to force the automatic grouping for the other
two.  Note that wildcards (described below) or lists of files in a 'name' key are expanded BEFORE
grouping takes place, but definitions of binning (also described below) are NOT: if you define a
file list with one star file and one galaxy file, and you bin the star sample into three pieces,
Stile will still group the two files together and you will get three star/galaxy cross-correlation
outputs for that group.

Finally, if you need more information in the inner level, you can make those items a dict:
```
single:
    CCD:
        catalog:
            star: star0.fits, star1.fits, {name: star2.dat, file_reader: ASCII}
            galaxy: galaxy0.fits, galaxy1.fits, galaxy2.fits
```
Just make sure the filename is indicated with "name".  The keywords in that innermost dict will
override any higher-level keys in the dict except the format keys (re-defining the format keys will
result in an error).

You can also turn off the manual grouping by adding "group: False", either to the top level of your
config file (in which case all grouping is turned off unless overridden by an explicit group
keyword), or at any level of a nested dict, which will turn off grouping for that level and all its
sub-levels.  In addition, manually adding a group number will cause Stile to try to group only the
items that don't already have a group number; the following dict:
```
 catalog:
   single:
     CCD:
       star: star0.fits, star1.fits
       galaxy: galaxy0.fits, galaxy1.fits, {name: galaxy2.fits, group: False}
     field:
       star: star3.fits
       galaxy: galaxy3.fits
```
would match star0 with galaxy0, star1 with galaxy1, and star3 with galaxy3.  The same behavior would
occur if you defined the "group" keyword to be anything other than True: `group: 3` would also cause
the file to be ignored for grouping purposes.

The file names are, by default, used in the output filenames so you can tell which output file
belongs to which input file.  If your file names are long or unwieldy, you can define a `nickname`
keyword to use instead:
```
 catalog:
   single:
     CCD:
       star: {name: star0.fits, nickname: s0}, {name: star1.fits, nickname: s1}
       galaxy: galaxy0.fits, galaxy1.fits, {name: galaxy2.fits, group: False}
     field:
       star: star3.fits
       galaxy: galaxy3.fits
```

Finally, you can use a "wildcard: True" flag at any level of your dicts, which will pass the
filename you specify through the Python module "glob".  "Glob" works roughly like Bash wildcarding:
asterisks are any number of characters including 0 characters, question marks are single characters,
etc.  The list of files will then be sorted by Python string sort.  The grouping still applies, too.
So, for example, if you had a directory with:
star000.dat, star001.dat, ..., star999.dat, galaxy000.dat, galaxy001.dat, ..., galaxy999.dat
then you could pass a dict:
```
wildcard: True,
catalog:
    single:
        field:
            star: star*.dat,
            galaxy: galaxy*.dat
```
and the wildcards would be expanded (yielding all 2000 files) and then the resulting lists would
pair star000.dat with galaxy000.dat, star001.dat with galaxy001.dat, etc.

Multiepoch data sets work a little differently since they must contain multiple files.  Instead of a
single file, Stile expects a list.  So the following config item:
```
 catalog:
   CCD:
     multiepoch:
       star: star0.fits, star1.fits
```
would only give you one data item: a multiepoch-CCD-catalog format with a "star" object type that
would cycle through [star0.fits, star1.fits] .  If you instead want to specify multiple sets, you
should specify a list of lists:
```
 catalog:
   CCD:
     multiepoch:
       star: [star0-0.fits, star0-1.fits], [star1-0.fits, star1-1.fits]
```
Whatever is the innermost list level cannot contain the dict-type descriptions that are used in
other file specifications.  This would fail:
```
 catalog:
   CCD:
     multiepoch:
       star: [star0-0.fits, {name: star0-0?.fits, wildcard: True}], [star1-0.fits, star1-1.fits]
```

#### Other file specification keywords ####

##### Field schemas (column names #####

If your files are ASCII tables, or if they're FITS files with column names different from what you
expect, Stile needs to know what all those columns should be called.  You can specify this with a
"fields" keyword.  The easiest thing to do is probably a list:
`fields = [ra, dec, id, z, g1, g2, comment]`
which gives the columns in order.

You can also specify via a dict, if you don't need to identify all the columns:
`fields = {ra: 0, dec: 1, z: 15, g1: 25, g2: 26}`
Note that in this form, column numbers start from 0!  You can also use FITS field names instead of
column numbers, if your data is in a FITS table.  (You don't need to specifically tell Stile any
FITS field names that already have the right names.)

Certain columns need to be named what Stile expects: `ra, dec, g1, g2, w` [for weight] should all
have those specific names; see the end of this file for a complete list.  But it's not a problem if
you want to specify a bunch of other columns.  Field schemas can be defined in the top level of your
file, in which case they'll be used for all the files that don't have a more specific field schema
defined.  You can also define the schema at any level of your dicts:
```
 fields: [ra, dec, g1, g2]
 catalog:
   single:
     CCD:
       fields: [ra, dec, x, y, g1, g2]
       star: star0.fits, star1.fits, star2.fits
       galaxy: galaxy0.fits, galaxy1.fits, 
               {name: galaxy2.fits, fields: [ra, dec, z, g1, g2]}
     field:
       star: star3.fits
       galaxy: galaxy3.fits
```       
       
If you want, you can also define the fields by format, like you defined the files:
```
 catalog:
    single:
       fields: [ra, dec, x, y, g1, g2]
    multiepoch:
       fields: [ra, dec, g1, g2]
```
This is ignored anywhere you gave the `fields` argument directly in the original files dict.

##### File readers #####

By default, Stile will assume your files are pure ASCII or FITS tables--the latter if the filename
ends in '.fit' or '.fits' in any capitalization.  But you can also tell it specifically which one
to use.

If you need to change something about the way Stile reads things by default--if your comments start
with "!" instead of the default "#", for example, you can specify file reader keywords by making
your file reader specification a dict:

```
file_reader: {name: ASCII, extra_kwargs: {comment: !})
```

The ASCII keyword arguments are the arguments for numpy.genfromtxt(); the FITS argument keywords are
for pyfits.open() [or astropy.io.fits.open()].  Note that by default the data is read from hdu=1,
the first extension, where table data is normally found.

If you're using Stile directly from Python, instead of the command line, you can also define the
file reader as a function of your own.  It should take as its argument only the filename you
specified in the config file/config dict, and also any keywords you pass in the extra_kwargs key.

##### Flags #####

If you would like to eliminate certain items in your data file before processing the systematics
tests, you can specify the 'flag_field' keyword.  This can contain a field or list of fields which
must evaluate False in order to be kept:
```
 catalog:
   single:
     CCD:
       fields: [ra, dec, x, y, g1, g2, flux_error]
       flag_field: flux_error
       star: star0.fits, star1.fits, star2.fits
```
It can also contain a dict of 'field': 'value_to_keep' pairs:
```
 catalog:
   single:
     CCD:
       fields: [ra, dec, x, y, g1, g2, flux_error, is_star]
       flag_field: {'flux_error': False, 'is_star': True}
       star: star0.fits, star1.fits, star2.fits
         
```
*NOTE*: Unlike other extra file specification keys such as 'group' or 'file_reader', 'flag_field'
keys are *appended together* rather than replaced by lower-level definitions.  (This is so you can
define global bad-measurement flags, and also object flags at a lower level of the dict, for
example.

If you instead need a range of values, you could use a binning scheme that defines only one bin.

##### Binning #####

Binning can be done with a 'bins' kwarg, which takes a dict (for one bin) or a list of dicts (for
bins which will be ANDed together).  You can do 'Step' bins or 'List' bins.  'Step' bins define bins
of equal width:
```
name: Step, field: ra, low: 0, high: 10, n_bins: 4, step: 2.5, use_log: False
```
The 'field' argument says which field to bin on.  You only need to define three of the four
['low', 'high', 'step', 'n_bins'], though as long as all four agree processing will continue if you
pass all four.  'use_log' can be omitted if it's False; otherwise the bins are equal size in
logarithmic steps instead of linear steps, and the 'step' size is delta(log field), not 
delta(field), although the 'low' and 'high' endpoints should still be in 'field' and not 
'log(field)' space.

'List' just takes a list of the endpoints of the bins:
```
name: List, field: y, endpoints: [0, 1, 2, 3, 5]
```
and assumes the first bin goes from 0-1, the second from 1-2, etc.  The list must be monotonic.
Bins for both kinds ('List' and 'Step') are half-open intervals, low<=val<high.

If multiple bins are defined--say, a scheme with 3 bins in magnitude and a scheme with 2 bins in
size--then the output will include bins 1/3 & 1/2, then 2/3 and 1/2, then 3/3 and 1/2, then 1/3 and
2/2, etc.  If you want to bin separately--do magnitude bins including all sizes, then size bins
including all magnitudes--just define the file twice with different 'bins' kwargs.  (To maintain
automatic grouping, you could include the other files in the group twice--Stile will automatically
merge those two definitions after groups are defined as long as all the kwargs are the same.)

### The 'sys_tests' items ###

Systematics tests are defined like the files above, with the exception that you do not need to
completely specify the format for a sys_test: an incomplete definition will be matched with all
possible files.  In particular, you should not define the object_type except for the Stat type of
test, as the other tests already know which object types they need.

For example, if you had files in {'epoch': 'single', 'extent': 'CCD', 'data_format': 'catalog'},
{'epoch': 'single', 'extent': 'field', 'data_format': 'catalog'}, and {'epoch': 'coadd', 'extent':
'CCD', 'data_format': 'catalog'}, and you defined a test with the format {'extent': 'CCD'}, it would
apply to the files in the first and third formats, but not the files in the second format.  Note
that the tests are only run if files exist for that format: if you find that you have defined tests
which are not running, make sure that you have suitable files defined as well.

The keywords 'nickname', 'wildcard', 'fields', 'flag_field', 'file_reader', and 'group' have no
impact on sys_tests and may result in an error if included.  'bins' can be defined--and in fact it
is more I/O efficient to define them for tests than for the individual files--but they cannot be
defined for any test that requires multiple object_types.  (This functionality is planned, but not
yet implemented.)  

The specific tests and their required extra arguments are:
```
name: CorrelationFunction, type: <type>, extra_kwargs: <extra_kwargs>[, bins: bin_list]
```
CorrelationFunction types are:
    - GalaxyShear: g_t|x of 'galaxy' objects around 'galaxy lens' objects 
      (shear-density correlation)
    - BrightStarShear: g_t|x of 'galaxy' objects around 'star bright' objects 
      (shear-density correlation)
    - StarXGalaxyDensity: xi of 'star' crossed with 'galaxy' objects (density-density correlation)
    - StarXGalaxyShear: xi_+|- of 'star' crossed with 'galaxy' objects (shear-shear correlation)
    - StarXStarShear: xi_+|- autocorrelation of 'star' objects (shear-shear correlation)
    - GalaxyDensity: xi autocorrelation of 'galaxy' objects (density-density correlation)
    - StarDensity: xi autocorrelation of 'star' objects (density-density correlation)
    
The 'extra_kwargs' are required for CorrelationFunctionSysTests, because the TreeCorr package does
not assume certain values to ensure proper processing.  The keys you should define are 'ra_units'
and 'dec_units' (for ra, dec coords) or 'x_units' and 'y_units' (for x, y coords), all of which
should have values selected from 'deg', 'radians', 'arcsec', 'arcmin', 'hour'; plus 'min_sep',
'max_sep', 'nbins' and 'sep_units' to define the binning for the correlation function.

```
name: ScatterPlot, type: <type>[, bins: <bin_list>, extra_kwargs: <extra_kwargs>]
```
ScatterPlot types are:
    - StarVsPSFG1: 'g1' vs 'g1_psf' for 'star' objects
    - StarVsPSFG2: 'g2' vs 'g2_psf' for 'star' objects
    - StarVsPSFSigma: 'sigma' vs 'sigma_psf' for 'star' objects
    - ResidualVsPSFG1: 'g1_psf'-'g1' vs 'g1' for 'star' objects
    - ResidualVsPSFG2: 'g2_psf'-'g2' vs 'g2' for 'star' objects
    - ResidualVsPSFSigma: 'sigma_psf'-'sigma' vs 'sigma' for 'star' objects

```
name: WhiskerPlot, type: type[, bins: <bin_list>, extra_kwargs: <extra_kwargs>]
```
WhiskerPlot types are:
    - Star: whisker plot of star shapes ('g1', 'g2') for 'star' objects
    - PSF: whisker plot of PSF shapes ('g1_psf', 'g2_psf') for 'star' objects
    - Residual: whisker plot of (PSF-star) shapes

```
name: Stat, field: <field>, object_type: <object_type>[, bins: <bin_list>, 
extra_kwargs: <extra_kwargs>]
```
Stat tests can be defined for any field in any object_type in your catalog.  (Of course, you may
also limit this via formats the same way you can for any of the other tests.)

In general, possible kwargs for the 'extra_kwargs' field can be found in the Python documentation
strings for the objects CorrelationFunctionSysTest, ScatterPlotSysTest, etc.

### Global arguments for config files ###

`save_memory`: if True, prioritize clearing out memory after using the data, at the expense of
sometimes reading in the same file several times for cross-correlations with other files.  If False
(the default), files read in for a cross-correlation type test will remain in memory until all
possible tests with that file are done.
`output_path`: base directory for output files (default: current directory)
`clobber`: a boolean variable indicating whether to clobber (overwrite) existing files
(default: False)

## Lists of options ##

### Stile expected field names ###

ra, dec: coordinates in RA/dec space, OR
x, y: coordinates in x/y space
g1, g2, sigma: shear and size for objects
g1_psf, g2_psf, sigma_psf: shear and size for the PSF at the location of the object
w: weights

### Extra keywords for file specification ###

More complete descriptions are given above, but for reference:

nickname: nickname for use in output files
wildcard: whether to expand filename strings with glob
fields: field specification for data tables
flag_field: fields used to indicate lines of data which should be removed
file_reader: how to read in the file
group: True, False, or a name of a group of files
bins: a set of binning schemes to slice and dice your data

### Formats ###

Epochs: single, coadd, or multiepoch.  Single and coadd epochs are assumed to have one measurement
per object, while multiepoch files are assumed to contain multiple measurements per object.  (We do
not currently implement any multiepoch tests, and the multiepoch file input has been less
well-tested than the other parts of the code: users who find bugs are encouraged to report them.)
Extents: CCD, field, or survey.  All the files are processed the same way--this is merely so that,
when we have a set of suggested systematics tests, we can distinguish which types of files should
have which systematics tests performed on them.  CCD denotes a single CCD (or something of that
size), field a field of view, and survey a larger area.
Data formats: image or catalog.  We currently implement no image tests, but retain the keyword for
future use.

Users can define their own extents and epochs.  However, with the nested-dict format, they must
appear at the same level of the dict as known Stile epochs or extents so the script can figure out
how to classify the new keys.  (The known Stile extents or epochs can, of course, be empty dicts!)
User-defined epochs will be processed as if they are 'single' or 'coadd' epochs, that is, they will
be assumed to have a single measurement per object.

### Object types ###

star: stars
star PSF, star no PSF: stars used or not used for PSF determination
star bright: some definition of a "bright" star sample
star random: random positions distributed like the stars
galaxy: galaxies
galaxy lens: a set of galaxies to use as lenses (for systematics tests, NOT SCIENCE!)
galaxy random: random positions distributed like the galaxies
galaxy lens random: random positions distributed like the galaxy lenses

# REPORT BUGS TO: https://github.com/msimet/Stile/issues #