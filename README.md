Stile
=====

Stile: the Systematics Tests In LEnsing pipeline

-------------------------------------
#### Installation instructions ####

Stile is a pure python package, no compilation needed.  We do not currently have a utility that will install it in your system path for you, but as long as this directory is in your $PYTHONPATH, you should be able to import and run.

#### Dependencies ####
To run Stile, you must have:

 - Python 2.7
 - NumPy

We also recommend:

 - [corr2](http://code.google.com/p/mjarvis/), Mike Jarvis's 2-point correlation function code.  All of our correlation function tests involve calls to corr2.  At the moment, corr2 needs to be in your path (i.e. you can call it with a simple "corr2" from the command line).
 - PyFITS/Astropy to handle FITS tables and images.  Stile can run on ASCII tables, but is much slower.
 - matplotlib to generate plots.

More dependencies may be added in the future.  
 
-------------------------------------

#### Current functionality ####

Right now, Stile can:

 - Generate an average shear signal around a set of points (if you've installed corr2), given two catalogs that you specify in a particular way, and plot the results (if you have matplotlib).
 - Perform a number of basic statistics measurements and print the results.
 - Perform any of the above tests for data with different binning schemes applied, using a simple method of specifying the bins.
 
-------------------------------------

#### Wishlist ####

Over the upcoming months, we plan to add:

 - A number of other systematics tests, including whisker plots, scatter plots/trend-fitting, histograms, and a broader array of correlation function tests.
 - A more flexible way of specifying your data set than hard-coding the file names in your script.
 - Automatic drivers to run as many tests as your data set allows.
 - A larger suite of example code.
 - More robust documentation of both the code and the tests.
 - Installation scripts.
 - A module to interface with the LSST/HSC pipeline directly.
 - Tests on images, and the utilities to make those tests easier.
 
