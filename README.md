Stile
=====

Stile: the Systematics Tests In LEnsing pipeline

-------------------------------------
#### Installation instructions ####

Stile is a pure python package, no compilation needed.  You can install it using:
> python setup.py install

#### Dependencies ####
To run Stile, you must have:

 - Python 2.7
 - NumPy

We also recommend:

 - [TreeCorr](http://github.com/rmjarvis/TreeCorr), Mike Jarvis's 2-point correlation function code.  All of our correlation function tests involve calls to this package. 
 - PyFITS/Astropy to handle FITS tables and images.  Stile can run on ASCII tables, but is much slower.
 - matplotlib to generate plots.
 - scipy to calculate some point statistics

More dependencies may be added in the future.
 
-------------------------------------
#### Documentation ####

The documentation is available online at http://stile.readthedocs.io/.  You can also build it using Sphinx in the `doc/` directory.
 
-------------------------------------

#### Current functionality ####

Right now, Stile can:

 - Generate an average shear signal around a set of points (if you've installed corr2), given two catalogs that you specify in a particular way, and plot the results (if you have matplotlib).
 - Perform a number of basic statistics measurements and print the results.
 - Make whisker plots.
 - Make scatter plots with trendlines.
 - Generate histograms.
 - Perform any of the above tests for data with different binning schemes applied, using a simple method of specifying the bins.
 - Interface with sufficiently recent versions of the HSC pipeline.
 
-------------------------------------

#### Wishlist ####

Over the upcoming months, we plan to add:

 - A more flexible way of specifying your data set than hard-coding the file names in your script.
 - Automatic drivers to run as many tests as your data set allows.
 - A larger suite of example code.
 - Tests on images, and the utilities to make those tests easier.
 
