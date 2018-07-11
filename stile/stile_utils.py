"""
stile_utils.py: Various utilities for the Stile pipeline.  Includes input parsing and some
numerical helper functions.
"""

import numpy
try:
    import matplotlib.pyplot as plt
    has_matplotlib = True
except ImportError:
    has_matplotlib = False

def Parser():
    """
    Returns an argparse Parser object with input args used by Stile and TreeCorr.
    """
    from . import treecorr_utils
    import argparse
    p = argparse.Parser(parent=treecorr_utils.Parser())
    #TODO: add, obviously, EVERYTHING ELSE
    return p


def FormatArray(d, fields=None):
    """
    Turn a regular NumPy array of arbitrary types into a formatted array, with optional field name
    description.

    This function uses the existing dtype of the array ``d``.  This means that arrays of
    heterogeneous objects may not return the dtype you expect (for example, ints will be
    converted to floats if there are floats in the array, or all numbers will be converted to
    strings if there are any strings in the array).  Predefining the format or using a function like
    :func:`numpy.genfromtxt` will prevent these issues, as will reading from a FITS file.

    :param d:      A NumPy array.
    :param fields: A dictionary whose keys are the names of the fields you'd like for the output
                   array, and whose values are field numbers (starting with 0) whose names those
                   keys should replace (or, if the array is already formatted, the existing field
                   names the keys should replace); alternately, a list with the same length as the
                   rows of ``d``. [default: None]
    :returns:      A formatted numpy array with the same shape as ``d`` except that the innermost
                   dimension has turned into a record field if it was not already one, optionally
                   with field names appropriately replaced.
    """
    # We want arrays to be numpy.arrays with field access (so we can say d['ra'] or something like
    # that).  In order for these to be created correctly, two conditions have to be met:
    # - The array needs to be initialized with the innermost dimension as a tuple rather than
    #   a list or other array, so NumPy knows that it's a record field;
    # - The array has to be created with a dtype that indicates there are multiple fields.  For
    #   convenience I'm going to do this as a single string of the form '?,?,?[...]' where each
    #   question mark is a single character (plus optional width for strings/voids) denoting what
    #   kind of data to expect. (This is the "array-protocol type string", see
    #   http://docs.scipy.org/doc/numpy/reference/arrays.dtypes.html)
    if not hasattr(d, 'dtype'):
        # If it's not an array, make it one.
        d = numpy.array(d)
    if not d.dtype.names:
        # If it is an array, but doesn't have a "names" attribute, that means it doesn't have
        # records/fields.  So we need to reformat the array.  Given the difficulty of generating
        # an individual dtype for each field, we'll just use the dtype of the overall array for
        # every entry, which involves no casting of types.
        d_shape = d.shape
        if len(d_shape) == 1:  # Assume this was a single row (not a set of 1-column rows)
            d = numpy.array([d])
            d_shape = d.shape
        # Cast this into a 2-d array
        new_d = d.reshape(-1, d_shape[-1])
        # Generate the dtype string
        if isinstance(d.dtype, str):
            dtype = ','.join([d.dtype]*len(d[0]))
        else:
            dtype_char = d.dtype.char
            if dtype_char == 'S' or dtype_char == 'O' or dtype_char == 'V' or dtype_char == 'U':
                dtype = ','.join([d.dtype.str]*len(new_d[0]))  # need the width as well as the char
            else:
                dtype = ','.join([dtype_char]*len(new_d[0]))
        # Make a new array with each row turned into a tuple and the correct dtype
        d = numpy.array([tuple(nd) for nd in new_d], dtype=dtype)
        if len(d_shape) > 1:
            # If this was a more-than-2d array, reshape it back to that original form, minus the
            # dimension we turned into a record (which will no longer appear in the shape).
            d = d.reshape(d_shape[:-1])
    if fields:
        # If the "fields" parameter was set, rewrite the numpy.dtype.names attribute to be the
        # field specification we want.
        if isinstance(fields, dict):
            names = list(d.dtype.names)
            for key in fields:
                names[fields[key]] = key
            d.dtype.names = names
        elif len(fields) == len(d.dtype.names):
            d.dtype.names = fields
        else:
            raise RuntimeError('Cannot use given fields: '+str(fields))
    return d


class Stats(object):
    """A Stats object can carry around and output the statistics of some array.

    Currently it can carry around two types of statistics:

    (1) Basic array statistics: typically one would use length (N), min, max, median, mean, standard
        deviation (stddev), variance, median absolute deviation ('mad') as defined using the
        ``simple_stats`` option at initialization.

    (2) Percentiles: the value at a given percentile level.

    The :class:`StatSysTest <stile.sys_tests.StatSysTest>` class can be used to create and populate
    values for one of these objects.  If you want to change the list of simple statistics, it's
    only necessary to change the code there, not here.
    
    Stats objects can be added together if they contain the same statistics 
    (attribute ``.simple_stats``).  Note that some statistics (length, min, max, mean, stddev,
    variance, skew, kurtosis) can be robustly recreated from summary statistics on subcatalogs, but
    other statistics (median, MAD, and percentiles) require approximations.  In particular:
        - If the catalog ranges overlap (stats1.min<stats2.max and stats1.max>stats2.min), then
          the percentiles, median, and MAD are averages of the subcatalogs weighted by N.
        - If the catalog ranges do NOT overlap, we form a linear interpolation scheme from the given
          percentiles, modified to account for total catalog length (so the 10th percentile becomes
          the 5th percentile when combined with a second catalog of the same length but higher
          data range), and their values.  Then we compute the points corresponding to the requested
          percentiles, including the 50th percentile for the median.  The MAD values are then given
          by:
            MAD_{1+2} = (N_1*(MAD_1+|median_1-median_{1+2}|) + N_2*(MAD_2+|median_2-median{1+2}|))
                            /(N_1+N_2)
          which accounts for, on average, the greater distance to the new median compared to the
          median of the subcatalog.
    Because of the linear interpolation, this scheme will work better with more densely sampled
    percentiles.  Still, the user should not expect these values to be exact, and should
    simulate the performance if robust requirements are needed.
    
    The internal adding routine considers only the statistics mentioned above--any additional
    statistics will need to be computed by hand for the combination of two Stats objects.
    """

    def __init__(self, simple_stats):
        self.simple_stats = simple_stats
        for stat in self.simple_stats:
            init_str = 'self.' + stat + '=None'
            exec(init_str)

        self.percentiles = None
        self.values = None

    def __str__(self):
        """This routine will print the contents of the ``Stats`` object in a nice format.

        We assume that the ``Stats`` object was created by a :class:`StatSysTest`, so that certain
        sanity checks have already been done (e.g., self.percentiles, if not None, is iterable)."""
        # Preamble:
        ret_str = 'Summary statistics:\n'

        # Loop over simple statistics and print them, if not None.  Generically if one is None then
        # all will be, so just check one.
        test_str = "test_val = self."+("%s"%self.simple_stats[0])
        exec(test_str)
        if test_val is not None:
            for stat in self.simple_stats:
                this_string = 'this_val = self.'+stat
                exec(this_string)
                ret_str += '\t%s: %f\n'%(stat, this_val)
            ret_str += '\n'

        # Loop over combinations of percentiles and values, and print them.
        if self.percentiles is not None:
            ret_str += 'Below are lists of (percentile, value) combinations:\n'
            for index in range(len(self.percentiles)):
                ret_str += '\t%f %f\n'%(self.percentiles[index], self.values[index])

        return ret_str

    def __add__(self, new):
        if new.simple_stats != self.simple_stats:
            raise ValueError("To add Stats objects, they must contain the same internal statistics")
        if new.percentiles != self.percentiles:
            raise ValueError("To add Stats objects, they must contain the same percentiles")
        combo = Stats(simple_stats=new.simple_stats)
        combo.min = numpy.min([new.min, self.min])
        combo.max = numpy.max([new.max, self.max])
        combo.N = new.N+self.N
        combo.mean = ((new.N*new.mean+self.N*self.mean)
                                /combo.N)
        combo.variance = (new.N*new.variance + new.N*new.mean**2 +
                                 self.N*self.variance + 
                                 self.N*self.mean**2)/combo.N-combo.mean**2
        combo.stddev = numpy.sqrt(combo.variance) 
        if 'skew' in new.simple_stats:
            self_2mom = self.variance + self.mean**2
            new_2mom = new.variance + new.mean**2
            self_3mom = self.skew*self.variance**1.5 + 3*self.mean*self_2mom - 2*self.mean**3
            new_3mom = new.skew*new.variance**1.5 + 3*new.mean*new_2mom - 2*new.mean**3
            self_4mom = ((self.kurtosis+3)*self.variance**2+4*self.mean*self_3mom
                        -6*self.mean**2*self_2mom+3*self.mean**4)
            new_4mom = ((new.kurtosis+3)*new.variance**2+4*new.mean*new_3mom-6*new.mean**2*new_2mom
                        +3*new.mean**4)
            combo_2mom = combo.variance + combo.mean**2
            combo.skew = ((self.N*self_3mom+new.N*new_3mom)/combo.N 
                            -3*combo.mean*combo_2mom+2*combo.mean**3)/combo.variance**1.5
            combo_3mom = combo.skew*combo.variance**1.5 + 3*combo.mean*combo_2mom - 2*combo.mean**3
            combo.kurtosis = (((self.N*self_4mom+new.N*new_4mom)/combo.N
                               -4*combo.mean*combo_3mom+6*combo.mean**2*combo_2mom-3*combo.mean**4)
                               /(combo.variance**2))-3
        if new.min > self.max or new.max < self.min:
            percentiles = new.percentiles
            if new.min >= self.max:
                self_perc_points = numpy.array(self.percentiles)*1.0*self.N/combo.N
                new_perc_points = 100-((100-numpy.array(new.percentiles))*1.0*new.N/combo.N)
                current_perc_points = numpy.concatenate([self_perc_points, new_perc_points])
                current_perc_values = numpy.concatenate([self.values, new.values])
            elif new.max <= self.min:
                self_perc_points = 100-((100-numpy.array(self.percentiles))*1.0*self.N/combo.N)
                new_perc_points = numpy.array(new.percentiles)*1.0*new.N/combo.N
                current_perc_points = numpy.concatenate([new_perc_points, self_perc_points])
                current_perc_values = numpy.concatenate([new.values, self.values])
            if not numpy.any(current_perc_points==0):
                current_perc_points = numpy.concatenate([[0], current_perc_points])
                current_perc_values = numpy.concatenate([[combo.min], current_perc_values])
            if not numpy.any(current_perc_points==100):
                current_perc_points = numpy.concatenate([current_perc_points, [100]])
                current_perc_values = numpy.concatenate([current_perc_values, [combo.max]])
            locs = numpy.digitize(numpy.concatenate([percentiles, [50]]), current_perc_points)
            vals = (current_perc_values[locs-1] 
                    + (numpy.concatenate([percentiles, [50]]) - current_perc_points[locs-1])
                        /(current_perc_points[locs]-current_perc_points[locs-1])
                    * (current_perc_values[locs]-current_perc_values[locs-1]))
            combo.percentiles = new.percentiles
            combo.values = vals[:-1]
            combo.median = vals[-1]
            combo.mad = (new.N*(new.mad + numpy.abs(new.median-combo.median))+
                         self.N*(self.mad + numpy.abs(self.median-combo.median)))/combo.N
        else:
            combo.percentiles = new.percentiles
            combo.values = (new.N*new.values + self.N*self.values)/combo.N
            combo.median = (new.N*new.median + self.N*self.median)/combo.N
            combo.mad = (new.N*new.mad + self.N*self.mad)/combo.N
        return combo


        
try:
    import astropy.table
    class Table(astropy.table.Table):
        @property
        def shape(self):
            return (len(self), len(self.colnames))
except ImportError:
    pass

fieldNames = {
    'dec': 'the declination of the object',
    'ra': 'the RA of the object',
    'x': 'the x coordinate of the object',
    'y': 'the y coordinate of the object',
    'g1': 'g1, a shear component in the ra direction',
    'g2': 'g2, a shear component 45 degrees from the ra direction',
    'sigma': 'a size parameter for objects with dimension [length] in arbitrary units',
    'psf_g1': 'the g1 of the psf at the location of this object',
    'psf_g2': 'the g2 of the psf at the location of this object',
    'psf_sigma': 'the sigma of the psf at the location of this object',
    'w': 'the weight to apply per object',
    'z': 'the redshift of the object'}

objectNames = {
    'galaxy': 'galaxy data',
    'star': 'star data',
    'galaxy lens': 'galaxies to be used as lenses in galaxy-galaxy lensing',
    'star PSF': 'stars used in PSF determination',
    'star bright': 'especially bright stars',
    'galaxy random': 'random catalog corresponding to the "galaxy" sample',
    'star random': 'random catalog corresponding to the "star" sample'
}


class CorrFuncResult(numpy.ndarray):
    def __new__(self, data, corrfunc_main=None, corrfunc_random=None, corrfunc_gg=None,
                       corrfunc_dd=None, corrfunc_rr=None, corrfunc_dr=None, corrfunc_rd=None,
                       plot_details=None):
        obj = numpy.asarray(data).view(self)
        self.corrfunc_main = corrfunc_main
        self.corrfunc_random = corrfunc_random
        self.corrfunc_gg = corrfunc_gg
        self.corrfunc_dd = corrfunc_dd
        self.corrfunc_rr = corrfunc_rr
        self.corrfunc_dr = corrfunc_dr
        self.corrfunc_rd = corrfunc_rd
        self.plot_details = plot_details
        return obj
    def plot(self, colors=['r', 'b'], log_yscale=False,
                   plot_bmode=True, plot_data_only=True, plot_random_only=True):
        """
        Plot the data returned from a :class:`BaseCorrelationFunctionSysTest` object and stored in
        this object.  This method chooses some sensible defaults, but much of its behavior can be 
        changed.

        :param colors:     A tuple of 2 colors, used for the first and second lines on any given
                           plot
        :param log_yscale: Whether to use a logarithmic y-scale [default: False]
        :param plot_bmode: Whether to plot the b-mode signal, if there is one [default: True]
        :param plot_data_only:   Whether to plot the data-only correlation functions, if present
                                 [default: True]
        :param plot_random_only: Whether to plot the random-only correlation functions, if present
                                 [default: True]
        :returns:          A matplotlib ``Figure`` which may be written to a file with
                           :func:`.savefig()`, if matplotlib can be imported; else None.
        """
        if not has_matplotlib:
            return None
        fields = self.dtype.names
        # Pick which radius measurement to use
        # TreeCorr changed the name of the output columns
        # This catches the case where we added the units to the label
        fields_no_units = [f.split(' [')[0] for f in fields]
        for t_r in ['meanR', 'R_nom', '<R>', 'R_nominal', 'R']:
            if t_r in fields:
                # Protect underscores since they blow up the plotting routines
                r = '\\_'.join(t_r.split('\\'))
                break
            elif t_r in fields_no_units:
                t_i = fields_no_units.index(t_r)
                r = '\\_'.join(fields[t_i].split('\\'))
                break
        else:
            raise ValueError('No radius parameter found in data')

        # Logarithmic x-axes have stupid default ranges: fix this.
        rstep = self[r][1]/self[r][0]
        xlim = [min(self[r])/rstep, max(self[r])*rstep]
        # Check what kind of data is in the array that .plot() received.
        for plot_details in self.plot_details:
            # Pick the one the data contains and use it; break before trying the others.
            if plot_details.t_field in fields:
                pd = plot_details
                break
        else:
            raise ValueError("No valid y-values found in data")
        if log_yscale:
            yscale = 'log'
        else:
            yscale = 'linear'
        fig = plt.figure()
        fig.subplots_adjust(hspace=0)  # no space between stacked plots
        plt.subplots(sharex=True)  # share x-axes
        # Figure out how many plots you'll need--never more than 3, so we just use a stacked column.
        if pd.datarandom_t_field:
            plot_data_only &= pd.datarandom_t_field+'d' in fields
            plot_random_only &= pd.datarandom_t_field+'r' in fields
        if plot_bmode and pd.x_field and pd.t_im_field:
            nrows = 2
        elif pd.datarandom_t_field:
            nrows = 1 + plot_data_only + plot_random_only
        else:
            nrows = 1
        # Plot the first thing
        curr_plot = 0
        ax = fig.add_subplot(nrows, 1, 1)
        ax.axhline(0, alpha=0.7, color='gray')
        ax.errorbar(self[r], self[pd.t_field], yerr=self[pd.sigma_field], color=colors[0],
                    label=pd.t_title)
        if pd.x_title and plot_bmode:
            ax.errorbar(self[r], self[pd.x_field], yerr=self[pd.sigma_field], color=colors[1],
                        label=pd.x_title)
        elif pd.t_im_title:  # Plot y and y_im if not plotting yb (else it goes on a separate plot)
            ax.errorbar(self[r], self[pd.t_im_field], yerr=self[pd.sigma_field], color=colors[1],
                        label=pd.t_im_title)
        ax.set_xscale('log')
        ax.set_yscale(yscale)
        ax.set_xlim(xlim)
        # To prevent too-long decimal y-axis ticklabels that push the label out of frame
        ax.ticklabel_format(axis='y', style='sci', scilimits=(-3,5))
        ax.set_ylabel(pd.y_title)
        ax.legend()
        if pd.x_field and plot_bmode and pd.t_im_field:
            # Both yb and y_im: plot (y, yb) on one plot and (y_im, yb_im) on the other.
            ax = fig.add_subplot(nrows, 1, 2)
            ax.axhline(0, alpha=0.7, color='gray')
            ax.errorbar(self[r], self[pd.t_im_field], yerr=self[pd.sigma_field], color=colors[0],
                        label=pd.t_im_title)
            ax.errorbar(self[r], self[pd.x_im_field], yerr=self[pd.sigma_field], color=colors[1],
                        label=pd.x_im_title)
            ax.set_xscale('log')
            ax.set_yscale(yscale)
            ax.ticklabel_format(axis='y', style='sci', scilimits=(-3,5))
            ax.set_xlim(xlim)
            ax.set_ylabel(pd.y_title)
            ax.legend()
        if plot_data_only and pd.datarandom_t_field:  # Plot the data-only measurements if requested
            curr_plot += 1
            ax = fig.add_subplot(nrows, 1, 2)
            ax.axhline(0, alpha=0.7, color='gray')
            ax.errorbar(self[r], self[pd.datarandom_t_field+'d'], yerr=self[pd.sigma_field],
                        color=colors[0], label=pd.datarandom_t_title+'d}$')
            if plot_bmode and pd.datarandom_x_field:
                ax.errorbar(self[r], self[pd.datarandom_x_field+'d'], yerr=self[pd.sigma_field],
                        color=colors[1], label=pd.datarandom_x_title+'d}$')
            ax.set_xscale('log')
            ax.set_yscale(yscale)
            ax.ticklabel_format(axis='y', style='sci', scilimits=(-3,5))
            ax.set_xlim(xlim)
            ax.set_ylabel(pd.y_title)
            ax.legend()
        # Plot the randoms-only measurements if requested
        if plot_random_only and pd.datarandom_t_field:
            ax = fig.add_subplot(nrows, 1, nrows)
            ax.axhline(0, alpha=0.7, color='gray')
            ax.errorbar(self[r], self[pd.datarandom_t_field+'r'], yerr=self[pd.sigma_field],
                        color=colors[0], label=pd.datarandom_t_title+'r}$')
            if plot_bmode and pd.datarandom_x_field:
                ax.errorbar(self[r], self[pd.datarandom_x_field+'r'], yerr=self[pd.sigma_field],
                        color=colors[1], label=pd.datarandom_x_title+'r}$')
            ax.set_xscale('log')
            ax.set_yscale(yscale)
            ax.ticklabel_format(axis='y', style='sci', scilimits=(-3,5))
            ax.set_xlim(xlim)
            ax.set_ylabel(pd.y_title)
            ax.legend()
        ax.set_xlabel(r)
        plt.tight_layout()
        return fig

if has_matplotlib:
    import matplotlib.figure
    class PlotResult(matplotlib.figure.Figure):
        def __init__(self, plot, data):
            self.data = data
            super(PlotResult, self).__init__(plot)
        
        def getData(self):
            return self.data
else:
    class PlotResult(object):
        pass
