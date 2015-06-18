"""sys_test_adapters.py
Contains classes to wrap Stile systematics tests with functions necessary to run the tests via the
HSC/LSST pipeline.
"""
import lsst.pex.config
from lsst.pex.exceptions import LsstCppException
from .. import sys_tests
import numpy

adapter_registry = lsst.pex.config.makeRegistry("Stile test outputs")


# We need to mask the data to particular object types; these pick out the flags we need to do that.
def MaskGalaxy(data, config):
    """
    Given `data`, an LSST source catalog, return a NumPy boolean array describing which rows
    correspond to galaxies.
    """
    # Will have to be more careful/clever about this when classification.extendedness is continuous.
    # These arrays are generally contiguous in memory--so we can just index them like a NumPy
    # recarray.
    try:
        return data['classification.extendedness']==1
    except LsstCppException:
        # But sometimes we've already masked the array--this will work in that case (but is slower
        # than above if the above is possible).
        key = data.schema.find('classification.extendedness').key
        return numpy.array([src[key]==1 for src in data])

def MaskStar(data, config):
    """
    Given `data`, an LSST source catalog, return a NumPy boolean array describing which rows
    correspond to stars.
    """
    try:
        return data['classification.extendedness']==0
    except LsstCppException:
        key = data.schema.find('classification.extendedness').key
        return numpy.array([src[key]==0 for src in data])

def MaskBrightStar(data, config):
    """
    Given `data`, an LSST source catalog, return a NumPy boolean array describing which rows
    correspond to bright stars according to a given S/N cutoff set by
    `config.bright_star_sn_cutoff`.
    """
    star_mask = MaskStar(data, config)
    key_psf = data.schema.find('flux.psf').key
    key_psf_err = data.schema.find('flux.psf.err').key
    bright_mask = (numpy.array([src[key_psf]/src[key_psf_err] for src in data]) >
                   config.bright_star_sn_cutoff)
    return numpy.logical_and(star_mask, bright_mask)

def MaskPSFStar(data, config):
    """
    Given `data`, an LSST source catalog, return a NumPy boolean array describing which rows
    correspond to the stars used to determine the PSF.
    """
    try:
        return data['calib.psf.used']==True
    except LsstCppException:
        try:
            key = data.schema.find('calib.psf.used').key
        except KeyError:
            key = data.schema.find('calib.psf.used.any').key
        return numpy.array([src.get(key)==True for src in data])

# Map the object type strings onto the above functions.
mask_dict = {'galaxy': MaskGalaxy,
             'galaxy lens': MaskGalaxy,  # should do something different here!
             'star': MaskStar,
             'star bright': MaskBrightStar,
             'star PSF': MaskPSFStar}


class BaseSysTestAdapter(object):
    """
    This is an abstract class, implementing a couple of useful masking and column functions for
    reuse in child classes.

    The basic function of a SysTestAdapter is to wrap a Stile SysTest object in a way that makes it
    easy to use with the LSST drivers found in base_tasks.py.  It should always have: an
    attribute `sys_test` that is a SysTest object; an attribute `name` that we can use to generate
    output filenames; a function __call__() that will run the test; a function `getMasks()` that
    returns a set of masks (one for each object type--such as "star" or "galaxy"--that is expected
    for the test) if given a source catalog and config object; and a function getRequiredColumns()
    that returns a list of tuples of required quantities (such as "ra" or "g1"), one tuple
    corresponding to each mask returned from getMasks().

    (More complete lists of the exact expected names for object types and required columns can be
    found in the documentation for the class `Stile.sys_tests.SysTest`.)

    BaseSysTestAdapter makes some of these functions easier.  In particular, it defines:
     - a function setupMasks() that can take a list of strings corresponding to object types and
       generate an attribute, self.mask_funcs, that describes the mask functions which getMasks()
       can then apply to the data to generate masks. Called with no arguments, it will attempt to
       read `self.sys_test.objects_list` for the list of objects (and will raise an error if that
       does not exist).
     - a function getMasks() that will apply the masks in self.mask_funcs to the data.
     - a function getRequiredColumns() that will return the list of required columns from
       self.sys_test.required_quantities if it exists, and raise an error otherwise.
    Of course, any of these can be overridden if desired.
    """
    # As long as we're not actually doing anything with the config object, we can just use the
    # default parent class.  If a real config class is needed, it should be defined separately
    # (inheriting from lsst.pex.config.Config) and the ConfigClass of the SysTestAdapter set to be
    # that class.  (There are examples in previous versions of this file.)
    ConfigClass = lsst.pex.config.Config

    def setupMasks(self, objects_list=None):
        """
        Generate a list of mask functions to match `objects_list`.  If no such list is given, will
        attempt to read the objects_list from self.sys_test, and raise an error if that is not
        found.
        """
        if objects_list==None:
            if hasattr(self.sys_test, 'objects_list'):
                objects_list = self.sys_test.objects_list
            else:
                raise ValueError('No objects_list given, and self.sys_test does not have an '
                                   'attribute objects_list')
        # mask_dict (defined above) maps string object types onto masking functions.
        self.mask_funcs = [mask_dict[obj_type] for obj_type in objects_list]


    def getMasks(self, data, config):
        """
        Given data, a source catalog from the LSST pipeline, return a list of masks.  Each element
        of the list is a mask corresponding to a particular object type, such as "star" or "galaxy."
        @param data  An LSST source catalog.
        @returns     A list of NumPy arrays; each array is made up of Bools that can be broadcast
                     to index the data, returning only the rows that meet the requirements of the
                     mask.
        """
        return [mask_func(data, config) for mask_func in self.mask_funcs]


    def getRequiredColumns(self):
        """
        Return a list of tuples of the specific quantities needed for the test, with each tuple in
        the list matching the data from the corresponding element of the list returned by
        getMasks().  For example, if the masks returned were a star mask and a galaxy mask, and we
        wanted to know the shear signal around galaxies, this should return
        >>> [('ra', 'dec'), ('ra', 'dec', 'g1', 'g2', 'w')]
        since we need to know the positions of the stars and the positions, shears, and weights of
        the galaxies.

        This particular implementation just returns the list of this form from self.sys_test, but
        that choice can be overridden by child classes.

        @returns  A list of tuples, one per mask returned by the method getMasks().  The elements
                  of the tuples are strings corresponding to known quantities from the LSST
                  pipeline.
        """
        return self.sys_test.required_quantities


    def __call__(self, task_config, *data, **kwargs):
        """
        Call this object's sys_test with the given data and kwargs, and return whatever the
        sys_test itself returns.
        """
        return self.sys_test(*data, **kwargs)

class ShapeSysTestAdapter(BaseSysTestAdapter):
    shape_fields = ['g1', 'g2', 'sigma', 'g1_err', 'g2_err', 'sigma_err',
                    'psf_g1', 'psf_g2', 'psf_sigma', 'psf_g1_err',  'psf_g2_err', 'psf_sigma_err']
                    
    def getRequiredColumns(self):
        reqs = self.sys_test.required_quantities
        return_reqs = []
        for req in reqs:
            return_reqs.append([r+'_'+self.shape_type if r in self.shape_fields else r 
                                for r in req])
        return return_reqs

    def fixArray(self, array):
        for field in self.shape_fields:
            if field in array.dtype.names:
                array[field] = array[field+'_'+self.shape_type]
        return array
        
    def __call__(self, task_config, *data, **kwargs):
        """
        Call this object's sys_test with the given data and kwargs, and return whatever the
        sys_test itself returns.
        """
        new_data = [self.fixArray(d) for d in data]
        return self.sys_test(config=task_config.treecorr_kwargs, *new_data, **kwargs)
        
class GalaxyShearAdapter(ShapeSysTestAdapter):
    """
    Adapter for the GalaxyShearSysTest.  See the documentation for that class or
    BaseSysTestAdapter for more information.
    """
    def __init__(self, config):
        self.shape_type = 'sky'
        self.config = config
        self.sys_test = sys_tests.GalaxyShearSysTest()
        self.name = self.sys_test.short_name
        self.setupMasks()
    def __call__(self, task_config, *data, **kwargs):
        """
        Call this object's sys_test with the given data and kwargs, and return whatever the
        sys_test itself returns.
        """
        new_data = [self.fixArray(d) for d in data]
        return self.sys_test(config=task_config.treecorr_kwargs, *data, **kwargs)


class BrightStarShearAdapter(ShapeSysTestAdapter):
    """
    Adapter for the BrightStarShearSysTest.  See the documentation for that class or
    BaseSysTestAdapter for more information.
    """
    def __init__(self, config):
        self.shape_type = 'sky'
        self.config = config
        self.sys_test = sys_tests.BrightStarShearSysTest()
        self.name = self.sys_test.short_name
        self.setupMasks()
    def __call__(self, task_config, *data, **kwargs):
        """
        Call this object's sys_test with the given data and kwargs, and return whatever the
        sys_test itself returns.
        """
        new_data = [self.fixArray(d) for d in data]
        return self.sys_test(task_config.treecorr_kwargs, *data, **kwargs)

class StarXGalaxyShearAdapter(ShapeSysTestAdapter):
    """
    Adapter for the StarXGalaxyShearSysTest.  See the documentation for that class or
    BaseSysTestAdapter for more information.
    """
    def __init__(self, config):
        self.shape_type = 'sky'
        self.config = config
        self.sys_test = sys_tests.StarXGalaxyShearSysTest()
        self.name = self.sys_test.short_name
        self.setupMasks()
    def __call__(self, task_config, *data, **kwargs):
        """
        Call this object's sys_test with the given data and kwargs, and return whatever the
        sys_test itself returns.
        """
        new_data = [self.fixArray(d) for d in data]
        return self.sys_test(config=task_config.treecorr_kwargs, *data, **kwargs)

class StarXStarShearAdapter(ShapeSysTestAdapter):
    """
    Adapter for the StarXStarShearSysTest.  See the documentation for that class or
    BaseSysTestAdapter for more information.
    """
    def __init__(self, config):
        self.shape_type = 'sky'
        self.config = config
        self.sys_test = sys_tests.StarXStarShearSysTest()
        self.name = self.sys_test.short_name
        self.setupMasks()
    def __call__(self, task_config, *data, **kwargs):
        """
        Call this object's sys_test with the given data and kwargs, and return whatever the
        sys_test itself returns.
        """
        new_data = [self.fixArray(d) for d in data]
        return self.sys_test(config=task_config.treecorr_kwargs, *data, **kwargs)

class StarSizeResidualAdapter(ShapeSysTestAdapter):
    """
    Adapter for the StarSizeResidualSysTest.  See the documentation for that class or
    BaseSysTestAdapter for more information.
    """
    def __init__(self, config):
        self.shape_type = 'sky'
        self.config = config
        self.sys_test = sys_tests.StarSizeResidualSysTest()
        self.name = self.sys_test.short_name
        self.setupMasks()

class Rho1Adapter(ShapeSysTestAdapter):
    """
    Adapter for the StarPSFResidXStarPSFResidShearSysTest.  See the documentation for that class or
    BaseSysTestAdapter for more information.
    """
    def __init__(self, config):
        self.shape_type = 'sky'
        self.config = config
        self.sys_test = sys_tests.Rho1SysTest()
        self.name = self.sys_test.short_name
        self.setupMasks()
    def __call__(self, task_config, *data, **kwargs):
        """
        Call this object's sys_test with the given data and kwargs, and return whatever the
        sys_test itself returns.
        """
        new_data = [self.fixArray(d) for d in data]
        return self.sys_test(config=task_config.treecorr_kwargs, *data, **kwargs)

class StatsPSFFluxAdapter(ShapeSysTestAdapter):
    """
    Adapter for the StatSysTest.  See the documentation for that class or BaseSysTestAdapter for
    more information.  In this case, we specifically request 'flux.psf' and object_type 'galaxy'.

    In the future, we plan to have this be more configurable; for now, this works as a test.
    """
    def __init__(self, config):
        self.config = config
        self.sys_test = sys_tests.StatSysTest(field='flux.psf')
        self.name = self.sys_test.short_name+'flux.psf'
#        self.mask_funcs = [mask_dict[obj_type] for obj_type in ['galaxy']]
        self.mask_funcs = [self.MaskPSFFlux]

    def MaskPSFFlux(self, data, config):
        base_mask = mask_dict['galaxy'](data, config)
        try:
            additional_mask = data['flux.psf.flags']==0
        except LsstCppException:
            key = data.schema.find('flux.psf.flags').key
            additional_mask = numpy.array([src.get(key)==False for src in data])
        return numpy.logical_and(base_mask, additional_mask)

    def getRequiredColumns(self):
        return (('flux.psf',),)

    def __call__(self, task_config, *data, **kwargs):
        return self.sys_test(*data, verbose=True, **kwargs)

class WhiskerPlotStarAdapter(ShapeSysTestAdapter):
    def __init__(self,config):
        self.shape_type = 'chip'
        self.config = config
        self.sys_test = sys_tests.WhiskerPlotStarSysTest()
        self.name = self.sys_test.short_name
        self.setupMasks()

    def __call__(self, task_config, *data):
        new_data = [self.fixArray(d) for d in data]
        return self.sys_test(*new_data, linewidth = 0.01, scale = task_config.whiskerplot_scale,
                              figsize = task_config.whiskerplot_figsize,
                              xlim = task_config.whiskerplot_xlim,
                              ylim = task_config.whiskerplot_ylim)

class WhiskerPlotPSFAdapter(ShapeSysTestAdapter):
    def __init__(self,config):
        self.shape_type = 'chip'
        self.config = config
        self.sys_test = sys_tests.WhiskerPlotPSFSysTest()
        self.name = self.sys_test.short_name
        self.setupMasks()

    def __call__(self, task_config, *data):
        new_data = [self.fixArray(d) for d in data]
        return self.sys_test(*new_data, linewidth = 0.01, scale = task_config.whiskerplot_scale,
                              figsize = task_config.whiskerplot_figsize,
                              xlim = task_config.whiskerplot_xlim,
                              ylim = task_config.whiskerplot_ylim)
class WhiskerPlotResidualAdapter(ShapeSysTestAdapter):
    def __init__(self,config):
        self.shape_type = 'chip'
        self.config = config
        self.sys_test = sys_tests.WhiskerPlotResidualSysTest()
        self.name = self.sys_test.short_name
        self.setupMasks()

    def __call__(self, task_config, *data):
        new_data = [self.fixArray(d) for d in data]
        return self.sys_test(*new_data, linewidth = 0.01, scale = task_config.whiskerplot_scale,
                              figsize = task_config.whiskerplot_figsize,
                              xlim = task_config.whiskerplot_xlim,
                              ylim = task_config.whiskerplot_ylim)

class ScatterPlotStarVsPSFG1Adapter(ShapeSysTestAdapter):
    def __init__(self,config):
        self.shape_type = 'sky'
        self.config = config
        self.sys_test = sys_tests.ScatterPlotStarVsPSFG1SysTest()
        self.name = self.sys_test.short_name
        self.setupMasks()

    def __call__(self, task_config, *data, **kwargs):
        try:
            per_ccd_stat = task_config.scatterplot_per_ccd_stat
        except  AttributeError:
            per_ccd_stat = False
        per_ccd_stat = None if per_ccd_stat == 'None' else per_ccd_stat
        new_data = [self.fixArray(d) for d in data]
        return self.sys_test(*new_data, per_ccd_stat = per_ccd_stat)

class ScatterPlotStarVsPSFG2Adapter(ShapeSysTestAdapter):
    def __init__(self,config):
        self.shape_type = 'sky'
        self.config = config
        self.sys_test = sys_tests.ScatterPlotStarVsPSFG2SysTest()
        self.name = self.sys_test.short_name
        self.setupMasks()

    def __call__(self, task_config,*data, **kwargs):
        try:
            per_ccd_stat = task_config.scatterplot_per_ccd_stat
        except  AttributeError:
            per_ccd_stat = False
        per_ccd_stat = None if per_ccd_stat == 'None' else per_ccd_stat
        new_data = [self.fixArray(d) for d in data]
        return self.sys_test(*new_data, per_ccd_stat = per_ccd_stat)

class ScatterPlotStarVsPSFSigmaAdapter(ShapeSysTestAdapter):
    def __init__(self,config):
        self.shape_type = 'sky'
        self.config = config
        self.sys_test = sys_tests.ScatterPlotStarVsPSFSigmaSysTest()
        self.name = self.sys_test.short_name
        self.setupMasks()

    def __call__(self, task_config,*data, **kwargs):
        try:
            per_ccd_stat = task_config.scatterplot_per_ccd_stat
        except  AttributeError:
            per_ccd_stat = False
        per_ccd_stat = None if per_ccd_stat == 'None' else per_ccd_stat
        new_data = [self.fixArray(d) for d in data]
        return self.sys_test(*new_data, per_ccd_stat = per_ccd_stat)

class ScatterPlotResidualVsPSFG1Adapter(ShapeSysTestAdapter):
    def __init__(self,config):
        self.shape_type = 'sky'
        self.config = config
        self.sys_test = sys_tests.ScatterPlotResidualVsPSFG1SysTest()
        self.name = self.sys_test.short_name
        self.setupMasks()

    def __call__(self,task_config,*data, **kwargs):
        try:
            per_ccd_stat = task_config.scatterplot_per_ccd_stat
        except  AttributeError:
            per_ccd_stat = False
        per_ccd_stat = None if per_ccd_stat == 'None' else per_ccd_stat
        new_data = [self.fixArray(d) for d in data]
        return self.sys_test(*new_data, per_ccd_stat = per_ccd_stat)

class ScatterPlotResidualVsPSFG2Adapter(ShapeSysTestAdapter):
    def __init__(self, config):
        self.shape_type = 'sky'
        self.config = config
        self.sys_test = sys_tests.ScatterPlotResidualVsPSFG2SysTest()
        self.name = self.sys_test.short_name
        self.setupMasks()

    def __call__(self,task_config,*data, **kwargs):
        try:
            per_ccd_stat = task_config.scatterplot_per_ccd_stat
        except  AttributeError:
            per_ccd_stat = False
        per_ccd_stat = None if per_ccd_stat == 'None' else per_ccd_stat
        new_data = [self.fixArray(d) for d in data]
        return self.sys_test(*new_data, per_ccd_stat = per_ccd_stat)

class ScatterPlotResidualVsPSFSigmaAdapter(ShapeSysTestAdapter):
    def __init__(self,config):
        self.shape_type = 'sky'
        self.config = config
        self.sys_test = sys_tests.ScatterPlotResidualVsPSFSigmaSysTest()
        self.name = self.sys_test.short_name
        self.setupMasks()

    def __call__(self,task_config,*data, **kwargs):
        try:
            per_ccd_stat = task_config.scatterplot_per_ccd_stat
        except  AttributeError:
            per_ccd_stat = False
        per_ccd_stat = None if per_ccd_stat == 'None' else per_ccd_stat
        new_data = [self.fixArray(d) for d in data]
        return self.sys_test(*new_data, per_ccd_stat = per_ccd_stat)

adapter_registry.register("StatsPSFFlux", StatsPSFFluxAdapter)
adapter_registry.register("GalaxyShear", GalaxyShearAdapter)
adapter_registry.register("BrightStarShear", BrightStarShearAdapter)
adapter_registry.register("StarXGalaxyShear", StarXGalaxyShearAdapter)
adapter_registry.register("StarXStarShear", StarXStarShearAdapter)
adapter_registry.register("StarSizeResidual", StarSizeResidualAdapter)
adapter_registry.register("Rho1", Rho1Adapter)
adapter_registry.register("WhiskerPlotStar", WhiskerPlotStarAdapter)
adapter_registry.register("WhiskerPlotPSF", WhiskerPlotPSFAdapter)
adapter_registry.register("WhiskerPlotResidual", WhiskerPlotResidualAdapter)
adapter_registry.register("ScatterPlotStarVsPSFG1", ScatterPlotStarVsPSFG1Adapter)
adapter_registry.register("ScatterPlotStarVsPSFG2", ScatterPlotStarVsPSFG2Adapter)
adapter_registry.register("ScatterPlotStarVsPSFSigma", ScatterPlotStarVsPSFSigmaAdapter)
adapter_registry.register("ScatterPlotResidualVsPSFG1", ScatterPlotResidualVsPSFG1Adapter)
adapter_registry.register("ScatterPlotResidualVsPSFG2", ScatterPlotResidualVsPSFG2Adapter)
adapter_registry.register("ScatterPlotResidualVsPSFSigma", ScatterPlotResidualVsPSFSigmaAdapter)
