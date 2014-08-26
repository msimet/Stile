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
        return numpy.array([src['classification.extendedness']==1 for src in data])

def MaskStar(data, config):
    """
    Given `data`, an LSST source catalog, return a NumPy boolean array describing which rows
    correspond to stars.
    """
    try:
        return data['classification.extendedness']==0
    except LsstCppException:
        return numpy.array([src['classification.extendedness']==1 for src in data])

def MaskBrightStar(data, config):
    """
    Given `data`, an LSST source catalog, return a NumPy boolean array describing which rows
    correspond to bright stars.  Right now this is set to be the upper 10% of stars in a given
    sample.
    """
    star_mask = MaskStar(data)
    bright_mask = (numpy.array([src['flux.psf']/src['flux.psf.err'] for src in data]) >
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
        return numpy.array([src.get('calib.psf.used')==True for src in data])

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
        >>> [('ra','dec'),('ra','dec','g1','g2','w')]
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

class GalaxyShearAdapter(BaseSysTestAdapter):
    """
    Adapter for the GalaxyShearSysTest.  See the documentation for that class or 
    BaseSysTestAdapter for more information.
    """
    def __init__(self, config):
        self.config = config
        self.sys_test = sys_tests.GalaxyShearSysTest()
        self.name = self.sys_test.short_name
        self.setupMasks()
    def __call__(self, task_config, *data, **kwargs):
        """
        Call this object's sys_test with the given data and kwargs, and return whatever the
        sys_test itself returns.
        """
        return self.sys_test(config=task_config.treecorr_kwargs, *data, **kwargs)

class BrightStarShearAdapter(BaseSysTestAdapter):
    """
    Adapter for the BrightStarShearSysTest.  See the documentation for that class or
    BaseSysTestAdapter for more information.
    """
    def __init__(self, config):
        self.config = config
        self.sys_test = sys_tests.BrightStarShearSysTest()
        self.name = self.sys_test.short_name
        self.setupMasks()
    def __call__(self, task_config, *data, **kwargs):
        """
        Call this object's sys_test with the given data and kwargs, and return whatever the
        sys_test itself returns.
        """
        return self.sys_test(task_config.treecorr_kwargs, *data, **kwargs)

class StarXGalaxyShearAdapter(BaseSysTestAdapter):
    """
    Adapter for the StarXGalaxyShearSysTest.  See the documentation for that class or
    BaseSysTestAdapter for more information.
    """
    def __init__(self, config):
        self.config = config
        self.sys_test = sys_tests.StarXGalaxyShearSysTest()
        self.name = self.sys_test.short_name
        self.setupMasks()
    def __call__(self, task_config, *data, **kwargs):
        """
        Call this object's sys_test with the given data and kwargs, and return whatever the
        sys_test itself returns.
        """
        return self.sys_test(config=task_config.treecorr_kwargs, *data, **kwargs)

class StarXStarShearAdapter(BaseSysTestAdapter):
    """
    Adapter for the StarXStarShearSysTest.  See the documentation for that class or
    BaseSysTestAdapter for more information.
    """
    def __init__(self, config):
        self.config = config
        self.sys_test = sys_tests.StarXStarShearSysTest()
        self.name = self.sys_test.short_name
        self.setupMasks()
    def __call__(self, task_config, *data, **kwargs):
        """
        Call this object's sys_test with the given data and kwargs, and return whatever the
        sys_test itself returns.
        """
        return self.sys_test(config=task_config.treecorr_kwargs, *data, **kwargs)

class StatsPSFFluxAdapter(BaseSysTestAdapter):
    """
    Adapter for the StatSysTest.  See the documentation for that class or BaseSysTestAdapter for
    more information.  In this case, we specifically request 'flux.psf' and object_type 'galaxy'.

    In the future, we plan to have this be more configurable; for now, this works as a test.
    """
    def __init__(self, config):
        self.config = config
        self.sys_test = sys_tests.StatSysTest(field='flux.psf')
        self.name = self.sys_test.short_name+'flux.psf'
        self.mask_funcs = [mask_dict[obj_type] for obj_type in ['galaxy']]

    def getRequiredColumns(self):
        return (('flux.psf',),)

    def __call__(self, task_config, *data, **kwargs):
        return self.sys_test(*data, verbose=True, **kwargs)

adapter_registry.register("StatsPSFFlux", StatsPSFFluxAdapter)
adapter_registry.register("GalaxyShear", GalaxyShearAdapter)
adapter_registry.register("BrightStarShear", BrightStarShearAdapter)
adapter_registry.register("StarXGalaxyShear", StarXGalaxyShearAdapter)
adapter_registry.register("StarXStarShear", StarXStarShearAdapter)
