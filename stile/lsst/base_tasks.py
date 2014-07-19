""" base_tasks.py
Contains the Task classes that interface between the LSST/HSC pipeline and the
systematics tests described by Stile.
"""

import lsst.pex.config
import lsst.pipe.base
import lsst.meas.mosaic
from lsst.meas.mosaic.mosaicTask import MosaicTask
from lsst.pipe.tasks.dataIds import PerTractCcdDataIdContainer
from .sys_test_adapters import adapter_registry
import numpy
try:
    import matplotlib
    # We should decide which backend to use (this line allows running matplotlib even on sessions 
    # without properly defined displays, eg through PBS).  This '.use' method needs to be called
    # before the first import of matplotlib.pyplot, and will generally issue a warning if you
    # call it at any other time; since this is the first place it is imported when we use these
    # Tasks, we need to do it here.
    matplotlib.use('Agg') 
    has_matplotlib = True
except ImportError:
    has_matplotlib = False

class SysTestData(object):
    """
    A simple container object holding the name of a sys_test, plus the corresponding masks and
    columns from the getMasks() and getRequiredColumns() methods of the sys_test.
    """
    def __init__(self):
        self.sys_test_name = None
        self.mask_list = None
        self.cols_list = None

class CCDSingleEpochStileConfig(lsst.pex.config.Config):
    # Set the default systematics tests for the CCD level.
    sys_tests = adapter_registry.makeField("tests to run",multi=True,
                    default = ["StatsPSFFlux","StarXGalaxyShear"])
        
class CCDSingleEpochStileTask(lsst.pipe.base.CmdLineTask):
    """
    A basic Task class to run CCD-level single-epoch tests.  Inheriting from 
    lsst.pipe.base.CmdLineTask lets us use the already-built command-line interface for the 
    data ID, rather than reimplementing this ourselves.  Calling 
    >>> CCDSingleEpochStileTask.parseAndRun()
    from within a script will send all the command-line arguments through the argument parser, then
    call the run() method sequentially, once per CCD defined by the input arguments.
    """
    # lsst magic
    ConfigClass = CCDSingleEpochStileConfig
    _DefaultName = "CCDSingleEpochStile"
    
    def __init__(self,**kwargs):
        lsst.pipe.base.CmdLineTask.__init__(self,**kwargs)
        self.sys_tests = self.config.sys_tests.apply()
        
    def run(self,dataRef):
        # Get the data ("catalog"), information on the systematics tests ("sys_data_list"), and
        # the extra computed columns to make sure we have all the required quantities 
        # ("extra_cols_dict").
        catalog, sys_data_list, extra_cols_dict = self.generateColumns(dataRef)
        # Now we have a source catalog, plus a dict of other computed quantities.  Step
        # through the masks and required quantities and generate a NumPy array for each pair, 
        # containing only the required quantities and only in the rows indicated by the mask.
        for sys_test,sys_test_data in zip(self.sys_tests,sys_data_list):
            new_catalogs = []
            for mask,cols in zip(sys_test_data.mask_list,sys_test_data.cols_list):
                new_catalog = {}
                for column in cols:
                    if column in extra_col_dict:
                        new_catalog[column] = extra_col_dict[column][mask]
                    elif column in catalog.schema:
                        try:
                            new_catalog[column] = catalog[column][mask]
                        except:
                            new_catalog[column] = (numpy.array([src[column] 
                                                   for src in catalog])[mask])
                new_catalogs.append(self.makeArray(new_catalog))
            # run the test!
            results = sys_test(*new_catalogs)
            # If there's anything fancy to do with plotting the results, do that.
            if hasattr(sys_test.sys_test,'plot'):
                fig = sys_test.sys_test.plot(results)
                fig.savefig(sys_test_data.sys_test_name+'.png')            
            if has_matplotlib and isinstance(results, matplotlib.figure.Figure):
                results.savefig(sys_test_data.sys_test_name+'.png')    

    def generateColumns(self,dataRef):
        """
        Pull a source catalog from the dataRef, along with any associated data products (such as 
        phometric calibration information).  Then generate required columns which are not already
        in the data array.  
        
        @param dataRef An LSST pipeline dataRef (a subclass of the butler, pointing only to a 
                       specific dataset).
        @returns       A three-element tuple, consisting of:
                         - The source catalog corresponding to the dataRef
                         - A list of SysTestData object, containing information on the columns
                           needed for each test as well as which masks should be applied to the
                           source catalog in order to limit it to the objects we want to analyze
                         - A dict whose keys are the quantities from the systematics tests that
                           were not already in the source catalog, and whose values are NumPy arrays
                           of those quantities with the same length as the source catalog.  Note 
                           that some of the values will be "nan" in these arrays--the values are
                           only computed within the masks that need those quantities.
        """
        # Pull the source catalog from the butler corresponding to the particular CCD in the 
        # dataRef.  
        catalog = dataRef.get("src",immediate=True)
        # Remove objects so badly measured we shouldn't use them in any test
        catalog = self.removeFlaggedObjects(catalog)
        # Get calibration info.  "fcr_md" is the more granular calibration generated by the
        # coaddition routines, while "calexp" is the original calibrated image. The 
        # datasetExists() call will fail if no tract is defined, hence the try-except block.
        try:
            if dataRef.datasetExists("fcr_md"):
                calib_data = dataRef.get("fcr_md")
                calib_type="fcr"
            else:
                calib_data = dataRef.get("calexp_md")
                calib_type = "calexp"
        except:
            calib_data = dataRef.get("calexp_md")
            calib_type = "calexp"
        sys_data_list = []
        extra_col_dict = {}
        # Now, pull the mask and required-quantity info from the individual systematics tests we're
        # going to run.  We'll generate any required quantities only for the data within the 
        # corresponding mask; we'll also check for more specific flags, such as flux measurement
        # failures for tests where we need the flux or magnitude or color of an object, and 
        # fold those into the overall mask.
        for sys_test in self.sys_tests:
            sys_test_data = SysTestData()
            sys_test_data.sys_test_name = sys_test.name
            # Masks expects: a tuple of masks, one for each required data set for the sys_test
            sys_test_data.mask_list = sys_test.getMasks(catalog)
            # cols expects: an iterable of iterables, describing for each required data set
            # the set of extra required columns. len(mask_list) should be equal to len(cols_list).
            sys_test_data.cols_list = sys_test.getRequiredColumns()
            # Generally, we will be updating the masks as we go to take care of the more granular
            # flags such as flux measurement errors.  However, the number of possible flags for
            # shape measurement is so large that checking for the flags each time was prohibitive
            # in terms of run time.  So we will grab the shape mask flags FIRST, for those masks
            # corresponding to data sets where we need shape quantities, and "and" those into the
            # base mask, rather than doing it every time we ask for a shape quantity.
            shape_masks = []
            for cols_list in sys_test_data.cols_list:
                if any([key in cols_list for key in 
                            ['g1','g1_err','g2','g2_err','sigma','sigma_err']]):
                    shape_masks.append(self._computeShapeMask(catalog))
                else:
                    shape_masks.append(True)
            sys_test_data.mask_list = [numpy.logical_and(mask,shape_mask) 
                for mask,shape_mask in zip(sys_test_data.mask_list,shape_masks)]
            # Generate any quantities that aren't already in the source catalog, but can
            # be generated from things that *are* in the source catalog.    
            for (mask,cols) in zip(sys_test_data.mask_list,sys_test_data.cols_list):
                for col in cols:
                    if not col in catalog.schema: # If it exists, don't recompute!
                        if not col in extra_col_dict:
                            # Generate a NumPy array of the right length and fill it with 'nan' to
                            # indicate we haven't computed anything yet.
                            extra_col_dict[col] = numpy.zeros(len(catalog))
                            extra_col_dict[col].fill('nan')
                        nan_mask = numpy.isnan(extra_col_dict[col])
                        # Only do uncomputed values
                        nan_and_col_mask = numpy.logical_and(nan_mask,mask) 
                        if any(nan_and_col_mask>0):
                            # "extra_mask" is the new mask with the quantity-specific flags
                            extra_col_dict[col][nan_and_col_mask], extra_mask = \
                                self.computeExtraColumn(col,catalog[nan_and_col_mask],
                                                        calib_data,calib_type)
                            if extra_mask is not None:
                                mask[nan_and_col_mask] = numpy.logical_and(mask[nan_and_col_mask],
                                                                           extra_mask)
            sys_data_list.append(sys_test_data)
        return catalog, sys_data_list, extra_col_dict

    def removeFlaggedObjects(self,catalog):
        """
        Remove objects which have certain flags we consider unrecoverable failures for weak lensing.
        Currently set to be quite conservative--we may want to relax this in the future. 
        
        @param catalog A source catalog pulled from the LSST pipeline.
        @returns       The source catalog, masked to the rows which don't have any of our defined
                       flags set.
        """
        flags = ['flags.negative', 'deblend.nchild', 'deblend.too-many-peaks',
                 'deblend.parent-too-big', 'deblend.failed', 'deblend.skipped', 
                 'deblend.has.stray.flux', 'flags.badcentroid', 'centroid.sdss.flags', 
                 'centroid.naive.flags', 'flags.pixel.edge', 'flags.pixel.interpolated.any',
                 'flags.pixel.interpolated.center', 'flags.pixel.saturated.any', 
                 'flags.pixel.saturated.center', 'flags.pixel.cr.any', 'flags.pixel.cr.center', 
                 'flags.pixel.bad','flags.pixel.suspect.any','flags.pixel.suspect.center']
        masks = [catalog[flag]==False for flag in flags]
        mask = masks[0]
        for new_mask in masks[1:]:
            mask = numpy.logical_and(mask,new_mask) 
        return catalog[mask]
            
    def makeArray(self,catalog_dict):
        """
        Take a dict whose keys contain NumPy arrays of the same length and turn it into a 
        formatted NumPy array.
        """
        dtypes = []
        # Generate the dtypes.
        for key in catalog_dict:
            dtypes.append((key,catalog_dict[key].dtype))
        len_list = [len(catalog_dict[key]) for key in catalog_dict]
        if not set(len_list)==0:
            raise RuntimeError('Different catalog lengths for different columns!')
        # Make an empty array and fill it column by column.
        data = numpy.zeros(len_list[0],dtype=dtypes)
        for key in catalog_dict:
            data[key] = catalog_dict[key]
        return data
        
    def _computeShapeMask(self,data):
        """
        Compute and return the mask for `data` that excludes pernicious shape measurement failures.
        """
        flags = ['flags.negative', 'deblend.nchild', 'deblend.too-many-peaks',
                 'deblend.parent-too-big', 'deblend.failed', 'deblend.skipped', 
                 'deblend.has.stray.flux', 'flags.badcentroid', 'centroid.sdss.flags', 
                 'centroid.naive.flags', 'flags.pixel.edge', 'flags.pixel.interpolated.any',
                 'flags.pixel.interpolated.center', 'flags.pixel.saturated.any', 
                 'flags.pixel.saturated.center', 'flags.pixel.cr.any', 'flags.pixel.cr.center', 
                 'flags.pixel.bad','flags.pixel.suspect.any','flags.pixel.suspect.center']
        masks = [data[flag]==False for flag in flags]
        mask = masks[0]
        for new_mask in masks[1:]:
            mask = numpy.logical_and(mask,new_mask)    
        return mask
                           
    def computeExtraColumn(self,col,data,calib_data,calib_type):
        """
        Compute the quantity `col` for the given `data`.
        
        @param col        A string indicating the quantity needed
        @param data       A (subset of a) source catalog from the LSST pipeline
        @param calib_data Photometric calibration data for flux/magnitude measurements
        @param calib_type Which type of calibration calib_data is ("fcr" or "calexp"--"fcr" for
                          coadds where available, else "calexp").
        @returns          A 2-element tuple.  The first element is a list or NumPy array of the
                          quantity indicated by `col`. The second is either None (if no further
                          masking is needed) or a NumPy array of boolean values indicating where the
                          quantity is reliable (True) or unusable in some way (False).
                          
        """
        if col=="ra":
            return [src.getRa().asDegrees() for src in data], None
        elif col=="dec":
            return [src.getDec().asDegrees() for src in data], None
        elif col=="mag_err":
            return (2.5/numpy.log(10)*numpy.array([src.getPsfFluxErr()/src.getPsfFlux()
                                                    for src in data])),
                    numpy.array([src.get('flux.psf.flags')==0 & 
                                 src.get('flux.psf.flags.psffactor')==0 for src in data]))
        elif col=="mag":
            # From Steve Bickerton's helpful HSC butler documentation
            if calib_type=="fcr":
                ffp = lsst.meas.mosaic.FluxFitParams(calib_data)
                x, y = data.getX(), data.getY()
                correction = numpy.array([ffp.eval(x[i],y[i]) for i in range(n)])
                zeropoint = 2.5*numpy.log10(fcr.get("FLUXMAG0")) + correction
            elif calib_type=="calexp":
                zeropoint = 2.5*numpy.log10(calib_data.get("FLUXMAG0"))
            return (zeropoint - 2.5*numpy.log10(data.getPsfFlux()),
                    numpy.array([src.get('flux.psf.flags')==0 & 
                                 src.get('flux.psf.flags.psffactor')==0 for src in data]))
        elif col=="g1":
            try:
                # This fails pretty quickly if it isn't going to work, so time-wise better to
                # try it and fail since this branch is much faster to run.
                moments = data.get('shape.sdss')
                ixx = moments.getIxx()
                iyy = moments.getIyy()
            except:
                moments = [src.get('shape.sdss') for src in data]
                ixx = numpy.array([mom.getIxx() for mom in moments])
                iyy = numpy.array([mom.getIxy() for mom in moments])
            # Shape measurement failures are already masked out, so we don't have to check again.
            return ((ixx-iyy)/(ixx+iyy), None)
        elif col=="g2":
            try:
                moments = data.get('shape.sdss')
                ixx = moments.getIxx()
                ixy = moments.getIxy()
                iyy = moments.getIyy()
            except:
                moments = [src.get('shape.sdss') for src in data]
                ixx = numpy.array([mom.getIxx() for mom in moments])
                ixy = numpy.array([mom.getIxy() for mom in moments])
                iyy = numpy.array([mom.getIxy() for mom in moments])
            return (2.*ixy/(ixx+iyy), None)
        elif col=="sigma":
            try:
                moments = data.get('shape.sdss')
                ixx = moments.getIxx()
                iyy = moments.getIyy()
            except:
                ixx = numpy.array([src.get('shape.sdss').getIxx() for src in data])
                iyy = numpy.array([src.get('shape.sdss').getIyy() for src in data])
            return (numpy.sqrt(0.5*(ixx+iyy)), None)
        elif col=="g1_err":
            try:
                moments = data.get('shape.sdss')
                ixx = moments.getIxx()
                iyy = moments.getIyy()
            except:
                ixx = numpy.array([src.get('shape.sdss').getIxx() for src in data])
                iyy = numpy.array([src.get('shape.sdss').getIyy() for src in data])
            try:
                moments_err = data.get('shape.sdss.err')
                cov_ixx = moments_err[0,0]
                cov_iyy = moments_err[1,1]
            except:
                cov_ixx = numpy.array([src.get('shape.sdss.err')[0,0] for src in data])
                cov_iyy = numpy.array([src.get('shape.sdss.err')[1,1] for src in data])
            dg1_dixx = 2.*iyy/(ixx+iyy)**2
            dg1_diyy = -2.*ixx/(ixx+iyy)**2
            return (numpy.sqrt(dg1_dixx**2 * cov_ixx + dg1_diyy**2 * cov_iyy), 
                    None)
        elif col=="g2_err":
            try:
                moments = data.get('shape.sdss')
                ixx = moments.getIxx()
                iyy = moments.getIyy()
                ixy = moments.getIxy()
            except:
                ixx = numpy.array([src.get('shape.sdss').getIxx() for src in data])
                iyy = numpy.array([src.get('shape.sdss').getIyy() for src in data])
                ixy = numpy.array([src.get('shape.sdss').getIxy() for src in data])
            try:
                moments_err = data.get('shape.sdss.err')
                cov_ixx = moments_err[0,0]
                cov_iyy = moments_err[1,1]
                cov_ixy = moments_err[2,2]
            except:
                cov_ixx = numpy.array([src.get('shape.sdss.err')[0,0] for src in data])
                cov_iyy = numpy.array([src.get('shape.sdss.err')[1,1] for src in data])
                cov_ixy = numpy.array([src.get('shape.sdss.err')[2,2] for src in data])
            dg2_dixx = -2.*ixy/(ixx+iyy)**2
            dg2_diyy = -2.*ixy/(ixx+iyy)**2
            dg2_dixy = 2./(ixx+iyy)
            return (numpy.sqrt(dg2_dixx**2 * cov_ixx + dg2_diyy**2 * cov_iyy + 
                               2. * dg2_dixy**2 * cov_ixy), None)
        elif col=="sigma_err":
            try:
                moments = data.get('shape.sdss')
                ixx = moments.getIxx()
                iyy = moments.getIyy()
            except:
                ixx = numpy.array([src.get('shape.sdss').getIxx() for src in data])
                iyy = numpy.array([src.get('shape.sdss').getIyy() for src in data])
            try:
                moments_err = data.get('shape.sdss.err')
                cov_ixx = moments_err[0,0]
                cov_iyy = moments_err[1,1]
            except:
                cov_ixx = numpy.array([src.get('shape.sdss.err')[0,0] for src in data])
                cov_iyy = numpy.array([src.get('shape.sdss.err')[1,1] for src in data])
            sigma = numpy.sqrt(0.5*(ixx+iyy))
            dsigma_dixx = 0.25/sigma
            dsigma_diyy = 0.25/sigma
            return (numpy.sqrt(dsigma_dixx**2 * cov_ixx + dsigma_diyy**2 * cov_iyy), 
                    None)
        elif col=="psf_g1":
            try:
                moments = data.get('shape.sdss.psf')
                ixx = moments.getIxx()
                iyy = moments.getIyy()
            except:
                ixx = numpy.array([src.get('shape.sdss.psf').getIxx() for src in data])
                iyy = numpy.array([src.get('shape.sdss.psf').getIyy() for src in data])
            return ((ixx-iyy)/(ixx+iyy),
                    numpy.array([src.get('shape.sdss.flags.psf')==0 for src in data]))
        elif col=="psf_g2":
            try:
                moments = data.get('shape.sdss.psf')
                ixx = moments.getIxx()
                ixy = moments.getIxy()
                iyy = moments.getIyy()
            except:
                ixx = numpy.array([src.get('shape.sdss.psf').getIxx() for src in data])
                ixy = numpy.array([src.get('shape.sdss.psf').getIxy() for src in data])
                iyy = numpy.array([src.get('shape.sdss.psf').getIyy() for src in data])
            return (2.*ixy/(ixx+iyy),
                    numpy.array([src.get('shape.sdss.flags.psf')==0 for src in data]))
        elif col=="psf_sigma":
            try:
                moments = data.get('shape.sdss.psf')
                ixx = moments.getIxx()
                iyy = moments.getIyy()
            except:
                ixx = numpy.array([src.get('shape.sdss.psf').getIxx() for src in data])
                iyy = numpy.array([src.get('shape.sdss.psf').getIyy() for src in data])
            return (numpy.sqrt(0.5*(ixx+iyy)),
                    numpy.array([src.get('shape.sdss.flags.psf')==0 for src in data]))
        elif col=="w":
            #TODO: better weighting measurement
            return numpy.array([1.]*len(data)), None
        raise NotImplementedError("Cannot compute col %s"%col)
        
    @classmethod
    def _makeArgumentParser(cls):
        # This ContainerClass lets us use the tract-level photometric calibration rather than just
        # using the calibrated exposures (calexp).
        parser = lsst.pipe.base.ArgumentParser(name=cls._DefaultName)
        parser.add_id_argument("--id", "forced_src", help="data ID, with raw CCD keys + tract",
                               ContainerClass=PerTractCcdDataIdContainer)
        return parser

    def writeConfig(self, *args, **kwargs):
        pass
    def writeSchema(self, *args, **kwargs):
        pass
    def writeMetadata(self, dataRef):
        pass

class CCDNoTractSingleEpochStileTask(CCDSingleEpochStileTask):
    """Like CCDSingleEpochStileTask, but we use a different argument parser that doesn't require
    an available coadd to run on the CCD level."""
    _DefaultName = "CCDNoTractSingleEpochStile"
    
    @classmethod
    def _makeArgumentParser(cls):
        parser = lsst.pipe.base.ArgumentParser(name=cls._DefaultName)
        parser.add_id_argument("--id", "src", help="data ID, with raw CCD keys")
        return parser


class StileFieldRunner(lsst.pipe.base.TaskRunner):
    """Subclass of TaskRunner for Stile field tasks.  Most of this code (incl this docstring) 
    pulled from measMosaic.

    FieldSingleEpochStileTask.run() takes a number of arguments, one of which is a list of dataRefs
    extracted from the command line (whereas most CmdLineTasks' run methods take a single dataRef,
    and are called repeatedly).  This class transforms the processed arguments generated by the
    ArgumentParser into the arguments expected by FieldSingleEpochStileTask.run().  It will still 
    call run() once per field if multiple fields are present, but not once per CCD as would
    otherwise occur.

    See pipeBase.TaskRunner for more information.
    """

    @staticmethod
    def getTargetList(parsedCmd, **kwargs):
        # organize data IDs by field
        refListDict = {}
        for ref in parsedCmd.id.refList:
            refListDict.setdefault(ref.dataId["field"], []).append(ref)
        # we call run() once with each field
        return [(field,
                 refListDict[field]
                 ) for field in sorted(refListDict.keys())]

    def __call__(self, args):
        task = self.TaskClass(config=self.config, log=self.log)
        result = task.run(*args)

class FieldSingleEpochStileConfig(lsst.pex.config.Config):
    # Set the default systematics tests for the field level.
    sys_tests = adapter_registry.makeField("tests to run",multi=True,
                    default = ["StatsPSFFlux","StarXGalaxyShear"])

class FieldSingleEpochStileTask(CCDSingleEpochStileTask,MosaicTask):
    """
    A basic Task class to run field-level single-epoch tests.  Inheriting from 
    lsst.pipe.base.CmdLineTask lets us use the already-built command-line interface for the 
    data ID, rather than reimplementing this ourselves.  Calling 
    >>> FieldSingleEpochStileTask.parseAndRun()
    from within a script will send all the command-line arguments through the argument parser, then
    call the run() method sequentially, once per CCD defined by the input arguments.
    
    The "Field" version of this class is different from the "CCD" level of the task in that we will
    have to combine catalogs from each CCD in the visit, and do some trickery with iterables to keep
    everything aligned.
    """
    # lsst magic 
    RunnerClass = StileFieldRunner
    canMultiprocess = False
    ConfigClass = FieldSingleEpochStileConfig
    _DefaultName = "FieldSingleEpochStile"

    def run(self, field, dataRefList):
        catalogs = []
        sys_data_lists = []
        extra_col_dicts = []
        for dataRef in dataRefList:
            catalog, sys_data_list, extra_col_dict = self.generateColumns(dataRef)
            catalogs.append(catalog)
            sys_data_lists.append(sys_data_list)
            extra_col_dicts.append(extra_col_dict)
        # We have a list of lists of sys data lists, where the top level is one per dataRef.
        # Flip the order so the top level is one per sys_test...
        sys_data_lists = [[sys_data_list[i] for sys_data_list in sys_data_lists] 
                          for i in range(len(sys_data_lists[0]))]
        # ....and check for consistency.
        for sys_data_list in sys_data_lists:
            names = [sdl.sys_test_name for sdl in sys_data_list]
            cols = [sdl.cols_list for sdl in sys_data_list]
            # Check length only for the list of masks, since they're a function of the data.
            len_masks = [len(sdl.mask_list) for sdl in sys_data_list]
            if not len(set(names))==1:
                raise RuntimeError('Sys test name is not consistent for all the CCDs in the '
                                   'field!')
            if not len(set(cols))==1:
                raise RuntimeError('Sys test column description is not consistent for all the
                                   'CCDs in the field!')
            if not len(set(len_masks))==1:
                raise RuntimeError('Number of masks for this sys test is not consistent for all '
                                   'the CCDs in the field!')
        # And now, collate things so this looks like the sys_data_list of the CCD version of
        # this class, but with each element of the mask_list being a list with the same length
        # as dataRefList (so one element per dataRef).
        sys_data_list = []
        for sdl in sys_data_lists:
            sys_data_list.append(SysTestData())
            sys_data_list.sys_test_name = sdl[0].sys_test_name
            sys_data_list.cols_list = sdl[0].cols_list
            sys_data_list.mask_list = [ [s.mask_list[i] for s in sdl] 
                                           for i in range(len(sdl.mask_list[0])]
                                       
        for sys_test,sys_data in zip(self.sys_tests,sys_data_list):
            new_catalogs = []
            for mask_list,cols in zip(sys_data.mask_list,sys_data.cols_list):
                new_catalog = {}
                for catalog, extra_col_dict, mask in zip(catalogs, extra_col_dicts, mask_list):
                    for column in cols:
                        if column in extra_col_dict:
                            newcol = extra_col_dict[column][mask]
                        elif column in catalog.schema:
                            try:
                                newcol = catalog[column][mask]
                            except:
                                newcol = numpy.array([src[column] for src in catalog])[mask]
                        if column in new_catalog:
                            new_catalog[column].append(newcol)
                        else:
                            new_catalog[column] = [newcol]
                new_catalogs.append(self.makeArray(new_catalog))
            results = sys_test(*new_catalogs)
            if hasattr(sys_test.sys_test,'plot'):
                fig = sys_test.sys_test.plot(results)
                fig.savefig(sys_test_data.sys_test_name+'.png')            
            if isinstance(results, matplotlib.figure.Figure):
                results.savefig(sys_test_data.sys_test_name+'.png')    
                
    def makeArray(self,catalog_dict):
        dtypes = []
        for key in catalog_dict:
            dtype_list = [cat.dtype for cat in catalog_dict[key]]
            dtype_list.sort()
            dtypes.append((key,dtype_list[-1]))
        len_list = [sum([len(cat) for cat in catalog_dict[key]]) for key in catalog_dict]
        print "len list", len_list, [len(cat) for cat in catalog_dict[key]]
        if not len(set(len_list))==1:
            raise RuntimeError('Different catalog lengths for different columns!')
        data = numpy.zeros(len_list[0],dtype=dtypes)
        for key in catalog_dict:
            current_position = 0
            for catalog in catalog_dict[key]:
                data[key][current_position:current_position+len(catalog)] = catalog
                current_position+=len(catalog)
        return data

