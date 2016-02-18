""" base_tasks.py
Contains the Task classes that interface between the LSST/HSC pipeline and the systematics tests
described by Stile.  At the moment, we use some functionality that is only available on the HSC
side of the pipeline, but eventually this will be usable for both.
"""

import os
import lsst.pex.config
import lsst.pipe.base
import lsst.meas.mosaic
import lsst.afw.geom as afwGeom
import lsst.afw.image as afwImage
import lsst.afw.table as afwTable
import lsst.afw.cameraGeom as cameraGeom
import lsst.afw.cameraGeom.utils as cameraGeomUtils
from lsst.meas.mosaic.mosaicTask import MosaicTask
from lsst.pipe.tasks.dataIds import PerTractCcdDataIdContainer
from lsst.pipe.tasks.coaddBase import ExistingCoaddDataIdContainer
from lsst.pex.exceptions import LsstCppException
from .sys_test_adapters import adapter_registry
import numpy
import re
import stile

# So we can cut too-long path names.  This assumes that the machine where the code is stored has the
# same settings as the machine where the files will be placed, but I think this is a safe assumption
# for most HSC use cases.
max_path_length = os.pathconf('.', 'PC_NAME_MAX')

parser_description = """
This is a script to run Stile through the LSST/HSC pipeline.

You can configure which systematic tests to run by setting the following option.
From command line, add ::
    -c "sys_tests.names=['TEST_NAME1', 'TEST_NAME2', ...]"

You can also specify this option by writing a file, e.g., ::

    ====================== config.py ======================
    import stile.lsst.base_tasks
    root.sys_tests.names=['TEST_NAME1', 'TEST_NAME2', ...]
    =======================================================

and then adding an option ::

    -C config.py

to the command line.

If you use a file, you can add and remove tests from a default by the following option::
    root.sys_tests.names.add('TEST_NAME')
    root.sys_tests.names.remove('TEST_NAME')
"""


class SysTestData(object):
    """
    A simple container object holding the name of a sys_test, plus the corresponding masks and
    columns from the ``getMasks()`` and ``getRequiredColumns()`` methods of the sys_test.
    """

    def __init__(self):
        self.sys_test_name = None
        self.mask_tuple_list = None
        self.cols_list = None


class CCDSingleEpochStileConfig(lsst.pex.config.Config):
    # Set the default systematics tests for the CCD level.
    sys_tests = adapter_registry.makeField("tests to run", multi=True,
                    default=[#"StatsPSFFlux", #"GalaxyXGalaxyShear", "BrightStarShear",
                             "StarXGalaxyShear", "StarXStarShear", "Rho1",
                             "StarXStarSizeResidual",
                             "WhiskerPlotStar", "WhiskerPlotPSF", "WhiskerPlotResidual",
                             "ScatterPlotStarVsPSFG1", "ScatterPlotStarVsPSFG2",
                             "ScatterPlotStarVsPSFSigma", "ScatterPlotResidualVsPSFG1",
                             "ScatterPlotResidualVsPSFG2", "ScatterPlotResidualVsPSFSigma",
                             "ScatterPlotResidualSigmaVsPSFMag"
                             ])
    treecorr_kwargs = lsst.pex.config.DictField(doc="extra kwargs to control TreeCorr",
                        keytype=str, itemtype=str,
                        default={'ra_units': 'degrees', 'dec_units': 'degrees',
                                 'min_sep': '0.005', 'max_sep': '0.2',
                                 'sep_units': 'degrees', 'nbins': '20'})
    # Generate a list of flag columns to be used in the .removeFlaggedObjects() method
    flags_keep_false = lsst.pex.config.ListField(dtype=str,
        doc="Flags that indicate unrecoverable failures",
        default=['flags.negative', 'deblend.nchild', 'deblend.too-many-peaks',
                 'deblend.parent-too-big', 'deblend.skipped',
                 'deblend.has.stray.flux', 'flags.badcentroid', 'centroid.sdss.flags',
                 'centroid.naive.flags', 'flags.pixel.edge', 'flags.pixel.interpolated.any',
                 'flags.pixel.interpolated.center', 'flags.pixel.saturated.any',
                 'flags.pixel.saturated.center', 'flags.pixel.cr.any', 'flags.pixel.cr.center',
                 'flags.pixel.bad', 'flags.pixel.suspect.any', 'flags.pixel.suspect.center'])
    flags_keep_true = []
    # Generate a list of flag columns to be used in the ._computeShapeMask() method for objects
    # where we have shape information
    do_hsm = lsst.pex.config.Field(dtype=bool, default=False, doc="Use HSM shapes for galaxies?")
    shape_flags = lsst.pex.config.ListField(dtype=str,
        doc="Flags that indicate failures for SDSS-type shape measurements",
        default=['shape.sdss.flags', 'shape.sdss.centroid.flags',
                 'shape.sdss.flags.unweightedbad', 'shape.sdss.flags.unweighted',
                 'shape.sdss.flags.shift', 'shape.sdss.flags.maxiter'])
    # And another list, but for "galaxy" types (where we use HSM regauss shapes)
    shape_flags_hsm = lsst.pex.config.ListField(dtype=str,
        doc="Flags that indicate failures for HSM-type shape measurements",
        default=['shape.hsm.regauss.flags'])
    bright_star_sn_cutoff = 50
    whiskerplot_figsize = lsst.pex.config.ListField(dtype=float,
        doc="figure size for whisker plot", default=[7., 10.])
    whiskerplot_xlim = lsst.pex.config.ListField(dtype=float,
        doc="x limit for whisker plot", default=[-100., 2100.])
    whiskerplot_ylim = lsst.pex.config.ListField(dtype=float,
        doc="y limit for whisker plot", default=[-100., 4200.])
    whiskerplot_scale = lsst.pex.config.Field(dtype=float,
        doc="length of whisker per inch", default=0.4)


class CCDSingleEpochStileTask(lsst.pipe.base.CmdLineTask):
    """
    A basic Task class to run CCD-level single-epoch tests.  Inheriting from
    lsst.pipe.base.CmdLineTask() lets us use the already-built command-line interface for the data
    ID, rather than reimplementing this ourselves.  Calling ::

        CCDSingleEpochStileTask.parseAndRun()

    from within a script will send all the command-line arguments through the argument parser, then
    call the ``run()`` method sequentially, once per CCD defined by the input arguments.
    """
    # lsst magic
    ConfigClass = CCDSingleEpochStileConfig
    _DefaultName = "CCDSingleEpochStile"
    # necessary basic parameters for treecorr to run
    def __init__(self, **kwargs):
        lsst.pipe.base.CmdLineTask.__init__(self, **kwargs)
        self.sys_tests = self.config.sys_tests.apply()
        self.catalog_type = 'src'

    @staticmethod
    def getFilenameBase(dataRef):
        """
        Get the basic strings needed for an output filename in the HSC directory structure.
        """
        # Hironao's dirty fix for getting a directory for saving results and plots
        # and a (visit, ccd) identifier for filename.
        # This part will be updated by Jim on branch "#20".
        # The directory is
        # $SUPRIME_DATA_DIR/rerun/[rerun/name/for/stile]/%(pointing)05d/%(filter)s/stile_output.
        # The filename includes a (visit, ccd) identifier -%(visit)07d-%(ccd)03d.
        src_filename = (dataRef.get("src_filename", immediate=True)[0]).replace('_parent/', '')
        dir = os.path.join(src_filename.split('output')[0], "stile_output")
        if os.path.exists(dir) == False:
            os.makedirs(dir)
        return dir, "-%07d-%03d" % (dataRef.dataId["visit"], dataRef.dataId["ccd"])

    def run(self, dataRef):
        # Pull the source catalog from the butler corresponding to the particular CCD in the
        # dataRef.
        catalog = dataRef.get(self.catalog_type, immediate=True,
                              flags=afwTable.SOURCE_IO_NO_FOOTPRINTS)

        dir, filename_chip = self.getFilenameBase(dataRef)

        # Remove objects so badly measured we shouldn't use them in any test.
        catalog = self.removeFlaggedObjects(catalog)
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
            # Masks expects: a tuple of tuples, with each tuple having a mask name and a mask,
            # and one tuple for each required data set for the sys_test
            sys_test_data.mask_tuple_list = sys_test.getMasks(catalog, self.config)
            # cols expects: an iterable of iterables, describing for each required data set
            # the set of extra required columns. len(mask_tuple_list) should be equal to
            # len(cols_list).
            sys_test_data.cols_list = sys_test.getRequiredColumns()
            # Generally, we will be updating the masks as we go to take care of the more granular
            # flags such as flux measurement errors.  However, the number of possible flags for
            # shape measurement is so large that checking for the flags each time was prohibitive
            # in terms of run time.  So we will grab the shape mask flags FIRST, for those masks
            # corresponding to data sets where we need shape quantities, and "and" those into the
            # base mask, rather than doing it every time we ask for a shape quantity.
            shape_masks = []
            for (mask_type, mask), cols_list in zip(sys_test_data.mask_tuple_list,
                                                    sys_test_data.cols_list):
                if any([key in cols_list for key in
                                ['g1_sky', 'g1_err_sky', 'g2_sky', 'g2_err_sky',
                                 'g1_chip', 'g1_err_chip', 'g2_chip', 'g2_err_chip',
                                 'sigma_sky', 'sigma_chip', 'sigma_err_sky', 'sigma_err_chip',
                                 'w']]):
                    shape_masks.append(self._computeShapeMask(catalog, mask_type))
                else:
                    shape_masks.append(True)
            sys_test_data.mask_tuple_list = [(mask_type, numpy.logical_and(mask, shape_mask))
               for (mask_type, mask), shape_mask in zip(sys_test_data.mask_tuple_list, shape_masks)]
            # Generate any quantities that aren't already in the source catalog, but can
            # be generated from things that *are* in the source catalog.
            for (mask, cols) in zip(sys_test_data.mask_tuple_list, sys_test_data.cols_list):
                self.generateColumns(dataRef, catalog, mask, cols, extra_col_dict)
            sys_data_list.append(sys_test_data)
        # Right now, we have a source catalog, plus a dict of other computed quantities.  Step
        # through the masks and required quantities and generate a NumPy array for each pair,
        # containing only the required quantities and only in the rows indicated by the mask.
        # First we need to add the non-chip or non-sky shape quantity names, if any.
        for sys_data in sys_data_list:
            for cols in sys_data.cols_list:
                for c in cols:
                    if '_sky' in c or '_chip' in c:
                        cols.append('_'.join(c.split('_')[:-1]))
        for sys_test, sys_test_data in zip(self.sys_tests, sys_data_list):
            new_catalogs = []
            for (mask_type, mask), cols in zip(sys_test_data.mask_tuple_list,
                                               sys_test_data.cols_list):
                new_catalog = {}
                for column in cols:
                    if column in extra_col_dict:
                        new_catalog[column] = extra_col_dict[column][mask]
                    elif column in catalog.schema:
                        try:
                            new_catalog[column] = catalog[column][mask]
                        except LsstCppException:
                            new_catalog[column] = (numpy.array([src[column]
                                                   for src in catalog])[mask])
                new_catalogs.append(self.makeArray(new_catalog))
            # run the test!
            results = sys_test(self.config, *new_catalogs)
            # If there's anything fancy to do with the results, do that.
            this_max_path_length = max_path_length-4-len(sys_test_data.sys_test_name)
            if isinstance(results, numpy.ndarray):
                stile.WriteASCIITable(os.path.join(dir,
                      sys_test_data.sys_test_name+filename_chip[:this_max_path_length]+'.dat'),
                      results, print_header=True)
            if hasattr(sys_test.sys_test, 'getData'):
                stile.WriteASCIITable(os.path.join(dir,
                      sys_test_data.sys_test_name+filename_chip[:this_max_path_length]+'.dat'),
                      sys_test.sys_test.getData(), print_header=True)
            if hasattr(sys_test.sys_test, 'plot'):
                fig = sys_test.sys_test.plot(results)
                fig.savefig(os.path.join(dir,
                      sys_test_data.sys_test_name+filename_chip[:this_max_path_length]+'.png'))
            if hasattr(results, 'savefig'):
                results.savefig(os.path.join(dir,
                      sys_test_data.sys_test_name+filename_chip[:this_max_path_length]+'.png'))

    def removeFlaggedObjects(self, catalog):
        """
        Remove objects which have certain flags we consider unrecoverable failures for weak lensing.
        Currently set to be quite conservative--we may want to relax this in the future.  The actual
        flags are set in the config object for this class and can be accessed as the variable
        ``self.config.flags``.  They can be altered on the command line or via configuration files
        as described in the parser help.

        :param catalog: A source catalog pulled from the LSST pipeline.
        :returns:       The source catalog, masked to the rows which don't have any of our defined
                        flags set.
        """
        masks = []
        if self.config.flags_keep_false:
            masks += [catalog[flag] == False for flag in self.config.flags_keep_false]
        if self.config.flags_keep_true:
            masks += [catalog[flag] == True for flag in self.config.flags_keep_true]
        if masks:
            mask = masks[0]
            for new_mask in masks[1:]:
                mask = numpy.logical_and(mask, new_mask)
            catalog = catalog[mask]
        return catalog

    def makeArray(self, catalog_dict):
        """
        Take a dict whose keys contain NumPy arrays of the same length and turn it into a
        formatted NumPy array.
        """
        dtypes = []
        # Generate the dtypes.
        for key in catalog_dict:
            dtypes.append((key, catalog_dict[key].dtype))
        len_list = [len(catalog_dict[key]) for key in catalog_dict]
        if not len(set(len_list)) == 1:
            raise RuntimeError('Different catalog lengths for different columns!')
        # Make an empty array and fill it column by column.
        data = numpy.zeros(len_list[0], dtype=dtypes)
        for key in catalog_dict:
            data[key] = catalog_dict[key]
        return data

    def generateColumns(self, dataRef, catalog, mask_tuple, raw_cols, extra_col_dict):
        """
        Generate required columns which are not already in the data array,  and update
        ``extra_col_dict`` to include them.  Also update the mask (``mask_tuple[1]``) to exclude any
        objects which have specific failures for the requested quantities, such as flux measurement
        failures for flux/magnitude measurements.

        :param dataRef:        A ``dataRef`` that is the source of the following data
        :param catalog:        A source catalog from the LSST pipeline.
        :param mask_tuple:     A 2-item tuple, with the mask name/type in the first element, and the
                               mask indicating which rows to generate columns for in the second
                               element.  ``mask_tuple[1]`` is updated by this function.
        :param cols:           An iterable of strings indicating which quantities are needed.
        :param extra_col_dict: A dict whose keys are quantity names and whose values are
                               NumPy arrays of those quantities.  ``nan`` is used to indicate
                               as-yet uncomputed results; elements which are something other than
                               ``nan`` are not recomputed.  ``extra_col_dict`` is updated by this
                               function.
        """
        cols = list(raw_cols) # so we can pop items and not mess up our column description elsewhere
        # The moments measurement is slow enough to make a difference if we're processing many
        # catalogs, so we do those first, and separately, all at once.  Here, we figure out if there
        # are any moments-related keys to pull out of the catalog.
        base_shape_keys = ['g1', 'g2', 'psf_g1', 'psf_g2', 'w', 'g1_err', 'g2_err', 'psf_g1_err',
                           'psf_g2_err', 'sigma', 'sigma_err', 'psf_sigma', 'psf_sigma_err']
        shape_keys = ['g1_sky', 'g1_chip', 'g2_sky', 'g2_chip', 'sigma_sky', 'sigma_chip']
        shape_err_keys = ['g1_err_sky', 'g1_err_chip', 'g2_err_sky', 'g2_err_chip',
                          'sigma_err_sky', 'sigma_err_chip', 'w']
        psf_shape_keys = ['psf_g1_sky', 'psf_g1_chip', 'psf_g2_sky', 'psf_g2_chip',
                          'psf_sigma_sky', 'psf_sigma_chip']
        psf_shape_err_keys = ['psf_g1_err_sky', 'psf_g1_err_chip', 'psf_g2_err_sky',
                              'psf_g2_err_chip', 'psf_sigma_err_sky', 'psf_sigma_err_chip']
        do_shape = [col for col in shape_keys if col in cols and not col in catalog.schema]
        do_err = [col for col in shape_err_keys if col in cols and not col in catalog.schema]
        do_psf = [col for col in psf_shape_keys if col in cols and not col in catalog.schema]
        do_psf_err = [col for col in psf_shape_err_keys
                      if col in cols and not col in catalog.schema]
        shape_cols = do_shape+do_err+do_psf+do_psf_err  # a list of shape-related columns to process
        # Add the base column names, to make sure there's a column for that in the final array.
        for key in base_shape_keys:
            if key+'_chip' or key+'_sky' in shape_cols:
                shape_cols.append(key)

        calib_type, calib_metadata, calib_metadata_shape = self.getCalibData(dataRef, shape_cols)

        # offset for (x,y) if extra_col_dict has a column 'CCD'. Currently getMm() returns values
        # in pixel. When the pipeline is updated, we should update this line as well.
        if dataRef.dataId.has_key('ccd') and extra_col_dict.has_key(
               'CCD') and ('x' in raw_cols or 'y' in raw_cols):
            xy0 = cameraGeomUtils.findCcd(dataRef.getButler().mapper.camera, cameraGeom.Id(
                dataRef.dataId.get('ccd'))
                                       ).getPositionFromPixel(afwGeom.PointD(0., 0.)).getMm()
        else:
            xy0 = None

        if shape_cols:
            for col in shape_cols:
                if col in cols:
                    cols.remove(col)
                # Make sure there's already a NumPy array in the dict associated with this key.
                # We use "nan" to mark the rows we haven't computed already.
                if col not in extra_col_dict:
                    extra_col_dict[col] = numpy.zeros(len(catalog))
                    extra_col_dict[col].fill('nan')
            # Now, figure out the rows where we need to compute at least one of these quantities
            nan_masks = [numpy.isnan(extra_col_dict[col]) for col in shape_cols]
            nan_mask = numpy.logical_or.reduce(nan_masks)
            nan_and_col_mask = numpy.logical_and(nan_mask, mask_tuple[1])
            # We will have to transform to sky coordinates if the locations are in (ra,dec).  But
            # we may also need the quantities in chip coordinates.
            do_sky_coords = True if numpy.any(['_sky' in col for col in raw_cols]) else False
            do_chip_coords = True if numpy.any(['_chip' in col for col in raw_cols]) else False
            if any(nan_and_col_mask > 0):
                # computeShapes returns a dict of ('key': column) pairs, and sometimes an extra
                # mask indicating where measurements were valid. Here, the extra mask is only used
                # if PSF shapes were computed, since we already computed the shape masking in run().
                for do_quantity, sky_coords in [(do_sky_coords, True), (do_chip_coords, False)]:
                    if do_quantity:
                        shapes_dict, extra_mask = self.computeShapes(catalog[nan_and_col_mask],
                            calib_metadata_shape, do_shape=do_shape, do_err=do_err, do_psf=do_psf,
                            do_psf_err=do_psf_err, sky_coords=sky_coords, mask_type=mask_tuple[0])
                        if extra_mask is not None:
                            mask_tuple[1][nan_and_col_mask] = numpy.logical_and(extra_mask,
                                                                   mask_tuple[1][nan_and_col_mask])
                        for col in shapes_dict:
                            if shapes_dict[col] is not None and col in extra_col_dict:
                                extra_col_dict[col][nan_and_col_mask] = shapes_dict[col]
        # Now we do the other quantities.  A lot of this is similar to the above code.
        for col in cols:
            if not col in catalog.schema:
                if not col in extra_col_dict:
                    extra_col_dict[col] = numpy.zeros(len(catalog))
                    extra_col_dict[col].fill('nan')
                nan_mask = numpy.isnan(extra_col_dict[col])
                nan_and_col_mask = numpy.logical_and(nan_mask, mask_tuple[1])
                if any(nan_and_col_mask > 0):
                    # "extra_mask" is the new mask with the quantity-specific flags
                    extra_col_dict[col][nan_and_col_mask], extra_mask = self.computeExtraColumn(
                        col, catalog[nan_and_col_mask], calib_metadata, calib_type, xy0,
                        mask_type=mask_tuple[0])
                    if extra_mask is not None:
                        mask_tuple[1][nan_and_col_mask] = numpy.logical_and(extra_mask,
                                                                   mask_tuple[1][nan_and_col_mask])

    def getCalibData(self, dataRef, shape_cols):
        # "fcr_md" is the more granular calibration generated by the
        # coaddition routines, while "calexp" is the original calibrated image. The
        # datasetExists() call will fail if no tract is defined, hence the try-except block.
        calib_metadata_shape = None
        try:
            if dataRef.datasetExists("fcr_md", immediate=True):
                calib_metadata = dataRef.get("fcr_md", immediate=True)
                calib_type = "fcr"
                if shape_cols:
                    calib_metadata_shape = dataRef.get("calexp_md", immediate=True)
            else:
                calib_metadata = dataRef.get("calexp_md", immediate=True)
                calib_type = "calexp"
                if shape_cols:
                    calib_metadata_shape = calib_metadata
        except:
            calib_metadata = dataRef.get("calexp_md", immediate=True)
            calib_type = "calexp"
            if shape_cols:
                calib_metadata_shape = calib_metadata

        return calib_type, calib_metadata, calib_metadata_shape

    def _computeShapeMask(self, data, mask_type):
        """
        Compute and return the mask for ``data`` that excludes pernicious shape measurement
        failures.
        """
        masks = list()
        if 'galaxy' in mask_type and self.config.do_hsm:
            for flag in self.config.shape_flags_hsm:
                key = data.schema.find(flag).key
                masks.append(numpy.array([src.get(key) == False for src in data]))
        else:
            for flag in self.config.shape_flags:
                key = data.schema.find(flag).key
                masks.append(numpy.array([src.get(key) == False for src in data]))
        mask = masks[0]
        for new_mask in masks[1:]:
            mask = numpy.logical_and(mask, new_mask)
        return mask

    def computeShapes(self, data, calib, do_shape=True, do_err=True, do_psf=True, do_psf_err=True,
                             sky_coords=True, mask_type=None):
        """
        Compute the shapes for the given ``data``, an LSST source catalog, with the associated
        ``calib`` calibrated exposure metadata (``'calexp'`` or ``'fcr'``, either works).

        :param data:       An LSST source catalog whose shape moments you would like to retrieve.
        :param calib:      The metadata from a calibrated exposure (``'calexp'`` or ``'fcr'``).  If
                           ``sky_coords=False`` (see below) this can be None.
        :param do_shape:   A bool indicating whether to compute ``(g1, g2, sigma)``.
        :param do_err:     A bool indicating whether to compute ``(g1_err, g2_err, sigma_err)``.
        :param do_psf:     A bool indicating whether to compute ``(psf_g1, psf_g2, psf_sigma)``.
        :param do_psf_err: A bool indicating whether to compute ``(psf_g1_err, psf_g2_err,
                           psf_sigma_err)``.
        :param sky_coords: If True, compute the moments in ra, dec coordinates; else compute in
                           native coordinates (x, y for CCD).
        :param mask_type:  The object type corresponding to the data in ``data`` [default: ``None``]
        :returns:          A tuple consisting of:
                              - A dict whose keys are column names (``'g1', 'psf_sigma'``, etc) and
                                whose values are a NumPy array of those quantities
                              - None if no new mask was needed, or a NumPy array of bools indicating
                                which rows had valid measurements.
        """
        if sky_coords:
            wcs = afwImage.makeWcs(calib)
            localLinearTransform = [wcs.linearizePixelToSky(src.getCentroid()).getLinear()
                                for src in data]
        if do_shape or do_err:
            if 'galaxy' in mask_type and self.config.do_hsm:  # For any galaxy type, use shape.hsm
                key_g1 = data.schema.find("shape.hsm.regauss.e1").key
                key_g2 = data.schema.find("shape.hsm.regauss.e2").key
                g1 = numpy.array([src.get(key_g1) for src in data])
                g2 = numpy.array([src.get(key_g2) for src in data])
                if sky_coords:
                    # we do not have size in shape.hsm, but it does not matter for shapes.
                    # The size derived in this code is meaningless though.
                    ixx_pixel = 1.+g1
                    iyy_pixel = 1.-g1
                    ixy_pixel = g2
                    ixx = numpy.array([lt[0,0]**2*ixxp+2.*lt[0,0]*lt[0,1]*ixyp+lt[0,1]**2*iyyp
                            for (ixxp, ixyp, iyyp, lt)
                            in zip(ixx_pixel, ixy_pixel, iyy_pixel, localLinearTransform)])
                    iyy = numpy.array([lt[1,0]**2*ixxp+2.*lt[1,0]*lt[1,1]*ixyp+lt[1,1]**2*iyyp
                            for (ixxp, ixyp, iyyp, lt)
                            in zip(ixx_pixel, ixy_pixel, iyy_pixel, localLinearTransform)])
                    ixy = numpy.array([(lt[0,0]*lt[1,0]*ixxp +
                                     (lt[0,0]*lt[1,1]+lt[0,1]*lt[1,0])*ixyp+
                                     lt[0,1]*lt[1,1]*iyyp)
                            for (ixxp, ixyp, iyyp, lt)
                            in zip(ixx_pixel, ixy_pixel, iyy_pixel, localLinearTransform)])
                else:
                    ixx = 1.+g1
                    iyy = 1.-g1
                    ixy = g2
            else:
                key = data.schema.find("shape.sdss").key
                moments = [src.get(key) for src in data]
                if sky_coords:
                    moments = [moment.transform(lt) for moment, lt in
                                      zip(moments, localLinearTransform)]
                ixx = numpy.array([mom.getIxx() for mom in moments])
                ixy = numpy.array([mom.getIxy() for mom in moments])
                iyy = numpy.array([mom.getIyy() for mom in moments])
        if do_err:
            if 'galaxy' in mask_type and self.config.do_hsm:
                key = data.schema.find('shape.hsm.regauss.sigma').key
                errs = numpy.array([src.get(key) for src in data])
                sigma_errs = numpy.array(errs.shape)
                sigma_errs.fill(1.)
            else:
                key = data.schema.find("shape.sdss.err").key
                covariances = numpy.array([src.get(key) for src in data])
                if sky_coords:
                    cov_ixx = numpy.zeros(covariances[:,0,0].shape)
                    cov_iyy = numpy.zeros(covariances[:,0,0].shape)
                    cov_ixy = numpy.zeros(covariances[:,0,0].shape)
                    for i, (cov, lt) in enumerate(zip(covariances,localLinearTransform)):
                        cov_ixx[i] = (lt[0,0]**4*cov[0,0] +
                                      (2.*lt[0,0]*lt[0,1])**2*cov[2,2] + lt[0,1]**4*cov[1,1])
                        cov_iyy[i] = (lt[1,0]**4*cov[0,0] +
                                      (2.*lt[1,0]*lt[1,1])**2*cov[2,2] + lt[1,1]**4*cov[1,1])
                        cov_ixy[i] = ((lt[0,0]*lt[1,0])**2*cov[0,0] +
                                      (lt[0,0]*lt[1,1]+lt[0,1]*lt[1,0])**2*cov[2,2] +
                                      (lt[0,1]*lt[1,1])**2*cov[1,1])
                else:
                    cov_ixx = covariances[:,0,0]
                    cov_iyy = covariances[:,1,1]
                    cov_ixy = covariances[:,2,2]
        if do_psf:
            key = data.schema.find("shape.sdss.psf").key
            psf_moments = [src.get(key) for src in data]
            if sky_coords:
                psf_moments = [moment.transform(lt) for moment, lt in
                                  zip(psf_moments, localLinearTransform)]
            psf_ixx = numpy.array([mom.getIxx() for mom in psf_moments])
            psf_ixy = numpy.array([mom.getIxy() for mom in psf_moments])
            psf_iyy = numpy.array([mom.getIyy() for mom in psf_moments])

        # Now, combine the moment measurements into the actual quantities we want.
        if do_shape:
            g1 = (ixx-iyy)/(ixx+iyy)
            g2 = 2.*ixy/(ixx+iyy)
            sigma = (ixx*iyy - ixy**2)**0.25
        else:
            g1 = None
            g2 = None
            sigma = None
        if do_err:
            if 'galaxy' in mask_type and self.config.do_hsm:
                g1_err = errs
                g2_err = errs
                sigma_err = sigma_errs
            else:
                dg1_dixx = 2.*iyy/(ixx+iyy)**2
                dg1_diyy = -2.*ixx/(ixx+iyy)**2
                g1_err = numpy.sqrt(dg1_dixx**2 * cov_ixx + dg1_diyy**2 * cov_iyy)
                dg2_dixx = -2.*ixy/(ixx+iyy)**2
                dg2_diyy = -2.*ixy/(ixx+iyy)**2
                dg2_dixy = 2./(ixx+iyy)
                g2_err = numpy.sqrt(dg2_dixx**2 * cov_ixx + dg2_diyy**2 * cov_iyy +
                                    dg2_dixy**2 * cov_ixy)
                dsigma_dixx = 0.25/sigma**3*iyy
                dsigma_diyy = 0.25/sigma**3*ixx
                dsigma_dixy = -0.5/sigma**3*ixy
                sigma_err = numpy.sqrt(dsigma_dixx**2 * cov_ixx + dsigma_diyy**2 * cov_iyy +
                                       dsigma_dixy**2 * cov_ixy)
            w = 1./(0.51**2+g1_err**2+g2_err**2)
        else:
            g1_err = None
            g2_err = None
            sigma_err = None
            w = [1.]*len(data)
        if do_psf:
            psf_g1 = (psf_ixx-psf_iyy)/(psf_ixx+psf_iyy)
            psf_g2 = 2.*psf_ixy/(psf_ixx+psf_iyy)
            psf_sigma = (psf_ixx*psf_iyy - psf_ixy**2)**0.25
            key = data.schema.find('flux.psf.flags').key
            extra_mask = numpy.array([src.get(key) == 0 for src in data])
        else:
            psf_g1 = None
            psf_g2 = None
            psf_sigma = None
            extra_mask = None
        fake_g1 = None if g1 is None else numpy.zeros(len(g1))
        fake_g2 = None if g2 is None else numpy.zeros(len(g2))
        fake_sigma = None if sigma is None else numpy.zeros(len(sigma))
        fake_g1_err = None if g1_err is None else numpy.zeros(len(g1_err))
        fake_g2_err = None if g2_err is None else numpy.zeros(len(g2_err))
        fake_sigma_err = None if sigma_err is None else numpy.zeros(len(sigma_err))
        fake_psf_g1 = None if psf_g1 is None else numpy.zeros(len(psf_g1))
        fake_psf_g2 = None if psf_g2 is None else numpy.zeros(len(psf_g2))
        fake_psf_sigma = None if psf_sigma is None else numpy.zeros(len(psf_sigma))

        if sky_coords:
            # convert degree to arcsec
            if sigma is not None:
                sigma *= 3600.
            if sigma_err is not None:
                sigma_err *= 3600.
            if psf_sigma is not None:
                psf_sigma *= 3600.
            return ({'g1': fake_g1, 'g2': fake_g2, 'g1_err': fake_g1_err, 'g2_err': fake_g2_err,
                     'sigma': fake_sigma, 'psf_g1': fake_psf_g1, 'psf_g2': fake_psf_g2,
                     'psf_sigma': fake_psf_sigma,
                     'g1_sky': g1, 'g2_sky': g2, 'sigma_sky': sigma, 'g1_err_sky': g1_err,
                     'g2_err_sky': g2_err, 'sigma_err_sky': sigma_err, 'w': w,
                     'psf_g1_sky': psf_g1, 'psf_g2_sky': psf_g2, 'psf_sigma_sky': psf_sigma},
                     extra_mask)
        else:
            return ({'g1': fake_g1, 'g2': fake_g2, 'g1_err': fake_g1_err, 'g2_err': fake_g2_err,
                     'sigma': fake_sigma, 'psf_g1': fake_psf_g1, 'psf_g2': fake_psf_g2,
                     'psf_sigma': fake_psf_sigma,
                     'g1_chip': g1, 'g2_chip': g2, 'sigma_chip': sigma, 'g1_err_chip': g1_err,
                     'g2_err_chip': g2_err, 'sigma_err_chip': sigma_err, 'w': w,
                     'psf_g1_chip': psf_g1, 'psf_g2_chip': psf_g2, 'psf_sigma_chip': psf_sigma},
                     extra_mask)

    def computeExtraColumn(self, col, data, calib_data, calib_type, xy0=None, mask_type=None):
        """
        Compute the quantity ``col`` for the given ``data``.

        :param col:        A string indicating the quantity needed.
        :param data:       A (subset of a) source catalog from the LSST pipeline.
        :param calib_data: Photometric calibration data for flux/magnitude measurements.
        :param calib_type: Which type of calibration calib_data is ("fcr" or "calexp"--"fcr" for
                           coadds where available, else "calexp").
        :param xy0:        Offset of a CCD. [default: None, meaning do not add any offset]
        :param mask_type:  The object type corresponding to the data in ``data`` [default: None]
        :returns:          A 2-element tuple.  The first element is a list or NumPy array of the
                           quantity indicated by ``col``. The second is either None (if no further
                           masking is needed) or a NumPy array of boolean values indicating where
                           the quantity is reliable (True) or unusable in some way (False).

        """
        if col == "ra":
            return [src.getRa().asDegrees() for src in data], None
        elif col == "dec":
            return [src.getDec().asDegrees() for src in data], None
        elif col == "x":
            if xy0:
                return [src.getX() + xy0.getX() for src in data], None
            else:
                return [src.getX() for src in data], None
        elif col == "y":
            if xy0:
                return [src.getY() + xy0.getY() for src in data], None
            else:
                return [src.getY() for src in data], None
        elif col == "mag_err":
            key = data.schema.find('flux.psf.flags').key
            return (2.5/numpy.log(10)*numpy.array([src.getPsfFluxErr()/src.getPsfFlux()
                                                    for src in data]),
                    numpy.array([src.get(key) == 0 for src in data]))
        elif col == "mag":
            # From Steve Bickerton's helpful HSC butler documentation
            if calib_type == "fcr":
                ffp = lsst.meas.mosaic.FluxFitParams(calib_data)
                x = [src.getX() for src in data]
                y = [src.getY() for src in data]
                correction = numpy.array([ffp.eval(x[i], y[i]) for i in range(len(x))])
                zeropoint = 2.5*numpy.log10(calib_data.get("FLUXMAG0")) + correction
            elif calib_type == "calexp":
                zeropoint = 2.5*numpy.log10(calib_data.get("FLUXMAG0"))
            key = data.schema.find('flux.psf.flags').key
            return (zeropoint - 2.5*numpy.log10(numpy.array([src.getPsfFlux() for src in data])),
                    numpy.array([src.get(key) == 0 for src in data]))
        elif col == "mag_inst":
            key = data.schema.find('flux.psf.flags').key
            return (-2.5*numpy.log10(numpy.array([src.getPsfFlux() for src in data])),
                    numpy.array([src.get(key) == 0 for src in data]))
        elif col == "w":
            # Use uniform weights for now if we don't use shapes ("w" will be removed from the
            # list of columns if shapes are computed).
            return numpy.array([1.]*len(data)), None
        raise NotImplementedError("Cannot compute field %s" % col)

    @classmethod
    def _makeArgumentParser(cls):
        # This ContainerClass lets us use the tract-level photometric calibration rather than just
        # using the calibrated exposures (calexp).
        parser = lsst.pipe.base.ArgumentParser(name=cls._DefaultName)
        parser.add_id_argument("--id", "forced_src", help="data ID, with raw CCD keys + tract",
                               ContainerClass=PerTractCcdDataIdContainer)
        parser.description = parser_description
        return parser

    def writeConfig(self, *args, **kwargs):
        pass
    def writeSchema(self, *args, **kwargs):
        pass
    def writeMetadata(self, dataRef):
        pass

class CCDNoTractSingleEpochStileTask(CCDSingleEpochStileTask):
    """Like :class:`CCDSingleEpochStileTask`, but we use a different argument parser that doesn't
    require an available coadd to run on the CCD level."""
    _DefaultName = "CCDNoTractSingleEpochStile"

    @classmethod
    def _makeArgumentParser(cls):
        parser = lsst.pipe.base.ArgumentParser(name=cls._DefaultName)
        parser.add_id_argument("--id", "src", help="data ID, with raw CCD keys")
        parser.description = parser_description
        return parser


class StileVisitRunner(lsst.pipe.base.TaskRunner):
    """Subclass of ``TaskRunner`` for Stile visit tasks.  Most of this code (incl this docstring)
    pulled from ``measMosaic``.

    ``VisitSingleEpochStileTask.run()`` takes a number of arguments, one of which is a list of
    dataRefs extracted from the command line (whereas most ``CmdLineTask``s' run methods take a
    single dataRef, and are called repeatedly).  This class transforms the processed arguments
    generated by the ArgumentParser into the arguments expected by
    ``VisitSingleEpochStileTask.run()``.  It will still call ``run()`` once per visit if multiple
    visits are present, but not once per CCD as would otherwise occur.

    See ``pipeBase.TaskRunner`` for more information.
    """

    @staticmethod
    def getTargetList(parsedCmd, **kwargs):
        # organize data IDs by visit
        refListDict = {}
        for ref in parsedCmd.id.refList:
            refListDict.setdefault(ref.dataId["visit"], []).append(ref)
        # we call run() once with each visit
        return [(visit,
                 refListDict[visit]
                 ) for visit in sorted(refListDict.keys())]

    def __call__(self, args):
        task = self.TaskClass(config=self.config, log=self.log)
        result = task.run(*args)


class VisitSingleEpochStileConfig(CCDSingleEpochStileConfig):
    # Set the default systematics tests for the visit level.  Some keys (eg "flags", "shape_flags")
    # inherited from CCDSingleEpochStileConfig.
    sys_tests = adapter_registry.makeField("tests to run", multi=True,
                    default=[#"StatsPSFFlux", #"GalaxyXGalaxyShear", "BrightStarShear",
                             "StarXGalaxyShear", "StarXStarShear", "Rho1",
                             "StarXStarSizeResidual",
                             "WhiskerPlotStar", "WhiskerPlotPSF", "WhiskerPlotResidual",
                             "ScatterPlotStarVsPSFG1", "ScatterPlotStarVsPSFG2",
                             "ScatterPlotStarVsPSFSigma", "ScatterPlotResidualVsPSFG1",
                             "ScatterPlotResidualVsPSFG2", "ScatterPlotResidualVsPSFSigma",
                             "ScatterPlotResidualSigmaVsPSFMag"
                             ])
    treecorr_kwargs = lsst.pex.config.DictField(doc="extra kwargs to control treecorr",
                        keytype=str, itemtype=str,
                        default={'ra_units': 'degrees', 'dec_units': 'degrees',
                                 'min_sep': '0.05', 'max_sep': '1',
                                 'sep_units': 'degrees', 'nbins': '20'})
    whiskerplot_figsize = lsst.pex.config.ListField(dtype=float,
        doc="figure size for whisker plot", default=[12., 10.])
    whiskerplot_xlim = lsst.pex.config.ListField(dtype=float,
        doc="x limit for whisker plot", default=[-20000., 20000.])
    whiskerplot_ylim = lsst.pex.config.ListField(dtype=float,
        doc="y limit for whisker plot", default=[-20000., 20000.])
    whiskerplot_scale = lsst.pex.config.Field(dtype=float,
        doc="length of whisker per inch", default=0.4)
    scatterplot_per_ccd_stat = lsst.pex.config.Field(dtype=str, default='median',
                         doc="Which statistics (median, mean, or None) to be performed in CCDs.")
    ccd_type = 'S7'

class VisitSingleEpochStileTask(CCDSingleEpochStileTask):
    """
    A basic Task class to run visit-level single-epoch tests.  Inheriting from
    ``lsst.pipe.base.CmdLineTask`` lets us use the already-built command-line interface for the
    data ID, rather than reimplementing this ourselves.  Calling ::

        VisitSingleEpochStileTask.parseAndRun()

    from within a script will send all the command-line arguments through the argument parser, then
    call the ``run()`` method sequentially, once per CCD defined by the input arguments.

    The "Visit" version of this class is different from the "CCD" level of the task in that we will
    have to combine catalogs from each CCD in the visit, and do some trickery with iterables to keep
    everything aligned.
    """
    # lsst magic
    RunnerClass = StileVisitRunner
    canMultiprocess = False
    ConfigClass = VisitSingleEpochStileConfig
    _DefaultName = "VisitSingleEpochStile"
    item_type = 'ccd'
    multi_item_type = None

    @staticmethod
    def getFilenameBase(dataRefList):
        """
        Get the basic strings needed for an output filename in the HSC directory structure.
        """
        # Hironao's dirty fix for getting a directory for saving results and plots
        # and a (visit, chip) identifier for filename.
        # This part will be updated by Jim on branch "#20".
        # The directory is
        # $SUPRIME_DATA_DIR/rerun/[rerun/name/for/stile]/%(pointing)05d/%(filter)s/stile_output.
        # The filename has a visit identifier -%(visit)07d-[ccds], where [ccds] is a reduced form
        # of a CCD list, e.g., if a CCD list is [0, 1, 2, 4, 5, 8, 10],
        # [ccds] becomes 0..2^4..5^8^10.
        src_filename = (dataRefList[0].get("src_filename",
                                           immediate=True)[0]).replace('_parent/', '')
        dir = os.path.join(src_filename.split('output')[0], "stile_output")
        if os.path.exists(dir) == False:
            os.makedirs(dir)
        ccds = [dataRef.dataId['ccd'] for dataRef in dataRefList]
        ccds.sort()
        ccd_str = "%03d" % ccds[0]
        for i, ccd in enumerate(ccds[1:]):
            if not(ccd - 1 in ccds) and (ccd+1 in ccds):
                ccd_str += "^%03d" % ccd
            elif (ccd - 1 in ccds) and not(ccd+1 in ccds):
                ccd_str += "..%03d" % ccd
            elif not(ccd - 1 in ccds) and not(ccd+1 in ccds):
                ccd_str += "^%03d" % ccd
        return dir, "-%07d-%s" % (dataRefList[0].dataId["visit"], ccd_str)

    def run(self, visit, dataRefList):
        # It seems like it would make more sense to put all of this in a separate function and run
        # it once per catalog, then collate the results at the end (just before running the test).
        # Turns out that, compared to the current implementation, that takes 2-3 times as long to
        # run (!) even before you get to the collation step.  So, we duplicate some code here in
        # the name of runtime, at the expense of some complexity in terms of nested lists of things.
        # Some of this code is annotated more clearly in the CCD* version of this class.

        # temporary fix for a patch that exists in dataRefList but not in catalogs.
        catalogs = list()
        for dataRef in dataRefList:
            try:
                catalogs.append(dataRef.get(self.catalog_type, immediate=True,
                                flags=afwTable.SOURCE_IO_NO_FOOTPRINTS))
            except RuntimeError as e:
                print e, ', skip this patch'
        catalogs = [self.removeFlaggedObjects(catalog) for catalog in catalogs]
        sys_data_list = []
        extra_col_dicts = [{} for catalog in catalogs]

        dir, filename_chips = self.getFilenameBase(dataRefList)

        # Some tests need to know which data came from which CCD
        for dataRef, catalog, extra_col_dict in zip(dataRefList, catalogs, extra_col_dicts):
            extra_col_dict['CCD'] = numpy.zeros(len(catalog), dtype=self.config.ccd_type)
            if self.multi_item_type:
                extra_col_dict['CCD'].fill(str(dataRef.dataId[self.multi_item_type])+'_'+
                                           str(dataRef.dataId[self.item_type]))
            else:
                extra_col_dict['CCD'].fill(dataRef.dataId[self.item_type])
        for sys_test in self.sys_tests:
            sys_test_data = SysTestData()
            sys_test_data.sys_test_name = sys_test.name
            # Masks expects: a tuple of tuples, with each tuple having a mask name and a mask,
            # and one tuple for each required data set for the sys_test
            temp_mask_tuple_list = [sys_test.getMasks(catalog, self.config) for catalog in catalogs]
            # cols expects: an iterable of iterables, describing for each required data set
            # the set of extra required columns.
            sys_test_data.cols_list = sys_test.getRequiredColumns()
            if any([key in c for cols_list in sys_test_data.cols_list for c in cols_list
                    for key in ['g1', 'g2', 'sigma']]):
                shape_masks = [[self._computeShapeMask(catalog, mask_type=mask[0])
                                for mask in mask_tuple_list]
                                for catalog, mask_tuple_list in zip(catalogs, temp_mask_tuple_list)]
            else:
                shape_masks = [[[True]*len(catalog) for mask in mask_tuple_list]
                               for catalog, mask_tuple_list in zip(catalogs, temp_mask_tuple_list)]
            # Now, temp_mask_tuple_list and shape_mask is ordered such that there is one list per
            # catalog, and the list for each catalog iterates through the sys tests.  But we
            # actually want one list per sys test, and the list for each sys test to iterate through
            # the catalogs!  So this next line A) combines the new shape masks with the old flag
            # masks, and B) switches the nesting of the mask list to be the ordering we expect.
            sys_test_data.mask_tuple_list = [[(mask_tuple_list[i][0],
                                           numpy.logical_and(mask_tuple_list[i][1],
                                           shape_mask_list[i]))
                    for mask_tuple_list, shape_mask_list in zip(temp_mask_tuple_list, shape_masks)]
                    for i in range(len(temp_mask_tuple_list[0]))]
            for (mask_tuple_list, cols) in zip(sys_test_data.mask_tuple_list,
                                               sys_test_data.cols_list):
                for dataRef, mask, catalog, extra_col_dict in zip(dataRefList, mask_tuple_list,
                                                                  catalogs, extra_col_dicts):
                    self.generateColumns(dataRef, catalog, mask, cols, extra_col_dict)
            # Some tests need to know which data came from which CCD, so we add a column for that
            # here to make sure it's propagated through to the sys_tests.
            sys_test_data.cols_list = [list(cols)+['CCD'] for cols in sys_test_data.cols_list]
            sys_data_list.append(sys_test_data)
        for sys_data in sys_data_list:
            for cols in sys_data.cols_list:
                for c in cols:
                    if '_sky' in c or '_chip' in c:
                        cols.append('_'.join(c.split('_')[:-1]))
        for sys_test, sys_test_data in zip(self.sys_tests, sys_data_list):
            new_catalogs = []
            for mask_tuple_list, cols in zip(sys_test_data.mask_tuple_list,
                                             sys_test_data.cols_list):
                new_catalog = {}
                for column in cols:
                    for catalog, extra_col_dict, (mask_type, mask) in zip(catalogs, extra_col_dicts,
                                                                          mask_tuple_list):
                        if column in extra_col_dict:
                            newcol = extra_col_dict[column][mask]
                        elif column in catalog.schema:
                            key = catalog.schema.find(column).key
                            try:
                                newcol = catalog.get(key)[mask]
                            except LsstCppException:
                                newcol = numpy.array([src.get(key) for src in catalog])[mask]
                        # The new_catalog dict has values which are lists of the quantity we want,
                        # one per dataRef.
                        if column in new_catalog:
                            new_catalog[column].append(newcol)
                        else:
                            new_catalog[column] = [newcol]
                new_catalogs.append(self.makeArray(new_catalog))
            results = sys_test(self.config, *new_catalogs)
            this_max_path_length = max_path_length-4-len(sys_test_data.sys_test_name)
            if isinstance(results, numpy.ndarray):
                stile.WriteASCIITable(os.path.join(dir,
                      sys_test_data.sys_test_name+filename_chips[:this_max_path_length]+'.dat'),
                      results, print_header=True)
            if hasattr(sys_test.sys_test, 'getData'):
                stile.WriteASCIITable(os.path.join(dir,
                      sys_test_data.sys_test_name+filename_chips[:this_max_path_length]+'.dat'),
                      sys_test.sys_test.getData(), print_header=True)
            if hasattr(results, 'savefig'):
                results.savefig(os.path.join(dir,
                      sys_test_data.sys_test_name+filename_chips[:this_max_path_length]+'.png'))
            fig = sys_test.sys_test.plot(results)
            fig.savefig(os.path.join(dir,
                      sys_test_data.sys_test_name+filename_chips[:this_max_path_length]+'.png'))

    def makeArray(self, catalog_dict):
        """
        Take a dict whose keys contain lists of NumPy arrays which will concatenate to the same
        length and turn it into a single formatted NumPy array.
        """
        dtypes = []
        for key in catalog_dict:
            # It's possible (though unlikely...?) that the different catalogs will generate columns
            # with different precision.  So here we exploit the fact that numpy dtypes (at least for
            # numbers) order more or less by precision.
            dtype_list = [cat.dtype for cat in catalog_dict[key]]
            dtype_list.sort()
            dtypes.append((key, dtype_list[-1]))
        # Make sure that when we concatenate the arrays from different catalogs, we'l get the same
        # length for every column in catalog_dict.
        len_list = [sum([len(cat) for cat in catalog_dict[key]]) for key in catalog_dict]
        if not len(set(len_list)) == 1:
            raise RuntimeError('Different catalog lengths for different columns!')
        # Then make a blank array and fill it with the values from the arrays.
        data = numpy.zeros(len_list[0], dtype=dtypes)
        for key in catalog_dict:
            current_position = 0
            for catalog in catalog_dict[key]:
                data[key][current_position:current_position+len(catalog)] = catalog
                current_position += len(catalog)
        return data


class VisitNoTractSingleEpochStileTask(VisitSingleEpochStileTask):
    """Like :class:`VisitSingleEpochStileTask`, but we use a different argument parser that doesn't
    require an available coadd to run on the CCD level."""
    _DefaultName = "VisitNoTractSingleEpochStile"

    @classmethod
    def _makeArgumentParser(cls):
        parser = lsst.pipe.base.ArgumentParser(name=cls._DefaultName)
        parser.add_id_argument("--id", "src", help="data ID, with raw CCD keys")
        parser.description = parser_description
        return parser

class StileMultiVisitRunner(lsst.pipe.base.TaskRunner):
    """Subclass of ``TaskRunner`` for Stile multiple-visit tasks.  Most of this code (incl this
    docstring) pulled from measMosaic.

    ``VisitSingleEpochStileTask.run()`` takes a number of arguments, one of which is a list of
    dataRefs extracted from the command line (whereas most ``CmdLineTask``s' run methods take a
    single dataRef, and are called repeatedly).  This class transforms the processed arguments
    generated by the ArgumentParser into the arguments expected by
    ``VisitSingleEpochStileTask.run()``.  It will still call ``run()`` once per visit if multiple
    visits are present, but not once per CCD as would otherwise occur.

    See ``pipeBase.TaskRunner`` for more information.
    """

    @staticmethod
    def getTargetList(parsedCmd, **kwargs):
        return [(None, parsedCmd.id.refList)]

    def __call__(self, args):
        task = self.TaskClass(config=self.config, log=self.log)
        result = task.run(*args)

class MultiVisitSingleEpochStileTask(VisitSingleEpochStileTask):
    """
    A basic Task class to run visit-level single-epoch tests.  Inheriting from
    ``lsst.pipe.base.CmdLineTask`` lets us use the already-built command-line interface for the
    data ID, rather than reimplementing this ourselves.  Calling ::
        VisitSingleEpochStileTask.parseAndRun()
    from within a script will send all the command-line arguments through the argument parser, then
    call the ``run()`` method sequentially, once per CCD defined by the input arguments.

    The "Visit" version of this class is different from the "CCD" level of the task in that we will
    have to combine catalogs from each CCD in the visit, and do some trickery with iterables to keep
    everything aligned.
    """
    # lsst magic
    RunnerClass = StileMultiVisitRunner
    _DefaultName = "MultiVisitSingleEpochStile"
    multi_item_type = 'visit'

    @staticmethod
    def getFilenameBase(dataRefList):
        """
        Get the basic strings needed for an output filename in the HSC directory structure.
        """
        # Hironao's dirty fix for getting a directory for saving results and plots
        # and a (visit, chip) identifier for filename.
        # This part will be updated by Jim on branch "#20".
        # The directory is
        # $SUPRIME_DATA_DIR/rerun/[rerun/name/for/stile]/%(pointing)05d/%(filter)s/stile_output.
        # The filename has a visit identifier -%(visit)07d-[ccds], where [ccds] is a reduced form
        # of a CCD list, e.g., if a CCD list is [0, 1, 2, 4, 5, 8, 10],
        # [ccds] becomes 0..2^4..5^8^10.
        src_filename = (dataRefList[0].get("src_filename",
                                           immediate=True)[0]).replace('_parent/', '')
        dir = os.path.join(src_filename.split('output')[0], "stile_output")
        if os.path.exists(dir) == False:
            os.makedirs(dir)
        visits = [dataRef.dataId['visit'] for dataRef in dataRefList]
        visits = list(set(visits))
        visits.sort()
        file_string_list = []
        for visit in visits:
            this_visit = [dataRef for dataRef in dataRefList
                    if dataRef.dataId['visit'] == visit]
            _, ccd_str = VisitSingleEpochStileTask.getFilenameBase(this_visit)
            _, _, ccds = ccd_str.split('-')
            file_string_list.append((visit, ccds))
        file_string_list_sorted = [['%07d'%file_string_list[0][0], file_string_list[0][1]]]
        for i, (visit, ccds) in enumerate(file_string_list[1:]):
            if (not (visit - 1 in visits) or not (ccds == file_string_list_sorted[-1])):
                file_string_list_sorted.append([visit, ccds])
            # This is: if this is another visit in a contiguous series of them, but it's either the
            # last one, or the next one has a different set of CCDs. (We know that
            # file_string_list is in order, because visits is sorted.)
            elif (visit-1 in visits) and (not (visit+1 in visits) or
                  (visit+1 in visits and not ccds == file_string_list[i+2][1])):
                file_string_list_sorted[-1][0] = file_string_list_sorted[-1][0]+"..%03d"%visit
        file_string = "-%s-%s" % tuple(file_string_list_sorted[0])
        for file_string_list in file_string_list_sorted[1:]:
            file_string = file_string + "-%s-%s" % tuple(file_string_list)
        return dir, file_string


class PatchSingleEpochStileConfig(CCDSingleEpochStileConfig):
    sys_tests = adapter_registry.makeField("tests to run", multi=True,
                    default=["StatsPSFFlux", #"GalaxyXGalaxyShear", "BrightStarShear",
                             "StarXGalaxyShear", "StarXStarShear", "Rho1",
                             "StarXStarSizeResidual",
                             "WhiskerPlotStar", "WhiskerPlotPSF", "WhiskerPlotResidual",
                             "ScatterPlotStarVsPSFG1", "ScatterPlotStarVsPSFG2",
                             "ScatterPlotStarVsPSFSigma", "ScatterPlotResidualVsPSFG1",
                             "ScatterPlotResidualVsPSFG2", "ScatterPlotResidualVsPSFSigma",
                             "ScatterPlotResidualSigmaVsPSFMag"
                             ])
    do_hsm = lsst.pex.config.Field(dtype=bool, default=True, doc="Use HSM shapes for galaxies?")
    treecorr_kwargs = lsst.pex.config.DictField(doc="extra kwargs to control TreeCorr",
                        keytype=str, itemtype=str,
                        default={'ra_units': 'degrees', 'dec_units': 'degrees',
                                 'min_sep': '0.0005', 'max_sep': '0.02',
                                 'sep_units': 'degrees', 'nbins': '20'})
    whiskerplot_figsize = lsst.pex.config.ListField(dtype=float,
        doc="figure size for whisker plot", default=[7., 5.])
    whiskerplot_xlim = lsst.pex.config.ListField(dtype=float,
        doc="x limit for whisker plot", default=[None, None])
    whiskerplot_ylim = lsst.pex.config.ListField(dtype=float,
        doc="y limit for whisker plot", default=[None, None])
    whiskerplot_scale = lsst.pex.config.Field(dtype=float,
        doc="length of whisker per inch", default=0.4)
    flags_keep_true = ['detect.is-primary']
    coaddName = lsst.pex.config.Field(
        doc="coadd name: typically one of deep or goodSeeing",
        dtype=str,
        default="deep",
    )
    coadd_catalog_type = lsst.pex.config.Field(
        doc="coadd catalog type: deepCoadd_meas or deepCoadd_src",
        dtype=str,
        default="deepCoadd_meas",
    )
    # Generate a list of flag columns to be used in the .removeFlaggedObjects() method
    flags_keep_false = lsst.pex.config.ListField(dtype=str,
        doc="Flags that indicate unrecoverable failures",
        default=['flags.negative', 'deblend.nchild', 'deblend.too-many-peaks',
                 'deblend.parent-too-big', 'deblend.skipped',
                 'deblend.has.stray.flux', 'flags.badcentroid', 'centroid.sdss.flags',
                 'centroid.naive.flags', 'flags.pixel.edge', 'flags.pixel.interpolated.any',
                 'flags.pixel.interpolated.center', 'flags.pixel.saturated.any',
                 'flags.pixel.saturated.center', 'flags.pixel.cr.any', 'flags.pixel.cr.center',
                 'flags.pixel.bad', 'flags.pixel.suspect.any', 'flags.pixel.suspect.center',
                 'flags.pixel.clipped.any'])

class PatchSingleEpochStileTask(CCDSingleEpochStileTask):
    """Like :class:`CCDSingleEpochStileTask`, but for use on single coadd patches instead of
    single CCDs."""
    _DefaultName = "PatchSingleEpochStile"
    ConfigClass = PatchSingleEpochStileConfig

    def __init__(self, **kwargs):
        lsst.pipe.base.CmdLineTask.__init__(self, **kwargs)
        self.sys_tests = self.config.sys_tests.apply()
        self.catalog_type = self.config.coadd_catalog_type

    @staticmethod
    def getFilenameBase(dataRef):
        """
        Get the basic strings needed for an output filename in the HSC directory structure.
        """
        # Hironao's dirty fix for getting a directory for saving results and plots
        # and a (tract,patch) identifier for filename.
        # This part will be updated by Jim on branch "#20".
        # The directory is
        # $SUPRIME_DATA_DIR/rerun/[rerun/name/for/stile]/%(pointing)05d/%(filter)s/stile_output.
        # The filename includes a (tract,patch) identifier -%(tract)07d-%(patch)s.
        src_filename = (dataRef.get("deepCoadd_src_filename", immediate=True)[0]
                       ).replace('_parent/', '')
        t_filename = re.split('(HSC-.)', src_filename)[:2]
        t_filename.append('stile_output')
        dir = os.path.join(*t_filename)
        if os.path.exists(dir) == False:
            os.makedirs(dir)
        return dir, "-%07d-%s" % (dataRef.dataId["tract"], dataRef.dataId["patch"])

    @classmethod
    def _makeArgumentParser(cls):
        parser = lsst.pipe.base.ArgumentParser(name=cls._DefaultName)
        parser.add_id_argument("--id", "deepCoadd_calexp", help="data ID, with patch/tract information",
                               ContainerClass=ExistingCoaddDataIdContainer)
        parser.description = parser_description
        return parser

    def getCalibData(self, dataRef, shape_cols):
        calib_metadata_shape = None
        calib_metadata = dataRef.get("deepCoadd_calexp_md", immediate=True)
        calib_type = "calexp"  # This is just so computeShapes knows the format
        if shape_cols:
            calib_metadata_shape = calib_metadata

        return calib_type, calib_metadata, calib_metadata_shape


class TractSingleEpochStileConfig(CCDSingleEpochStileConfig):
    sys_tests = adapter_registry.makeField("tests to run", multi=True,
                    default=[#"StatsPSFFlux", #"GalaxyXGalaxyShear", "BrightStarShear",
                             "StarXGalaxyShear", "StarXStarShear", "Rho1",
                             "StarXStarSizeResidual",
                             "WhiskerPlotStar", "WhiskerPlotPSF", "WhiskerPlotResidual",
                             "ScatterPlotStarVsPSFG1", "ScatterPlotStarVsPSFG2",
                             "ScatterPlotStarVsPSFSigma", "ScatterPlotResidualVsPSFG1",
                             "ScatterPlotResidualVsPSFG2", "ScatterPlotResidualVsPSFSigma",
                             "ScatterPlotResidualSigmaVsPSFMag"
                             ])
    do_hsm = lsst.pex.config.Field(dtype=bool, default=True, doc="Use HSM shapes for galaxies?")
    treecorr_kwargs = lsst.pex.config.DictField(doc="extra kwargs to control treecorr",
                        keytype=str, itemtype=str,
                        default={'ra_units': 'degrees', 'dec_units': 'degrees',
                                 'min_sep': '0.05', 'max_sep': '1',
                                 'sep_units': 'degrees', 'nbins': '20'})
    flags_keep_true = ['detect.is-primary']
    coaddName = lsst.pex.config.Field(
        doc="coadd name: typically one of deep or goodSeeing",
        dtype=str,
        default="deep",
    )
    coadd_catalog_type = lsst.pex.config.Field(
        doc="coadd catalog type: deepCoadd_meas or deepCoadd_src",
        dtype=str,
        default="deepCoadd_meas",
    )
    whiskerplot_figsize = lsst.pex.config.ListField(dtype=float,
        doc="figure size for whisker plot", default=[12., 10.])
    whiskerplot_xlim = lsst.pex.config.ListField(dtype=float,
        doc="x limit for whisker plot", default=[None, None])
    whiskerplot_ylim = lsst.pex.config.ListField(dtype=float,
        doc="y limit for whisker plot", default=[None, None])
    whiskerplot_scale = lsst.pex.config.Field(dtype=float,
        doc="length of whisker per inch", default=0.4)
    # Generate a list of flag columns to be used in the .removeFlaggedObjects() method
    flags_keep_false = lsst.pex.config.ListField(dtype=str,
        doc="Flags that indicate unrecoverable failures",
        default=['flags.negative', 'deblend.nchild', 'deblend.too-many-peaks',
                 'deblend.parent-too-big', 'deblend.skipped',
                 'deblend.has.stray.flux', 'flags.badcentroid', 'centroid.sdss.flags',
                 'centroid.naive.flags', 'flags.pixel.edge', 'flags.pixel.interpolated.any',
                 'flags.pixel.interpolated.center', 'flags.pixel.saturated.any',
                 'flags.pixel.saturated.center', 'flags.pixel.cr.any', 'flags.pixel.cr.center',
                 'flags.pixel.bad', 'flags.pixel.suspect.any', 'flags.pixel.suspect.center',
                 'flags.pixel.clipped.any'])

    ccd_type = 'S7'  # NumPy string dtype, 7 characters long

class StileTractRunner(lsst.pipe.base.TaskRunner):
    """Subclass of ``TaskRunner`` for Stile tract tasks.  Most of this code (incl this docstring)
    pulled from ``measMosaic``.

    ``TractSingleEpochStileTask.run()`` takes a number of arguments, one of which is a list of
    dataRefs extracted from the command line (whereas most CmdLineTasks' run methods take a single
    dataRef, and are called repeatedly).  This class transforms the processed arguments generated
    by the ArgumentParser into the arguments expected by ``TractSingleEpochStileTask.run()``.  It
    will still call ``run()`` once per visit if multiple visits are present, but not once per patch
    as would otherwise occur.

    See ``pipeBase.TaskRunner`` for more information.
    """

    @staticmethod
    def getTargetList(parsedCmd, **kwargs):
        # organize data IDs by tract
        refListDict = {}
        for ref in parsedCmd.id.refList:
            refListDict.setdefault(ref.dataId["tract"], []).append(ref)
        # we call run() once with each visit
        return [(tract,
                 refListDict[tract]
                 ) for tract in sorted(refListDict.keys())]

    def __call__(self, args):
        task = self.TaskClass(config=self.config, log=self.log)
        result = task.run(*args)

class TractSingleEpochStileTask(VisitSingleEpochStileTask):
    """Like :class:`VisitSingleEpochStileTask`, but with individual elements being patches instead
    of CCDs.  Since the code layout is different, we inherit from Visit, and then make the changes
    to the filename and calibration data information as we did for the Patch-level Task.

    The patch names, which are strings, are sent to the various SysTests as a fake 'CCD'
    column in the data array (in analogy to the real 'CCD' column created by
    ``VisitSingleEpochStileTask``), so we can use the existing architecture to split by patch
    instead of CCD."""
    RunnerClass = StileTractRunner
    _DefaultName = "TractSingleEpochStile"
    ConfigClass = TractSingleEpochStileConfig
    item_type = 'patch'  # Key to populate a fake 'CCD' column for the tests that will use it

    def __init__(self, **kwargs):
        lsst.pipe.base.CmdLineTask.__init__(self, **kwargs)
        self.sys_tests = self.config.sys_tests.apply()
        self.catalog_type = self.config.coadd_catalog_type

    @staticmethod
    def getFilenameBase(dataRefList):
        """
        Get the basic strings needed for an output filename in the HSC directory structure.
        """
        # Hironao + Melanie's dirty fix for getting a directory for saving results and plots
        # and a (tract, patch) identifier for filename.
        # This part will be updated by Jim on branch "#20".
        # The directory is
        # $SUPRIME_DATA_DIR/rerun/[rerun/name/for/stile]/%(tract)07d/%(filter)s/stile_output.
        # The filename includes a (tract, patch) identifier -%(tract)07d-%(patch)s.
        src_filename = (dataRefList[0].get("deepCoadd_src_filename", immediate=True)[0]
                       ).replace('_parent/', '')
        t_filename = re.split('(HSC-.)', src_filename)[:2]
        t_filename.append('stile_output')
        dir = os.path.join(*t_filename)
        if os.path.exists(dir) == False:
            os.makedirs(dir)

        patches = [dataRef.dataId['patch'].split(',') for dataRef in dataRefList]
        patches = [(int(p[0]), int(p[1])) for p in patches]
        patches.sort()
        min_x = min([p[0] for p in patches])
        max_x = max([p[0] for p in patches])
        min_y = min([p[1] for p in patches])
        max_y = max([p[1] for p in patches])
        patches_str = ''
        while patches:
            curr_patch = patches[0]
            j_x = 0
            j_y = 0  # To guard against cases where the next loops don't happen
            for j_x in range(0, max_x-min_x+2):  # +2 to make sure we overshoot for later -1
                if (curr_patch[0]+j_x, curr_patch[1]) not in patches:
                    break
            j_x -= 1
            for j_y in range(0, max_y-min_y+2):
                if not numpy.all(numpy.array([(curr_patch[0]+i_x, curr_patch[1]+j_y) in patches
                                          for i_x in range(j_x+1)])):
                    break
            j_y -= 1
            if j_x > 0 or j_y > 0:
                if patches_str:
                    patches_str += '^'
                patches_str += '(%i,%i)..(%i,%i)' % (curr_patch[0], curr_patch[1],
                                                     curr_patch[0]+j_x, curr_patch[1]+j_y)
                for i_x in range(j_x+1):
                    for i_y in range(j_y+1):
                        patches.remove((curr_patch[0]+i_x, curr_patch[1]+i_y))
            else:
                if patches_str:
                    patches_str += '^'
                patches_str += '(%i,%i)' % curr_patch
                patches.remove(curr_patch)
        return dir, "-%07d-%s" % (dataRefList[0].dataId["tract"], patches_str)

    @classmethod
    def _makeArgumentParser(cls):
        parser = lsst.pipe.base.ArgumentParser(name=cls._DefaultName)
        parser.add_id_argument("--id", "deepCoadd_calexp", help="data ID, with patch/tract information",
                               ContainerClass=ExistingCoaddDataIdContainer)
        parser.description = parser_description
        return parser

    def getCalibData(self, dataRef, shape_cols):
        calib_metadata_shape = None
        calib_metadata = dataRef.get("deepCoadd_calexp_md", immediate=True)
        calib_type = "calexp"  # This is just so computeShapes knows the format
        if shape_cols:
            calib_metadata_shape = calib_metadata

        return calib_type, calib_metadata, calib_metadata_shape


class StileMultiTractRunner(lsst.pipe.base.TaskRunner):
    """Subclass of ``TaskRunner`` for Stile multi-tract tasks.  Most of this code (incl this
    docstring) pulled from ``measMosaic``.

    ``MultiTractSingleEpochStileTask.run()`` takes a number of arguments, one of which is a list of
    dataRefs extracted from the command line (whereas most CmdLineTasks' run methods take a single
    dataRef, and are called repeatedly).  This class transforms the processed arguments generated by
    the ArgumentParser into the arguments expected by ``TractSingleEpochStileTask.run()``.  It will
    only call ``run()`` once per command line, regardless of how many tracts are passed.

    See ``pipeBase.TaskRunner`` for more information.
    """
    @staticmethod
    def getTargetList(parsedCmd, **kwargs):
        return [(None, parsedCmd.id.refList)]

    def __call__(self, args):
        task = self.TaskClass(config=self.config, log=self.log)
        result = task.run(*args)


class MultiTractSingleEpochStileTask(TractSingleEpochStileTask):
    """Like :class:`TractSingleEpochStileTask`, but analyzes multiple tracts per call instead of
    just one."""
    RunnerClass = StileMultiTractRunner
    _DefaultName = "MultiTractSingleEpochStile"
    ConfigClass = TractSingleEpochStileConfig
    multi_item_type = 'tract'
    item_type = 'patch'

    @staticmethod
    def getFilenameBase(dataRefList):
        """
        Get the basic strings needed for an output filename in the HSC directory structure.
        """
        # Combine the strings output by the TractSingleEpochStileTask into one filename.
        src_filename = (dataRefList[0].get("deepCoadd_src_filename", immediate=True)[0]
                       ).replace('_parent/', '')
        t_filename = re.split('(HSC-.)', src_filename)[:2]
        t_filename.append('stile_output')
        dir = os.path.join(*t_filename)
        if os.path.exists(dir) == False:
            os.makedirs(dir)

        tracts = [dataRef.dataId['tract'] for dataRef in dataRefList]
        tracts = list(set(tracts))
        tracts.sort()
        file_string_list = []
        for tract in tracts:
            this_tract = [dataRef for dataRef in dataRefList
                          if dataRef.dataId['tract'] == tract]
            _, patch_str = TractSingleEpochStileTask.getFilenameBase(this_tract)
            _, _, patches = patch_str.split('-')
            file_string_list.append((tract, patches))
        file_string_list_sorted = [['%07d'%file_string_list[0][0], file_string_list[0][1]]]
        for i, (tract, patches) in enumerate(file_string_list[1:]):
            if (not (tract - 1 in tracts) or not (patches == file_string_list_sorted[-1])):
                file_string_list_sorted.append([tract, patches])
            # This is: if this is another tract in a contiguous series of them, but it's either the
            # last one, or the next one has a different set of patches. (We know that
            # file_string_list is in order, because tracts is sorted.)
            elif (tract-1 in tracts) and (not (tract+1 in tracts) or
                  (tract+1 in tracts and not patches == file_string_list[i+2][1])):
                file_string_list_sorted[-1][0] = file_string_list_sorted[-1][0]+"..%03d"%tract
        file_string = "-%s-%s" % tuple(file_string_list_sorted[0])
        for file_string_list in file_string_list_sorted[1:]:
            # TODO: should these be carats?
            file_string = file_string + "-%s-%s" % tuple(file_string_list)
        return dir, file_string

    @classmethod
    def _makeArgumentParser(cls):
        parser = lsst.pipe.base.ArgumentParser(name=cls._DefaultName)
        parser.add_id_argument("--id", "deepCoadd_calexp", help="data ID, with patch/tract information",
                               ContainerClass=ExistingCoaddDataIdContainer)
        parser.description = parser_description
        return parser
