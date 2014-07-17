import lsst.pex.config
import lsst.pipe.base
import lsst.meas.mosaic
from lsst.meas.mosaic import MosaicTask
from lsst.pipe.tasks.dataIds import PerTractCcdDataIdContainer
from .sys_test_adapters import adapter_registry
import numpy
try:
    import matplotlib
    # We should decide which backend to use (this line allows running matplotlib even on sessions 
    # without properly defined displays, eg through PBS)
    matplotlib.use('Agg') 
    import matplotlib.pyplot as plt
    has_matplotlib = True
except ImportError:
    has_matplotlib = False


class CCDSingleEpochStileConfig(lsst.pex.config.Config):
    sys_tests = adapter_registry.makeField("tests to run",multi=True,
                    default = ["StatsPSFFlux","StarXGalaxyShear"])
    
class SysTestData(object):
    def __init__(self):
        self.test_name = None
        self.mask_list = None
        self.cols_list = None
    
class CCDSingleEpochStileTask(lsst.pipe.base.CmdLineTask):
    ConfigClass = CCDSingleEpochStileConfig
    required_columns = None
    _DefaultName = "CCDSingleEpochStile"
    
    def __init__(self,**kwargs):
        lsst.pipe.base.CmdLineTask.__init__(self,**kwargs)
        self.sys_tests = self.config.sys_tests.apply()
        
    def run(self,dataRef):
        catalog = dataRef.get("src",immediate=True)
        catalog = self.removeFlags(catalog)
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
        for sys_test in self.sys_tests:
            sys_test_data = SysTestData()
            sys_test_data.test_name = sys_test.name
            # Masks expects: a tuple of masks, one for each required data set for the sys_test
            sys_test_data.mask_list = sys_test.getMasks(catalog)
            # cols expects: an iterable of iterables, describing for each required data set
            # the set of extra required columns.
            sys_test_data.cols_list = sys_test.getRequiredColumns()
            for (mask,cols) in zip(sys_test_data.mask_list,sys_test_data.cols_list):
                generateColumns(self,mask,cols,catalog,calib_data,calib_type,extra_col_dict)
            sys_data_list.append(sys_test_data)
        
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
                            new_catalog[column] = numpy.array([src[column] for src in catalog])[mask]
                new_catalogs.append(self.makeArray(new_catalog))
            results = sys_test(*new_catalogs)
            if hasattr(sys_test.test,'plot'):
                fig = sys_test.test.plot(results)
                fig.savefig(sys_test_data.test_name+'.png')            
            if isinstance(results, matplotlib.figure.Figure):
                results.savefig(sys_test_data.test_name+'.png')    

    def removeFlags(self,catalog):
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
        dtypes = []
        for key in catalog_dict:
            dtypes.append((key,catalog_dict[key].dtype))
        len_list = [len(catalog_dict[key]) for key in catalog_dict]
        if not set(len_list)==0:
            raise RuntimeError('Different catalog lengths for different columns!')
        data = numpy.zeros(len_list[0],dtype=dtypes)
        for key in catalog_dict:
            data[key] = catalog_dict[key]
        return data
        
    def generateColumns(self,mask,cols,catalog,calib_data,calib_type,extra_col_dict):
        for col in cols:
            if not col in catalog.schema:
                if not col in extra_col_dict:
                    extra_col_dict[col] = numpy.zeros(len(catalog))
                    extra_col_dict[col].fill('nan')
                nan_mask = numpy.isnan(extra_col_dict[col])
                nan_and_col_mask = numpy.logical_and(nan_mask,mask)
                if any(nan_and_col_mask>0):
                    extra_col_dict[col][nan_and_col_mask], extra_mask = self.computeExtraColumn(
                        col,catalog[nan_and_col_mask],calib_data,calib_type)
                    if extra_mask is not None:
                        mask[nan_and_col_mask] = numpy.logical_and(mask[nan_and_col_mask],extra_mask)

    def _computeShapeMask(self,data):
        flags = ['flags.negative', 'deblend.nchild', 'deblend.too-many-peaks',
                 'deblend.parent-too-big', 'deblend.failed', 'deblend.skipped', 
                 'deblend.has.stray.flux', 'flags.badcentroid', 'centroid.sdss.flags', 
                 'centroid.naive.flags', 'flags.pixel.edge', 'flags.pixel.interpolated.any',
                 'flags.pixel.interpolated.center', 'flags.pixel.saturated.any', 
                 'flags.pixel.saturated.center', 'flags.pixel.cr.any', 'flags.pixel.cr.center', 
                 'flags.pixel.bad','flags.pixel.suspect.any','flags.pixel.suspect.center']
        masks = [[src.get(flag)==False for src in data] for flag in flags]
        mask = masks[0]
        for new_mask in masks[1:]:
            mask = numpy.logical_and(mask,new_mask)    
	return mask
                           
    def computeExtraColumn(self,col,data,calib_data,calib_type):
        if col=="ra":
            return [src.getRa().asDegrees() for src in data], None
        elif col=="dec":
            return [src.getDec().asDegrees() for src in data], None
        elif col=="mag_err":
            return (2.5/numpy.log(10)*(sources.getPsfFluxErr()/sources.getPsfFlux()),
	            numpy.array([src.get('flux.psf.flags')==0 & 
                                 src.get('flux.psf.flags.psffactor')==0 for src in data]))
        elif col=="mag":
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
                moments = data.get('shape.sdss')
                ixx = moments.getIxx()
                iyy = moments.getIyy()
            except:
                ixx = numpy.array([src.get('shape.sdss').getIxx() for src in data])
                iyy = numpy.array([src.get('shape.sdss').getIyy() for src in data])
            return ((ixx-iyy)/(ixx+iyy), self._computeShapeMask(data))
        elif col=="g2":
            try:
                moments = data.get('shape.sdss')
                ixx = moments.getIxx()
                ixy = moments.getIxy()
                iyy = moments.getIyy()
            except:
                ixx = numpy.array([src.get('shape.sdss').getIxx() for src in data])
                ixy = numpy.array([src.get('shape.sdss').getIxy() for src in data])
                iyy = numpy.array([src.get('shape.sdss').getIyy() for src in data])
            return (2.*ixy/(ixx+iyy), self._computeShapeMask(data))
        elif col=="sigma":
            try:
                moments = data.get('shape.sdss')
                ixx = moments.getIxx()
                iyy = moments.getIyy()
            except:
                ixx = numpy.array([src.get('shape.sdss').getIxx() for src in data])
                iyy = numpy.array([src.get('shape.sdss').getIyy() for src in data])
            return (numpy.sqrt(0.5*(ixx+iyy)), self._computeShapeMask(data))
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
	            self._computeShapeMask(data))
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
                               2. * dg2_dixy**2 * cov_ixy), self._computeShapeMask(data))
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
	            self._computeShapeMask(data))
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
            return numpy.array([1.]*len(data)), None
        raise NotImplementedError("Cannot compute col %s"%col)
        
    @classmethod
    def _makeArgumentParser(cls):
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
    _DefaultName = "CCDNoTractSingleEpochStile"
    
    @classmethod
    def _makeArgumentParser(cls):
        parser = lsst.pipe.base.ArgumentParser(name=cls._DefaultName)
        parser.add_id_argument("--id", "src", help="data ID, with raw CCD keys")
        return parser


class StileFieldRunner(pipeBase.TaskRunner):
    """Subclass of TaskRunner for Stile field tasks.  Most of this code (incl this docstring) 
    pulled from measMosaic.

    FieldSingleEpochStileTask.run() takes a number of arguments, one of which is a list of dataRefs
    extracted from the command line (whereas most CmdLineTasks' run methods take
    single dataRef, are are called repeatedly).  This class transforms the processed
    arguments generated by the ArgumentParser into the arguments expected by
    FieldSingleEpochStileTask.run().

    See pipeBase.TaskRunner for more information.
    """

    @staticmethod
    def getTargetList(parsedCmd, **kwargs):
        # organize data IDs by field
        refListDict = {}
        for ref in parsedCmd.id.refList:
            refListDict.setdefault(ref.dataId["field"], []).append(ref)
        # we call run() once with each tract
        return [(parsedCmd.butler,
                 field,
                 refListDict[field],
                 parsedCmd.debug
                 ) for field in sorted(refListDict.keys())]

    def __call__(self, args):
        task = self.TaskClass(config=self.config, log=self.log)
        result = task.run(*args)

class FieldSingleEpochStileConfig(lsst.pex.config.Config):
    sys_tests = adapter_registry.makeField("tests to run",multi=True,
                    default = ["StatsPSFFlux","StarXGalaxyShear"])

class FieldSingleEpochStileTask(CCDSingleEpochStileTask,MosaicTask):
    RunnerClass = StileFieldRunner
    canMultiprocess = False
    ConfigClass = FieldSingleEpochStileConfig
    _DefaultName = "FieldSingleEpochStile"

    def run(self, butler, field, dataRefList, debug, verbose=False):
        skyMap = butler.get("deepCoadd_skyMap", immediate=True)
        fieldInfo = skyMap[field]
        dataRefListOverlapWithField, dataRefListToUse = self.checkOverlapWithTract(fieldInfo, dataRefList)
        
        catalogs = [dataRef.get("src",immediate=True) for dataRef in dataRefListOverlapWithField]
        catalogs = [self.removeFlags(catalog) for catalog in catalogs]
        calib_data_list = []
        calib_types = []
        for dataRef in dataRefListOverlapWithField:
            try:
                if dataRef.datasetExists("fcr_md"):
                    calib_data_list.append(dataRef.get("fcr_md"))
                    calib_type.append("fcr")
                else:
                    calib_data_list.append(dataRef.get("calexp_md"))
                    calib_type.append("calexp")
            except:
                calib_data_list.append(dataRef.get("calexp_md"))
                calib_type.append("calexp")
                
        sys_data_lists = []
        extra_col_dicts = [{} for catalog in catalogs] 
        for sys_test in self.sys_tests:
            sys_test_data = SysTestData()
            sys_test_data.test_name = sys_test.name
            # Masks expects: a tuple of masks, one for each required data set for the sys_test
            sys_test_data.mask_list = [sys_test.getMasks(catalog) for catalog in catalogs]
            # cols expects: an iterable of iterables, describing for each required data set
            # the set of extra required columns.
            sys_test_data.cols_list = sys_test.getRequiredColumns()
            for (mask_list,cols) in zip(sys_test_data.mask_list,sys_test_data.cols_list):
                for mask, catalog, calib_data, calib_type, extra_col_dict in zip(mask_list, catalogs, calib_data_list, calib_types, extra_col_dicts):
                    self.GenerateColumns(mask,cols,catalog,calib_data,calib_type,extra_col_dict)
            sys_data_list.append(sys_test_data)
        
        for sys_test,sys_test_data in zip(self.sys_tests,sys_data_list):
            new_catalogs = []
            for mask_list,cols in zip(sys_test_data.mask_list,sys_test_data.cols_list):
                new_catalog = {}
                for catalog, extra_col_dict, mask in zip(catalogs, extra_col_dicts, mask_list):
                    for column in cols:
                        if column in extra_col_dict:
                            new_catalog[column] = new_catalog.get(column,[]).append(extra_col_dict[column][mask])
                        elif column in catalog.schema:
                            try:
                                new_catalog[column] = new_catalog.get(column,[]).append(catalog[column][mask])
                            except:
                                new_catalog[column] = new_catalog.get(column,[]).append(numpy.array([src[column] for src in catalog])[mask])
                new_catalogs.append(self.makeArray(new_catalog))
            results = sys_test(*new_catalogs)
            if hasattr(sys_test.test,'plot'):
                fig = sys_test.test.plot(results)
                fig.savefig(sys_test_data.test_name+'.png')            
            if isinstance(results, matplotlib.figure.Figure):
                results.savefig(sys_test_data.test_name+'.png')    
                
    def makeArray(self,catalog_dict):
        dtypes = []
        for key in catalog_dict:
            dtype_list = [cat.dtype for cat in catalog_dict[key]]
            dtype_list.sort()
            dtypes.append((key,dtype_list[-1]))
        len_list = [sum([len(cat) for cat in catalog_dict[key]]) for key in catalog_dict]
        if not set(len_list)==0:
            raise RuntimeError('Different catalog lengths for different columns!')
        data = numpy.zeros(len_list[0],dtype=dtypes)
        for key in catalog_dict:
            current_position = 0
            for catalog in catalog_dict[key]:
                data[key][current_position:current_position+len(catalog)] = catalog
                current_position+=len(catalog)
        return data

