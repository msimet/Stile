import lsst.pex.config
import lsst.pipe.base
import lsst.meas.mosaic
from lsst.pipe.tasks.dataIds import PerTractCcdDataIdContainer
from .sys_test_adapters import adapter_registry
import numpy


class CCDSingleEpochStileConfig(lsst.pex.config.Config):
    sys_tests = adapter_registry.makeField("tests to run",multi=True,
                    default = ["StatsPSFFlux"])
    
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
        if dataRef.datasetExists("fcr_md"):
            calib_data = dataRef.get("fcr_md")
            calib_type="fcr"
        else:
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
                for col in cols:
                #print col, col in catalog.schema, catalog.schema, "check"
                    if not col in catalog.schema:
                        if not col in extra_col_dict:
                            extra_col_dict[col] = numpy.zeros(len(catalog))
                            extra_col_dict[col].fill('nan')
                        nan_mask = extra_col_dict[col]=='nan'
                        nan_and_col_mask = numpy.logical_and(nan_mask,mask)
                        extra_col_dict[col][nan_and_col_mask] = self.computeExtraColumn(
                            col,catalog[nan_and_col_mask],calib_data,calib_type)
            sys_data_list.append(sys_test_data)
        
        for sys_test,sys_test_data in zip(self.sys_tests,sys_data_list):
            new_catalogs = []
            for mask,cols in zip(sys_test_data.mask_list,sys_test_data.cols_list):
                new_catalog = {}
                for column in cols:
                    if column in extra_col_dict:
                        new_catalog[column] = cols[column]
                    elif column in catalog.schema:
                        new_catalog[column] = catalog[column]
                new_catalogs.append(self.makeArray(new_catalog))
            results = sys_test(*new_catalogs)
            
    
    def removeFlags(self,catalog):
        flags = ['deblend.too-many-peaks','deblend.parent-too-big','deblend.failed',
                 'deblend.skipped','flags.badcentroid','flags.pixel.edge','flags.pixel.bad',
                 'flux.aperture.flags','flux.gaussian.flags','flux.kron.flags','flux.naive.flags',
                 'flux.psf.flags']
        mask = numpy.logical_and([catalog[flag]==False for flag in flags])
        return catalog[mask]
            
    def makeArray(self,catalog_dict):
        dtypes = []
        for key in catalog_dict:
            dtypes.append((key,catalog_dict[key].dtype))
        data = numpy.zeros(len(catalog_dict[key]),dtype=dtypes)
        for key in catalog_dict:
            data[key] = catalog_dict[key]
        return data

    def computeExtraColumn(self,col,data,calib_data,calib_type):
        if col=="ra":
            return data.getRa().asDegrees()
        elif col=="dec":
                return data.getDec().asDegrees()
        elif col=="mag_err":
            return 2.5/numpy.log(10)*(sources.getPsfFluxErr()/sources.getPsfFlux())
        elif col=="mag":
            if calib_type=="fcr":
                ffp = lsst.meas.mosaic.FluxFitParams(calib_data)
                x, y = data.getX(), data.getY()
                correction = numpy.array(ffp.eval(x[i],y[i]) for i in range(n)])
                zeropoint = 2.5*numpy.log10(fcr.get("FLUXMAG0")) + correction
            elif calib_type=="calexp":
                zeropoint = 2.5*numpy.log10(calib_data.get("FLUXMAG0"))
            return zeropoint - 2.5*numpy.log10(data.getPsfFlux())
        elif col=="w":
            raise NotImplementedError("Need to figure out something clever for weights")
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
