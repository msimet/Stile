import lsst.pex.config
import lsst.pipe.base
from lsst.pipe.tasks.dataIds import PerTractRawDataIDContainer
from .sys_tests import adapter_registry
import numpy


class CCDSingleEpochStileConfig(lsst.pex.config.Config):
    sys_tests = adapter_registry.make_field("tests to run",multi=True,
                    default = ["StarGalaxyCrossCorrelation"])
    
class SysTestData(object):
    def __init__(self):
        self.test_name = None
        self.mask_list = None
        self.cols_list = None
    
class CCDSingleEpochStileTask(lsst.pipe.base.CmdLineTask):
    ConfigClass = CCDSingleEpochStileConfig
    required_columns = None
    
    def __init__(self,**kwargs):
        lsst.pipe.base.CmdLineTask.__init__(self,**kwargs)
        self.sys_tests = self.config.sys_tests.apply()
        
    def run(self,dataRef):
        catalog = dataRef.get("src",immediate=True)
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
            for (mask,cols) in zip(sys_test.mask_list,sys_test.cols_list):
                for col in cols:
                    if not col in catalog.schema:
                        if not col in extra_col_dict:
                            extra_col_dict[col] = numpy.zeroes(len(catalog))
                            extra_col_dict[col].fill('nan')
                        nan_mask = col_dict[col]=='nan'
                        nan_and_col_mask = numpy.logical_and(nan_mask,mask)
                        extra_col_dict[col][nan_and_col_mask] = self.computeExtraColumn(                                             col,catalog[nan_and_col_mask])
            sys_data_list.append(sys_test_data)
        
        for sys_test,sys_test_data in zip(self.sys_tests,sys_data_list):
            new_catalogs = []
            for mask,cols in zip(sys_test.mask_list,sys_test.cols_list):
                for column in cols:
                    if column in extra_col_dict:
                        new_catalog[column] = cols[column]
                    else column in catalog.schema:
                        new_catalog[column] = catalog[column]
                new_catalogs.append(self.makeArray(new_catalog))
            sys_test(*new_catalogs,verbose=True)
    
    def makeArray(self,catalog_dict):
        dtypes = []
        for key in catalog_dict:
            dtypes.append((key,catalog_dict[key].dtype))
        data = numpy.zeros(len(catalog_dict[key]),dtype=dtypes)
        for key in catalog_dict:
            data[key] = catalog_dict[key]
        return data

    def computeExtraColumn(self,col,data):
        raise NotImplementedError()
        
    @classmethod
    def _makeArgumentParser(cls):
        parser = lsst.pipe.base.ArgumentParser(name=cls._DefaultName)
        parser.add_id_argument("--id", "forced_src", help="data ID, with raw CCD keys + tract",
                               ContainerClass=PerTractRawDataIDContainer)
        return parser

    def writeConfig(self, *args, **kwargs):
        pass
    def writeSchema(self, *args, **kwargs):
        pass
    def writeMetadata(self, dataRef):
        pass
