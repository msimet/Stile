import pex.config
from .. import sys_tests
import numpy

adapter_registry = lsst.pex.config.makeRegistry()

class StarGalaxyCrossCorrelationAdapterConfig(pex.config.Config):
    pass
    
class StarGalaxyCrossCorrelationAdapter(object):
    def __init__(self,config):
        self.config = config
        self.test = sys_tests.StarGalaxyCrossCorrelationSysTest()
        self.name = self.test.short_name
     
    def __call__(self,*data):
        self.test(*data)
    
    def getMasks(self):
        raise NotImplementedError()
        
    def getRequiredColumns(self):
        raise NotImplementedError()
    
class StatsPSFFluxAdapterConfig(pex.config.Config):
    pass
    
class StatsPSFFluxAdapter(object):
    def __init__(self,config):
        self.config = config
        self.test = sys_tests.Stats(field='psf.flux')
        self.name = self.test.short_name+'_psf.flux'

    def __call__(self,*data):
        self.test(*data)
    
    def getMasks(self,catalog):
        return_cat = numpy.zeros(len(catalog),dtype=bool)
        return_cat.fill(True)
        return [return_cat]
        
    def getRequiredColumns(self):
        return ['psf.flux']
        