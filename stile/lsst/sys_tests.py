import lsst.pex.config
from .. import sys_tests
import numpy

adapter_registry = lsst.pex.config.makeRegistry("Stile test outputs")

class StarGalaxyCrossCorrelationAdapterConfig(lsst.pex.config.Config):
    pass
    
class StarGalaxyCrossCorrelationAdapter(object):
    ConfigClass = StarGalaxyCrossCorrelationAdapterConfig
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
    
class StatsPSFFluxAdapterConfig(lsst.pex.config.Config):
    pass
    
class StatsPSFFluxAdapter(object):
    ConfigClass = StatsPSFFluxAdapterConfig
    def __init__(self,config):
        self.config = config
        self.test = sys_tests.StatSysTest(field='flux.psf')
        self.name = self.test.short_name+'flux.psf'

    def __call__(self,*data):
        self.test(*data)
    
    def getMasks(self,catalog):
        return_cat = numpy.zeros(len(catalog),dtype=bool)
        return_cat.fill(True)
        return [return_cat]
        
    def getRequiredColumns(self):
        return (('flux.psf',),)
        
adapter_registry.register("StatsPSFFlux",StatsPSFFluxAdapter)
