import pex.config
from .. import sys_tests

adapter_registry = lsst.pex.config.makeRegistry()

class StarGalaxyCrossCorrelationAdapterConfig(pex.config.Config):

class StarGalaxyCrossCorrelationAdapter(object):
    def __init__(self,config)
        self.config = config
        self.test = sys_tests.StarGalaxyCrossCorrelationSysTest()
        self.name = self.test.short_name
     
    def __call__(self,*data):
        self.test(*data)
    
    def getMasks(self):
        raise NotImplementedError()
        
    def getRequiredColumns(self):
        raise NotImplementedError()
    