import lsst.pex.config
from .. import sys_tests
import numpy

adapter_registry = lsst.pex.config.makeRegistry("Stile test outputs")

def MaskGalaxy(data):
    return data['classification_extendedness']==1

def MaskStar(data):
    return data['classification_extendedness']==0

def MaskBrightStar(data):
    # We could hard-code a level, or do this:
    star_mask = mask_star(data)
    top_tenth = numpy.percentile(data['flux.psf'][star_mask],0.9)
    top_tenth_mask = data['flux.psf']>top_tenth
    return numpy.logical_and(star_mask,top_tenth_mask)

def MaskGalaxyLens(data):
    # Should probably...pick the nearest/biggest/brightest ones? Randomly select?
    return mask_galaxy(data)

def MaskPSFStar(data):
    return data['calib.psf.used']==True

mask_dict = {'galaxy': MaskGalaxy,
             'star': MaskStar,
             'star bright': MaskBrightStar,
             'galaxy lens': MaskGalaxyLens,
             'star PSF': MaskPSFStar}

class StarGalaxyCrossCorrelationAdapterConfig(lsst.pex.config.Config):
    pass
    
class StarGalaxyCrossCorrelationAdapter(object):
    ConfigClass = StarGalaxyCrossCorrelationAdapterConfig
    def __init__(self,config):
        self.config = config
        self.test = sys_tests.StarGalaxyCrossCorrelationSysTest()
        self.name = self.test.short_name
        self.mask_funcs = [mask_dict[obj_type] for obj_type in self.test.objects_list]
     
    def __call__(self,*data):
        self.test(*data)
    
    def getMasks(self,data):
        return [mask_func(data) for mask_func in self.mask_funcs]
        
    def getRequiredColumns(self):
        return self.test.required_quantities
    
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
