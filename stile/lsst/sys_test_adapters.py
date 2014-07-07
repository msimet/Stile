import lsst.pex.config
from .. import sys_tests
import numpy

adapter_registry = lsst.pex.config.makeRegistry("Stile test outputs")

def MaskGalaxy(data):
    try:
        return data['classification.extendedness']==1
    except:
	return numpy.array([src['classification.extendedness']==1 for src in data])

def MaskStar(data):
    try:
        return data['classification.extendedness']==0
    except:
        return numpy.array([src['classification.extendedness']==1 for src in data])

def MaskBrightStar(data):
    # We could hard-code a level, or do this:
    star_mask = MaskStar(data)
    try:
        top_tenth = numpy.percentile(data['flux.psf'][star_mask],0.9)
        top_tenth_mask = data['flux.psf']>top_tenth
    except:
        flux = numpy.array([src['flux.psf'] for src in data])
	top_tenth = numpy.percentile(flux[star_mask],0.9)
	top_tenth_mask = flux>top_tenth
    return numpy.logical_and(star_mask,top_tenth_mask)

def MaskGalaxyLens(data):
    # Should probably...pick the nearest/biggest/brightest ones? Randomly select?
    return MaskGalaxy(data)

def MaskPSFStar(data):
    try:
        return data['calib.psf.used']==True
    except:
        return numpy.array([src['calib.psf.used']==True for src in data])

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
        self.test(*data,verbose=True)
    
    def getMasks(self,catalog):
    	return MaskGalaxy(catalog)
        return_cat = numpy.zeros(len(catalog),dtype=bool)
        return_cat.fill(True)
        return [return_cat]
        
    def getRequiredColumns(self):
        return (('flux.psf',),)
        
adapter_registry.register("StatsPSFFlux",StatsPSFFluxAdapter)
