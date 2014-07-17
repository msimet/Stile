import lsst.pex.config
from .. import sys_tests
import numpy

adapter_registry = lsst.pex.config.makeRegistry("Stile test outputs")

default_corr2_args = { 'ra_units': 'degrees', 
                                   'dec_units': 'degrees',
                                   'min_sep': 0.05,
                                   'max_sep': 1,
                                   'sep_units': 'degrees',
                                   'nbins': 20
                     }

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
        return numpy.array([src.get('calib.psf.used')==True for src in data])

mask_dict = {'galaxy': MaskGalaxy,
             'star': MaskStar,
             'star bright': MaskBrightStar,
             'galaxy lens': MaskGalaxyLens,
             'star PSF': MaskPSFStar}

class StarXGalaxyDensityConfig(lsst.pex.config.Config):
    pass
    
class StarXGalaxyDensityAdapter(object):
    ConfigClass = StarXGalaxyDensityConfig
    def __init__(self,config):
        self.config = config
        self.test = sys_tests.StarXGalaxyDensitySysTest()
        self.name = self.test.short_name
        self.mask_funcs = [mask_dict[obj_type] for obj_type in self.test.objects_list]
     
    def __call__(self,*data):
        return self.test(default_corr2_args,*data)
    
    def getMasks(self,data):
        return [mask_func(data) for mask_func in self.mask_funcs]
        
    def getRequiredColumns(self):
        return self.test.required_quantities
    
class StarXGalaxyShearConfig(lsst.pex.config.Config):
    pass
    
class StarXGalaxyShearAdapter(object):
    ConfigClass = StarXGalaxyShearConfig
    def __init__(self,config):
        self.config = config
        self.test = sys_tests.StarXGalaxyShearSysTest()
        self.name = self.test.short_name
        self.mask_funcs = [mask_dict[obj_type] for obj_type in self.test.objects_list]
     
    def __call__(self,*data):
        return self.test(default_corr2_args,*data)
    
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
        self.mask_funcs = [mask_dict[obj_type] for obj_type in ['galaxy']]

    def __call__(self,*data):
        return self.test(*data,verbose=True)
    def getMasks(self,data):
        return [mask_func(data) for mask_func in self.mask_funcs]
        
    def getRequiredColumns(self):
        return (('flux.psf',),)

class ScatterPlotStarVsPsfG1AdapterConfig(lsst.pex.config.Config):
    pass
    
class ScatterPlotStarVsPsfG1Adapter(object):
    ConfigClass = StatsPSFFluxAdapterConfig
    def __init__(self,config):
        self.config = config
        self.test = sys_tests.ScatterPlotStarVsPsfG1SysTest()
        self.name = self.test.short_name
        self.mask_funcs = [mask_dict[obj_type] for obj_type in self.test.objects_list]

    def __call__(self,*data):
        return self.test(*data, lim=5)

    def getMasks(self,data):
        return [mask_func(data) for mask_func in self.mask_funcs]
        
    def getRequiredColumns(self):
        return self.test.required_quantities

class ScatterPlotStarVsPsfG2AdapterConfig(lsst.pex.config.Config):
    pass
    
class ScatterPlotStarVsPsfG2Adapter(object):
    ConfigClass = StatsPSFFluxAdapterConfig
    def __init__(self,config):
        self.config = config
        self.test = sys_tests.ScatterPlotStarVsPsfG2SysTest()
        self.name = self.test.short_name
        self.mask_funcs = [mask_dict[obj_type] for obj_type in self.test.objects_list]

    def __call__(self,*data):
        return self.test(*data, lim=5)

    def getMasks(self,data):
        return [mask_func(data) for mask_func in self.mask_funcs]
        
    def getRequiredColumns(self):
        return self.test.required_quantities

class ScatterPlotStarVsPsfSigmaAdapter(object):
    ConfigClass = StatsPSFFluxAdapterConfig
    def __init__(self,config):
        self.config = config
        self.test = sys_tests.ScatterPlotStarVsPsfSigmaSysTest()
        self.name = self.test.short_name
        self.mask_funcs = [mask_dict[obj_type] for obj_type in self.test.objects_list]

    def __call__(self,*data):
        return self.test(*data, lim=5)

    def getMasks(self,data):
        return [mask_func(data) for mask_func in self.mask_funcs]
        
    def getRequiredColumns(self):
        return self.test.required_quantities

class ScatterPlotResidualVsPsfG1AdapterConfig(lsst.pex.config.Config):
    pass
    
class ScatterPlotResidualVsPsfG1Adapter(object):
    ConfigClass = StatsPSFFluxAdapterConfig
    def __init__(self,config):
        self.config = config
        self.test = sys_tests.ScatterPlotResidualVsPsfG1SysTest()
        self.name = self.test.short_name
        self.mask_funcs = [mask_dict[obj_type] for obj_type in self.test.objects_list]

    def __call__(self,*data):
        return self.test(*data, lim=5)

    def getMasks(self,data):
        return [mask_func(data) for mask_func in self.mask_funcs]
        
    def getRequiredColumns(self):
        return self.test.required_quantities

class ScatterPlotResidualVsPsfG2AdapterConfig(lsst.pex.config.Config):
    pass
    
class ScatterPlotResidualVsPsfG2Adapter(object):
    ConfigClass = StatsPSFFluxAdapterConfig
    def __init__(self,config):
        self.config = config
        self.test = sys_tests.ScatterPlotResidualVsPsfG2SysTest()
        self.name = self.test.short_name
        self.mask_funcs = [mask_dict[obj_type] for obj_type in self.test.objects_list]

    def __call__(self,*data):
        return self.test(*data, lim=5)

    def getMasks(self,data):
        return [mask_func(data) for mask_func in self.mask_funcs]
        
    def getRequiredColumns(self):
        return self.test.required_quantities

class ScatterPlotResidualVsPsfSigmaAdapterConfig(lsst.pex.config.Config):
    pass
    
class ScatterPlotResidualVsPsfSigmaAdapter(object):
    ConfigClass = StatsPSFFluxAdapterConfig
    def __init__(self,config):
        self.config = config
        self.test = sys_tests.ScatterPlotResidualVsPsfSigmaSysTest()
        self.name = self.test.short_name
        self.mask_funcs = [mask_dict[obj_type] for obj_type in self.test.objects_list]

    def __call__(self,*data):
        return self.test(*data, lim=5)

    def getMasks(self,data):
        return [mask_func(data) for mask_func in self.mask_funcs]
        
    def getRequiredColumns(self):
        return self.test.required_quantities
        
adapter_registry.register("StatsPSFFlux",StatsPSFFluxAdapter)
#adapter_registry.register("StarXGalaxyDensity",StarXGalaxyDensityAdapter)
adapter_registry.register("StarXGalaxyShear",StarXGalaxyShearAdapter)
adapter_registry.register("ScatterPlotStarVsPsfG1", ScatterPlotStarVsPsfG1Adapter)
adapter_registry.register("ScatterPlotStarVsPsfG2", ScatterPlotStarVsPsfG2Adapter)
adapter_registry.register("ScatterPlotStarVsPsfSigma", ScatterPlotStarVsPsfSigmaAdapter)
adapter_registry.register("ScatterPlotResidualVsPsfG1", ScatterPlotResidualVsPsfG1Adapter)
adapter_registry.register("ScatterPlotResidualVsPsfG2", ScatterPlotResidualVsPsfG2Adapter)
adapter_registry.register("ScatterPlotResidualVsPsfSigma", ScatterPlotResidualVsPsfSigmaAdapter)
