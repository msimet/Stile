import numpy
from sys_tests import BaseCorrelationFunctionSysTest

class CosmoSet(object):
    """ A class that holds the outputs & configuration parameters for a number of SysTets which
    we will combine to make a cosmological forecast. """
    pass
    
class XiSet(CosmoSet):
    required_tests = {'CorrelationFunctionSysTest': ['Rho1', 'Rho2', 'Rho3', 'Rho4', 'Rho5'],
                      'StatSysTest': ['psf_size', 'size']}
    def __init__(star_data, galaxy_data, config, list_of_outputs):
        self.star_data = star_data
        self.galaxy_data = galaxy_data
        self.list_of_outputs = list_of_outputs
        self.config = config
    def computeTraces(self):
        trace_plus_arr = (self.galaxy_data['psf_size']-self.galaxy_data['size'])/self.galaxy_data['size']
        trace_1_arr = self.galaxy_data['psf_size']/self.galaxy_data['size']
        sst = stile.StatSysTest()
        res = sst(trace_plus_arr)
        trace_plus = res.mean
        res = sst(trace_1_arr)
        trace_1 = res.mean
        return trace_plus, trace_gal
    def computeAlpha(self):
        gp = GalaxyPSFCorrelationFunction()
        gp_res = gp(self.star_data, self.galaxy_data, config=self.config)
        pp = PSFPSFCorrelationFunction()
        pp_res = pp(self.star_data, config=self.config)
        stat = StatSysTest()
        res = stat(self.galaxy_data['g1'])
        e1_gal = res.mean
        res = stat(self.galaxy_data['g2'])
        e2_gal = res.mean
        res = stat(self.star_data['psf_g1'])
        e1_psf = res.mean
        res = stat(self.star_data['psf_g2'])
        e2_psf = res.mean
        # todo: check imaginary parts
        return (gp_res['xi+']-(e1_gal*e1_psf + e2_gal*e2_psf))/(pp_res['xi+']-e1_psf**2-e2_psf**2)
    def computeError(self):
        gg = GalaxyCorrelationFunction()
        gg_res = gg(self.galaxy_data, self.config)
        trace_mean, trace_1 = self.computeTraces()
        alpha = self.computeAlpha()
        rho1, rho2, rho3, rho4, rho5 = self.list_of_outputs[:5]
        return (2*trace_mean*gg_res['xi+'] + trace_1**2*rho1
                -alpha*trace_1*rho2+trace_1**2*rho3
                +trace_1**2*rho4 - alpha*trace_1*rho5)


class GalaxyCorrelationFunction(BaseCorrelationFunctionSysTest):
    short_name = 'xi_gp'
    long_name = 'Galaxy shape correlation function'
    objects_list = ['galaxy']
    required_quantities = [('ra', 'dec', 'g1', 'g2', 'w')]
    def __call__(self, data, data2=None, random=None, random2=None, config=None, **kwargs):
        return self.getCF('gg', data, data2, random, random2,
                          config=config, **kwargs)        
    
        
class GalaxyPSFCorrelationFunction(BaseCorrelationFunctionSysTest):
    short_name = 'xi_gp'
    long_name = 'Galaxy-PSF correlation function'
    objects_list = ['star PSF', 'galaxy']
    required_quantities = [('ra', 'dec', 'psf_g1', 'psf_g2', 'w'),
                           ('ra', 'dec', 'g1', 'g2', 'w')]

    def __call__(self, data, data2=None, random=None, random2=None, config=None, **kwargs):
        new_data = numpy.rec.fromarrays([data['ra'], data['dec'], data['psf_g1'],
                                         data['psf_g2'], data['w']],
                                         names = ['ra', 'dec', 'g1', 'g2', 'w'])
        if random:
            new_random = numpy.rec.fromarrays([random['ra'], random['dec'], random['psf_g1'],
                                               random['psf_g2'], random['w']],
                                               names = ['ra', 'dec', 'g1', 'g2', 'w'])

        else:
            new_random = random
        return self.getCF('gg', new_data, data2, new_random, random2,
                          config=config, **kwargs)        

class PSFPSFCorrelationFunction(BaseCorrelationFunctionSysTest):
    short_name = 'xi_pp'
    long_name = 'PSF-PSF correlation function'
    objects_list = ['star PSF']
    required_quantities = [('ra', 'dec', 'psf_g1', 'psf_g2', 'w')]

    def __call__(self, data, data2=None, random=None, random2=None, config=None, **kwargs):
        new_data = numpy.rec.fromarrays([data['ra'], data['dec'], data['psf_g1'],
                                         data['psf_g2'], data['w']],
                                         names = ['ra', 'dec', 'g1', 'g2', 'w'])
        if data2 is not None:
             new_data2 = numpy.rec.fromarrays([data2['ra'], data2['dec'], data2['g1']-data2['psf_g1'],
                                          data2['g2']-data2['psf_g2'], data2['w']],
                                          names = ['ra', 'dec', 'g1', 'g2', 'w'])
        else:
            new_data2 = data2
        if random:
            new_random = numpy.rec.fromarrays([random['ra'], random['dec'], random['psf_g1'],
                                               random['psf_g2'], random['w']],
                                               names = ['ra', 'dec', 'g1', 'g2', 'w'])

        else:
            new_random = random
        if random2 is not None:
            new_random2 = numpy.rec.fromarrays([random2['ra'], random2['dec'],
                                                random2['psf_g1'],
                                                random2['psf_g2'], random2['w']],
                                                names = ['ra', 'dec', 'g1', 'g2', 'w'])
        else:
            new_random2 = random2
        return self.getCF('gg', new_data, new_data2, new_random, new_random2,
                          config=config, **kwargs)        

