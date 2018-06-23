import numpy
import unittest

try:
    import stile
except ImportError:
    sys.path.append('..')
    import stile

class TestSysTests(unittest.TestCase):
    def test_binned_whisker_plots(self):
        source_data = stile.ReadASCIITable('../examples/example_source_catalog.dat',
                fields={'id': 0, 'ra': 1, 'dec': 2, 'z': 3, 'g1': 4, 'g2': 5})
        data = numpy.rec.fromarrays([source_data['ra'], source_data['dec'], source_data['g1'],
                    source_data['g2'], source_data['g1']+0.1*source_data['ra'], source_data['g2']+0.05*source_data['dec'],
                    numpy.ones_like(source_data['ra']), numpy.ones_like(source_data['ra'])+0.02],
                    names = ['x', 'y', 'g1', 'g2', 'psf_g1', 'psf_g2', 'sigma', 'psf_sigma'])
        obj = stile.WhiskerPlotSysTest("BinnedStar")
        obj(data)
        obj = stile.WhiskerPlotSysTest("BinnedPSF")
        obj(data)
        obj = stile.WhiskerPlotSysTest("BinnedResidual")
        obj(data)
        data['x'] = numpy.log10(1+data['x'])
        obj = stile.WhiskerPlotSysTest("BinnedStar")
        obj(data)
        obj = stile.WhiskerPlotSysTest("BinnedStar")
        obj(data, split_by_objects=False, n_x=50, n_y=50)
        
        

if __name__=='__main__':
    unittest.main()

