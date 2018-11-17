import numpy
import unittest

try:
    import stile
except ImportError:
    sys.path.append('..')
    import stile

class TestSysTests(unittest.TestCase):
    def test_histogram_scatterplot(self):
        source_data = stile.ReadASCIITable('../examples/example_source_catalog.dat',
                fields={'id': 0, 'ra': 1, 'dec': 2, 'z': 3, 'g1': 4, 'g2': 5})
        data = numpy.rec.fromarrays([source_data['ra'], source_data['dec'], source_data['g1'],
                    source_data['g2'], source_data['g1']+0.1*source_data['ra'], source_data['g2']+0.05*source_data['dec'],
                    numpy.ones_like(source_data['ra']), numpy.ones_like(source_data['ra'])+0.02,
                    source_data['g1']*0.05, source_data['g2']*0.05, source_data['g1']*0.05, 
                    source_data['g2']*0.05],
                    names = ['x', 'y', 'g1', 'g2', 'psf_g1', 'psf_g2', 'sigma', 'psf_sigma',
                             'g1_err', 'g2_err', 'psf_g1_err', 'psf_g2_err'])
        obj = stile.ScatterPlotSysTest('StarVsPSFG1')
        obj(data, color='g')
        obj(data, histogram=True, color='g')
            
    
if __name__=='__main__':
    unittest.main()

