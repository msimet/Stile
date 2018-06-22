import numpy
import unittest

try:
    import stile
except ImportError:
    sys.path.append('..')
    import stile


class TestSysTests(unittest.TestCase):
    def test_ScatterPlotRequirements(self):
        data = stile.ReadASCIITable('../examples/example_source_catalog.dat',
                                    fields={'id': 0, 'ra': 1, 'dec': 2, 'z': 3, 'g1': 4, 'g2': 5})
        test_data = numpy.rec.fromarrays([data['ra'], data['dec'], 
                data['g1'], data['g2'], data['g1']+0.05, data['g2']+0.01, 0.1*data['g1'], 0.1*data['g2']], 
                names=['ra', 'dec', 'g1', 'g2', 'psf_g1', 'psf_g2', 'g1_err', 'g2_err'])
        req_x = numpy.array([numpy.min(data['g1']), numpy.max(data['g1'])])
        req_y = req_x
        req_y_range = [req_y*0.9, req_y*1.1]

        scatterplot = stile.ScatterPlotSysTest('StarVsPSFG1')
        results = scatterplot(test_data)
        results.savefig('p1.png')
        res = scatterplot.plot(results, requirement_x=req_x, requirement_y=req_y)
        res.savefig('p2.png')
        res = scatterplot.plot(results, requirement_x=req_x, requirement_y=req_y, requirement_color='orange', requirement_linestyle='dotted')
        res.savefig('p3.png')
        res = scatterplot.plot(results, requirement_x=req_x, requirement_y=req_y, requirement_y_range=req_y_range)
        res.savefig('p4.png')
        res = scatterplot.plot(results, requirement_x=req_x, requirement_y=req_y, requirement_y_range=req_y_range, requirement_color='orange', requirement_linestyle='dotted')
        res.savefig('p5.png')

if __name__=='__main__':
    unittest.main()

