import sys
try:
    import stile
except ImportError:
    sys.path.append('..')
    import stile

class temp_data_handler():
    def __init__(self):
        self.temp_dir = '.'

expected_result = []
        
def test_CorrelationFunctionSysTest():
    stile_args = {'corr2_kwargs': { 'ra_units': 'degrees', 
                                    'dec_units': 'degrees',
                                    'min_sep': 0.05,
                                    'max_sep': 1,
                                    'sep_units': 'degrees',
                                    'nbins': 20
                                    } }
    cf = stile.CorrelationFunctionSysTest()
    dh = temp_data_handler()
    results = cf.getCorrelationFunction(stile_args,dh,'ng',
                                        ('name','../examples/example_lens_catalog.dat'),
                                        ('name','../examples/example_source_catalog.dat')))
    numpy.testing.assert_almost_equal(results,expected_result)
    kwargs = stile_args['corr2_kwargs']
    stile_args['corr2_kwargs'] = {}
    results2 = cf.getCorrelationFunction(stile_args,dh,'ng',
                                         ('name','../examples/example_lens_catalog.dat'),
                                         ('name','../examples/example_source_catalog.dat')),
                                         **kwargs)
    numpy.testing.assert_equal(results,results2)
    numpy.testing.assert_raises(,cf.getCorrelationFunction)
    numpy.testing.assert_raises(,cf.getCorrelationFunction,stile_args,dh,'ng',
                                 ('name','../examples/example_lens_catalog.dat'),
                                 ('name','../examples/example_source_catalog.dat'))
    numpy.testing.assert_raises(,cf.getCorrelationFunction,stile_args,dh,'ng',
                                 ('name','../examples/example_lens_catalog.dat'))
    numpy.testing.assert_raises(,cf.getCorrelationFunction,stile_args,dh,'hello',
                                 ('name','../examples/example_lens_catalog.dat'),
                                 ('name','../examples/example_source_catalog.dat'))
    realshear = stile.RealShearSysTest()
    results3 = realshear(stile_args,dh,('name','../examples/example_lens_catalog.dat'),
                                       ('name','../examples/example_source_catalog.dat'))
    numpy.testing.assert_equal(results,results3)
    
if __name__=='__main__':
    test_CorrelationFunctionSysTest()
