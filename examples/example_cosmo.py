import sys
sys.path.append('..')
import stile
import stile.cosmo_forecasts
import dummy

def main():
    # setups
    list_of_tests = [stile.CorrelationFunctionSysTest('Rho1'),
                     stile.CorrelationFunctionSysTest('Rho2'),
                     stile.CorrelationFunctionSysTest('Rho3'),
                     stile.CorrelationFunctionSysTest('Rho4'),
                     stile.CorrelationFunctionSysTest('Rho5')]
    data = stile.ReadTable('/homes/m/msimet/CFHTLenS/CFHTLens_W1_passed.fits', 
                           fields={'psf_g1': 'PSF_g1', 'psf_g2': 'PSF_g2', 'w': 'weight',
                                   'size': 'scalelength', 'psf_size': 'FLUX_RADIUS'})
    data['size'] *= data['size']
    data['psf_size'] *= data['psf_size']
    
    stile_args = {'ra_units': 'degrees', 'dec_units': 'degrees',
                  'min_sep': 0.1, 'max_sep': 500, 'sep_units': 'arcmin 'nbins': 25}
    list_of_results = [test(data, config=stile_args) for test in list_of_tests]
    
    cosmo = stile.cosmo_forecasts.XiSet(data, data, config, list_of_results)
    res = cosmo.computeError()
    

if __name__=='__main__':
    main()

