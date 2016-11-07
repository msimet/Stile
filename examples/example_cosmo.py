import sys
sys.path.append('..')
import stile
import stile.cosmo_forecasts
import destest.src.cosmo as cosmo
import numpy

def main():
    # setups
    list_of_tests = [stile.CorrelationFunctionSysTest('Rho1'),
                     stile.CorrelationFunctionSysTest('Rho2'),
                     stile.CorrelationFunctionSysTest('Rho3'),
                     stile.CorrelationFunctionSysTest('Rho4'),
                     stile.CorrelationFunctionSysTest('Rho5')]
    if True:
        # Run on a subset of CFHTLenS data
        data = stile.ReadTable(#'/homes/m/msimet/CFHTLenS/CFHTLens_W1_passed.fits', 
                               '/global/homes/m/msimet/CFHTLens_W2_reject0.fits',
                               fields={'psf_g1': 'PSF_e1', 'psf_g2': 'PSF_e2', 'w': 'weight', 'g1': 'e1', 'g2': 'e2',
                                       'sigma': 'scalelength', 'psf_sigma': 'FLUX_RADIUS'})
        data = numpy.array(data)[:1000]
        data['sigma'] *= data['sigma']
        data['psf_sigma'] *= data['psf_sigma']
    else:
        # Dummy data
        data = numpy.rec.fromarrays((numpy.array([0.5]*10), numpy.array([0.5]*10), numpy.array([0.5]*10), numpy.array([0.5]*10), numpy.linspace(1, 10, num=10), numpy.linspace(1, 10, num=10), numpy.linspace(1, 10, num=10), numpy.linspace(1, 10, num=10), numpy.linspace(1, 10, num=10)),
                names = ['g1', 'g2', 'psf_g1', 'psf_g2', 'w', 'sigma', 'psf_sigma', 'dec', 'ra'])
    
    # Run stile and get results
    config = {'ra_units': 'degrees', 'dec_units': 'degrees',
                  'min_sep': 0.1, 'max_sep': 500, 'sep_units': 'arcmin', 'nbins': 25}
    list_of_results = [test(data, config=config) for test in list_of_tests]
    
    
    cosmo_obj = stile.cosmo_forecasts.XiSet(data, data, config, list_of_results)
    deltaxi = cosmo_obj.computeError()
    print deltaxi

    # Prep destest
    workdir='../destest'
    c0=cosmo._cosmosis(infile=workdir+'cosmosis.ini',fitsfile=workdir+'lsst_default.fits',values=workdir+'values_fixed.ini')
    c0.xi(1,1,theta=deltaxi['meanr'])
    cosmo.run.submit_rho_leakage_test(
        deltaxi, # delta xi+
        0., # a config dict that could be passed from stile - unused currently (don't remember the format of the output from the stile example)
        test='rho_leakage', # name of the directory into which all the results go: workdir+test
        workdir=workdir, # working directory where files are/output goes
        procs=1, # number of procs to use for cosmosis
        hr=3, # time limit on job
        cosmosisrootdir='', # where cosmosis is. not currently needed, but could be cleaned up/generalised to use this in the way the photo-z version does
        cosmosissource='source /scratch2/scratchdirs/troxel/cosmosis/setup-cosmosis-nersc-edison', # source command for cosmosis
        inifile='cosmosis.ini', # location of cosmosis script
        valuefile='values.ini', # location of values file (controls what parameters you vary). there's the option in destest/src/cosmo.py to write this file on the fly, but for simplicity I've just saved it.
        submit=False) # whether to try submitting the job directly from python. This only works for me currently, but you could edit the function for your environment. If false, it writes the bash script containing what you need to run for cosmosis.
      # You'll need to modify the bash script a bit. I haven't finished debugging running cosmosis in submitted mpi jobs on edison yet, so my tests were on the login node, and you can't mpirun from there (its srun on nersc anyway, so other things need to change too once I get that working).    

if __name__=='__main__':
    main()

