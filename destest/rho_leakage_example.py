import numpy as np
import src.cosmo as cosmo

# Note this uses twopoint.py and enum34.py which are taken from github/joezuntz/2point. I've added them into this repo just to make things work, but we'll need to find a better way of calling that from a clone of Joe's repo.

if __name__=='__main__':

  # Write empty deltaxi structure
  deltaxi=np.empty(25, dtype=[('meanr','f8')]+[('xi','f8')])
  # Agreed upon binning for test
  deltaxi['meanr']=((np.logspace(np.log10(.1),np.log10(500),26))[:-1]+(np.logspace(np.log10(.1),np.log10(500),26))[1:])/2.
  # Where the files are and output goes
  workdir='/scratch2/scratchdirs/troxel/Stile/destest/'
  # Get a default xi+ to modify to create a dummy delta xi+
  c0=cosmo._cosmosis(infile=workdir+'cosmosis.ini',fitsfile=workdir+'lsst_default.fits',values=workdir+'values_fixed.ini')
  c0.xi(1,1,theta=deltaxi['meanr'])
  # Simulate 5% PSF leakage into xi+
  deltaxi['xi']=c0.xip*.05

  # Where all the work happens. This function: 
  # 1) takes the psf leakage and adds it to a theory xi+ that is the same theory that the leakage is made from
  # 2) writes this modified xi+ and xi- to a twopoint fits file with n(z) and covariance taken from the lsst_default.fits file
  # 3) sets up a bash script to do the cosmosis run 
  # You'll need to source cosmosis before running this because of step (1). We may use a measured xi+- as baseline later (?) and won't need to do this. 
  # Currently only modifies xi+ (additive systematics), but could modify xi- as well (multiplicative or other), with two line changes.
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