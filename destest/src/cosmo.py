import numpy as np
import os
import scipy
import fitsio as fio

import catalog
import config
import fig
import txt

vary = {
  
  'omega_m' : False,
  'h0' : False,
  'omega_b' : False,
  'sigma8_input' : False,
  'tau' : False,
  'n_s' : False,
  'A_s' : False,
  'omega_k' : False,
  'w' : False,
  'wa' : False,
  'omnuh2' : False,
  'massless_nu' : False,
  'massive_nu' : False,
  'b0' : False,
  'b1' : False,
  'b2' : False,
  'b3' : False,
  'b4' : False,
  'b5' : False,
  'bias_1' : False,
  'bias_2' : False,
  'bias_3' : False,
  'bias_4' : False,
  'bias_5' : False,
  'bias_6' : False,
  'm1' : False,
  'm2' : False,
  'm3' : False,
  'm4' : False,
  'm5' : False,
  'm6' : False,
  'A' : False,
  'alpha' : False,
  'a_planck' : False

}

prior = {
  
  'omega_m' : False,
  'h0' : False,
  'omega_b' : False,
  'sigma8_input' : False,
  'tau' : False,
  'n_s' : False,
  'A_s' : False,
  'omega_k' : False,
  'w' : False,
  'wa' : False,
  'omnuh2' : False,
  'massless_nu' : False,
  'massive_nu' : False,
  'b0' : False,
  'b1' : False,
  'b2' : False,
  'b3' : False,
  'b4' : False,
  'b5' : False,
  'bias_1' : False,
  'bias_2' : False,
  'bias_3' : False,
  'bias_4' : False,
  'bias_5' : False,
  'bias_6' : False,
  'm1' : False,
  'm2' : False,
  'm3' : False,
  'm4' : False,
  'm5' : False,
  'm6' : False,
  'A' : False,
  'alpha' : False,
  'a_planck' : False

}


dc1_params = {
  
  'omega_m' : (0.05, 0.3156, 0.6),
  'h0' : (0.4, 0.6727, 0.9),
  'omega_b' : (0.02, 0.0491685, 0.09),
  'sigma8_input' : (0.5, 0.831, 1.2),
  'tau' : (0.01, 0.08, 0.8),
  'n_s' : (0.84, 0.9645, 1.06),
  'A_s' : (1e-9, 2.1e-9, 3e-9),
  'omega_k' : (-1.0, 0.0, 1.0),
  'w' : (-2.1, -1.0, 0.0),
  'wa' : (-1.0, 0.0, 1.0),
  'omnuh2' : (0.0, 0.00065, 0.001),
  'massless_nu' : (1.0, 2.046, 4.0),
  'massive_nu' : (0, 1, 2),
  'b0' : (0.5, 1., 2.5),
  'b1' : (0.5, 1., 2.5),
  'b2' : (0.5, 1., 2.5),
  'b3' : (0.5, 1., 2.5),
  'b4' : (0.5, 1., 2.5),
  'b5' : (0.5, 1., 2.5),
  'bias_1' : (-0.1, 0., 0.1),
  'bias_2' : (-0.1, 0., 0.1),
  'bias_3' : (-0.1, 0., 0.1),
  'bias_4' : (-0.1, 0., 0.1),
  'bias_5' : (-0.1, 0., 0.1),
  'bias_6' : (-0.1, 0., 0.1),
  'm1' : (-0.1, 0., 0.1),
  'm2' : (-0.1, 0., 0.1),
  'm3' : (-0.1, 0., 0.1),
  'm4' : (-0.1, 0., 0.1),
  'm5' : (-0.1, 0., 0.1),
  'm6' : (-0.1, 0., 0.1),
  'A' : (-3., 1., 5.0),
  'alpha' : (-5., 0., 5.0),
  'a_planck' : (.5,1.,1.5)

}

dc1_priors = {
  
  'omega_m' : (0.3156, 0.5),
  'h0' : (0.6726, 0.2),
  'omega_b' : (0.0491685, 0.05),
  'sigma8_input' : (0.831, 0.25),
  'tau' : (0.08, 0.5),
  'n_s' : (0.9645, 0.2),
  'A_s' : (2.215e-9, 1e-9),
  'omega_k' : (0.0, 0.1),
  'w' : (-1.0, 1.),
  'wa' : (0.0, 1.),
  'omnuh2' : (0.00065, .0001),
  'massless_nu' : (2.046, 1.),
  'massive_nu' : (1, 1),
  'bias_1' : (0., 0.1),
  'bias_2' : (0., 0.1),
  'bias_3' : (0., 0.1),
  'bias_4' : (0., 0.1),
  'bias_5' : (0., 0.1),
  'bias_6' : (0., 0.1),
  'm1' : (0., 0.02),
  'm2' : (0., 0.02),
  'm3' : (0., 0.02),
  'm4' : (0., 0.02),
  'm5' : (0., 0.02),
  'm6' : (0., 0.02),
  'A' : (1., 1.0),
  'alpha' : (0., 1.),
  'a_planck' : (1.,.1)

}

def chi(z,omegam=0.27,H=100):
  from astropy import cosmology
  from astropy.cosmology import FlatLambdaCDM
  cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
  return cosmo.comoving_distance(z).value    

class run(object):

  @staticmethod
  def loop_submit():

    run.submit(label='dc1_sig8_Om_03',nodes=1,procs=32,hr=48,pts=1000,mneff=0.5,mntol=0.5,ia=False,pz=False,mbias=False,params=dc1_params,priors=dc1_priors,data='data/datavector_cosmosis.txt',cov='data/cov.npy',nofz='data/n_of_zs.txt')
    run.submit(label='dc1_sig8_Om_03',nodes=1,procs=32,hr=48,pts=1000,mneff=0.5,mntol=0.5,ia=False,pz=True,mbias=False,params=dc1_params,priors=dc1_priors,data='data/datavector_cosmosis.txt',cov='data/cov.npy',nofz='data/n_of_zs.txt')
    run.submit(label='dc1_sig8_Om_03',nodes=1,procs=32,hr=48,pts=1000,mneff=0.5,mntol=0.5,ia=False,pz=False,mbias=True,params=dc1_params,priors=dc1_priors,data='data/datavector_cosmosis.txt',cov='data/cov.npy',nofz='data/n_of_zs.txt')
    run.submit(label='dc1_sig8_Om_03',nodes=1,procs=32,hr=48,pts=1000,mneff=0.5,mntol=0.5,ia=False,pz=True,mbias=True,params=dc1_params,priors=dc1_priors,data='data/datavector_cosmosis.txt',cov='data/cov.npy',nofz='data/n_of_zs.txt')
    run.submit(label='dc1_sig8_Om_03',nodes=1,procs=32,hr=48,pts=1000,mneff=0.5,mntol=0.5,ia=True,pz=True,mbias=True,params=dc1_params,priors=dc1_priors,data='data/datavector_cosmosis.txt',cov='data/cov.npy',nofz='data/n_of_zs.txt')

    return

  @staticmethod
  def submit(label='dc1',nodes=1,procs=32,hr=48,pts=200,mneff=0.8,mntol=0.5,ia=False,pz=False,mbias=False,planck=False,tomobins=3,paramlist=['sigma8_input'],params=dc1_params,priors=dc1_priors,data='data/datavector_cosmosis.txt',cov='data/cov.npy',nofz='data/n_of_zs.txt',cldir='',resume=False,submit=True):
    """
    A wrapper to submit cosmosis runs. Currently works for my PBS environment. Will add alternate option to just print necessary commands to a file to run as desired.

    Use:

    ....


    """

    from popen2 import popen2
    import subprocess as sp
    import time

    for param in paramlist:
      vary[param]=True

    name=label
    sout=label+'_pz-'+str(pz)+'_m-'+str(mbias)+'_ia-'+str(ia)+'_planck-'+str(planck)
    outfile='out/'+sout+'.txt' #'multinest-'+str(pts)+'_'+str(mneff)+'_'+str(mntol)+
    mnoutfile='out/'+sout+'.multinest'
    spriors=config.cosmosiscosmodir+sout+'_priors.ini'
    sparams=config.cosmosiscosmodir+sout+'_values.ini'

    mods=r' '
    like=r'xipm '
    like=r'wl '
    if pz:
      mods+=r'photoz_bias '
      for i in xrange(tomobins):
        vary['bias_'+str(i+1)]=True
        prior['bias_'+str(i+1)]=True
    if ia:
      sia=r'T'
      mods+=r'linear_alignment shear_shear add_intrinsic '
      vary['A']=True
    else:
      sia=r'F'
      mods+=r'shear_shear '
    if mbias:
      mods+=r'shear_m_bias '
      for i in xrange(tomobins):
        vary['m'+str(i+1)]=True
        prior['m'+str(i+1)]=True
    if planck:
      # vary['a_planck']=True
      # prior['a_planck']=True
      mods+=r'planck'
      like+=r'planck2015 '

    print vary
    print prior

    make.values(params,vary,ia,pz,mbias,planck,tomobins,sparams)
    if make.priors(priors,prior,ia,pz,mbias,planck,tomobins,spriors)==0:
      spriors=''    

    if submit:

      p = sp.Popen('qsub', shell=True, bufsize=1, stdin=sp.PIPE, stdout=sp.PIPE, close_fds=True, cwd=config.cosmosiscosmodir)
      output,input = p.stdout, p.stdin

      job_string = """#!/bin/bash
      #PBS -l nodes=%s:ppn=%s
      #PBS -l walltime=%s:00:00
      #PBS -N %s
      #PBS -o %s.log
      #PBS -j oe
      #PBS -m abe 
      #PBS -M michael.troxel@manchester.ac.uk
      module use /home/zuntz/modules/module-files
      module load python
      module use /etc/modulefiles/
      cd /home/troxel/cosmosis/
      source my-source
      cd %s
      cd $PBS_O_WORKDIR
      export POINTS="%s"
      export MNEFF="%s"
      export MNTOL="%s"
      export OUTFILE="%s"
      export DATA="%s"
      export COV="%s"
      export NOFZ="%s"
      export IA="%s"
      export MODS="%s"
      export LIKE="%s"
      export PRIORS="%s"
      export PARAMS="%s"
      export CLDIR="%s"
      export MNOUT="%s"
      export MNRESUME="%s"
      mpirun -n %s cosmosis --mpi data_in.ini
      postprocess -o plots -p %s %s""" % (str(nodes),str(procs),str(hr),name,sout,config.cosmosiscosmodir,str(pts),str(mneff),str(mntol),outfile,data,cov,nofz,sia,mods,like,spriors,sparams,cldir,mnoutfile,str(resume),str(procs),sout,outfile)    
      output,outputerr=p.communicate(input=job_string)
      print job_string
      print output
    else:

      jobstring="""#!/bin/bash
      cd /home/troxel/cosmosis/
      source my-source
      cd %s
      cd $PBS_O_WORKDIR
      export POINTS="%s"
      export MNEFF="%s"
      export MNTOL="%s"
      export OUTFILE="%s"
      export DATA="%s"
      export COV="%s"
      export NOFZ="%s"
      export IA="%s"
      export MODS="%s"
      export LIKE="%s"
      export PRIORS="%s"
      export PARAMS="%s"
      export CLDIR="%s"
      export MNOUT="%s"
      export MNRESUME="%s"
      mpirun -n %s cosmosis --mpi data_in.ini
      postprocess -o plots -p %s %s""" % (config.cosmosiscosmodir,str(pts),str(mneff),str(mntol),outfile,data,cov,nofz,sia,mods,like,spriors,sparams,cldir,mnoutfile,str(resume),str(procs),sout,outfile)  
      with open('cosmosis_cosmo.submit','w') as f:
        f.write(jobstring)

    time.sleep(0.1)

    return

  @staticmethod
  def submit_pz_spec_test(pz0,test,testtype,bins=3,boot=False,cosmo=False,ia=False,nodes=1,procs=32,hr=48,params=dc1_params,priors=dc1_priors,submit=True,fillin=False):
    """
    A wrapper to submit cosmosis runs specifically for photo-z spec validation. Currently works for my PBS environment. Will add alternate option to just print necessary commands to a file to run as desired. Needs work. Currently works with Cls due to need of synthetic covariance, but will switch back to xi if used in WL analysis once covariances are available. Could also make it optional which (xi vs cl) to use.

    Use:

    ....


    """

    if pz0.pztype+'.fits.gz' not in os.listdir(config.pztestdir+test+'/nofz'):
      print 'Missing '+pz0.pztype+'.fits.gz'
    if 'notomo_'+pz0.pztype+'.fits.gz' not in os.listdir(config.pztestdir+test+'/nofz'):
      print 'Missing '+'notomo_'+pz0.pztype+'.fits.gz'
    if 'spec_'+pz0.pztype+'.fits.gz' not in os.listdir(config.pztestdir+test+'/nofz'):
      print 'Missing '+'spec_'+pz0.pztype+'.fits.gz'
    if 'spec_notomo_'+pz0.pztype+'.fits.gz' not in os.listdir(config.pztestdir+test+'/nofz'):
      print 'Missing '+'spec_notomo_'+pz0.pztype+'.fits.gz'
    if boot&hasattr(pz0,'boot'):
      for i in xrange(pz0.boot):
        if pz0.pztype+'_'+str(i)+'.fits.gz' not in os.listdir(config.pztestdir+test+'/nofz'):
          print 'Missing '+pz0.pztype+'_'+str(i)+'.fits.gz'
        if 'notomo_'+pz0.pztype+'_'+str(i)+'.fits.gz' not in os.listdir(config.pztestdir+test+'/nofz'):
          print 'Missing '+'notomo_'+pz0.pztype+'_'+str(i)+'.fits.gz'  

    if submit:
      from popen2 import popen2
      import subprocess as sp2
      import time

    if cosmo:
      procs=32
    else:
      procs=1

    if submit:
      p = sp2.Popen('qsub', shell=True, bufsize=1, stdin=sp2.PIPE, stdout=sp2.PIPE, close_fds=True, cwd=config.pztestdir)
      output,input = p.stdout, p.stdin

      jobstring0="""#!/bin/bash
      #PBS -l nodes=%s:ppn=%s
      #PBS -l walltime=%s:00:00
      #PBS -N %s
      #PBS -o %s.log
      #PBS -j oe
      #PBS -m abe 
      #PBS -M michael.troxel@manchester.ac.uk
      module use /home/zuntz/modules/module-files
      module load python
      module use /etc/modulefiles/
      cd /home/troxel/cosmosis/
      source my-source
      cd %s
      """ % (str(nodes),str(procs),str(hr),test+'_'+pz0.name+'_'+str(boot)+'_'+str(cosmo),test+'_'+pz0.name+'_'+str(boot)+'_'+str(cosmo),config.pztestdir)
    else:
      jobstring0="""#!/bin/bash
      cd %s
      %s
      cd %s
      """ % (config.cosmosisrootdir,config.cosmosissource,config.pztestdir)

    ellmin=200.
    ellmax=2000.
    nell=10

    if testtype=='bao':
      ellmin=10.
      ellmax=1000.
      nell=500

    params['sigma8_input']=(0.65, .8, .95)
    vary['sigma8_input']=False
    if cosmo:
      if testtype!='lss':
        vary['sigma8_input']=True
      if testtype!='wl':
        for i in range(bins):
          vary['b'+str(i)]=True
    elif (testtype=='wl')|(testtype=='lss'):
      testtype='tcp'

    if testtype=='lss':
      nzdatablocks="""'position'"""
      datablocks="""'galaxy_cl'"""
    if testtype=='bao':
      nzdatablocks="""'position'"""
      datablocks="""'galaxy_cl'"""
    if testtype=='wl':
      nzdatablocks="""'shear'"""
      datablocks="""'shear_cl'"""
    if (testtype=='tcp')|(testtype=='bao'):
      nzdatablocks="""'shear position'"""
      datablocks="""'shear_cl shear_galaxy_cl galaxy_cl'"""
    snz='nz_shear'
    lnz='nz_position'

    modules="""'consistency camb sigma8_rescale halofit growth extrapolate fits_nz"""
    priorsfile="""''"""

    area=5000.
    ngshear=8/bins
    ngshear=repr(" ".join(str(s) for s in ngshear*np.ones(bins)))
    nglss=-.12/bins
    nglss=repr(" ".join(str(s) for s in nglss*np.ones(bins)))
    nglss="""'.02 .04 .06'"""
    sige=0.25
    sige=repr(" ".join(str(s) for s in sige*np.ones(bins)))

    gbperbin='T'
    ss='F'
    sp='F'
    pp='F'
    ii='F'
    si='F'
    pi='F'
    if (testtype=='lss'):
      pp='position-position'
    elif testtype=='wl':
      ss='shear-shear'
      if ia:
        ii='shear-shear'
        si='shear-shear'
    elif (testtype=='tcp')|(testtype=='bao'):
      ss='shear-shear'
      sp='shear-position'
      pp='position-position'
      if ia:
        ii='shear-shear'
        si='shear-shear'
        pi='position-shear'
    if ia:
      modules+=""" IA stitch"""
    modules+=""" unbiased_galaxies pk_to_cl"""
    if ia:
      modules+=""" add_intrinsic"""
    if testtype!='wl':
      modules+=""" ggl_bias"""

    mnpoints=200
    mntolerance=0.5
    mnefficiency=0.8
    gridpoints=50
    if cosmo:
      gausscov='F'
      if testtype=='wl':
        sampler='grid'#'fisher'#
      else:
        sampler='multinest'
      if testtype=='lss':
        short='ggl_bias'
      else:
        short="""''"""
      modules+=""" 2pt_like'"""
      like='2pt'
    else:
      gausscov='T'
      sampler='test'
      if testtype=='bao':
        modules+=""" 2pt_matter"""
      modules+=""" save_c_ell_fits'"""
      like="""''"""
      short="""''"""


    jobstring=jobstring0
    if cosmo:
      if boot:
        ii2=-1
        for nofz in os.listdir(config.pztestdir+test+'/nofz'):
          if (pz0.name in nofz)&(pz0.name+'.fits.gz' not in nofz):
            if 'spec' not in nofz:
              infile=config.pztestdir+test+'/out/spec_'+nofz[:-8]+'.fits.gz'
              nzinfile=config.pztestdir+test+'/nofz/'+nofz[:-8]+'.fits.gz'
              outfile=config.pztestdir+test+'/out/'+testtype+'_spec_'+nofz[:-8]+'_'+nofz[:-8]+'.txt'
              savedir="""''"""#config.pztestdir+test+'/out/spec_'+nofz[:-8]+'_'+nofz[:-8]
              savefile=config.pztestdir+test+'/out/spec_'+nofz[:-8]+'_'+nofz[:-8]+'.fits.gz'
              valuesfile=config.pztestdir+test+'/ini/spec_'+nofz[:-8]+'_'+nofz[:-8]+'_values.ini'
              if fillin&(testtype+'_spec_'+nofz[:-8]+'_'+nofz[:-8]+'_means.txt' in os.listdir(config.pztestdir+test+'/out')):
                continue
              ii2+=1
              if 'notomo' in nofz:
                make.values(params,vary,False,False,False,False,1,valuesfile)
              else:
                make.values(params,vary,False,False,False,False,bins,valuesfile)
              jobstring+="""mpirun -n %s cosmosis --mpi %stcp.ini -p runtime.sampler=%s output.filename=%s test.save_dir=%s multinest.live_points=%s multinest.tolerance=%s multinest.efficiency=%s grid.nsample_dimension=%s pipeline.modules=%s pipeline.values=%s pipeline.priors=%s pipeline.likelihoods=%s pipeline.shortcut=%s pk_to_cl.ell_min=%s pk_to_cl.ell_max=%s pk_to_cl.n_ell=%s pk_to_cl.shear-shear=%s pk_to_cl.shear-position=%s pk_to_cl.position-position=%s pk_to_cl.intrinsic-intrinsic=%s pk_to_cl.shear-intrinsic=%s pk_to_cl.position-intrinsic=%s save_c_ell_fits.ell_min=%s save_c_ell_fits.ell_max=%s save_c_ell_fits.n_ell=%s save_c_ell_fits.shear_nz_name=%s save_c_ell_fits.position_nz_name=%s save_c_ell_fits.filename=%s save_c_ell_fits.survey_area=%s save_c_ell_fits.number_density_shear_bin=%s save_c_ell_fits.number_density_lss_bin=%s save_c_ell_fits.sigma_e_bin=%s ggl_bias.perbin=%s fits_nz.nz_file=%s fits_nz.data_sets=%s 2pt_like.data_file=%s 2pt_like.data_sets=%s 2pt_like.gaussian_covariance=%s 2pt_like.survey_area=%s 2pt_like.number_density_shear_bin=%s 2pt_like.number_density_lss_bin=%s 2pt_like.sigma_e_bin=%s
              """ % (procs,config.pztestdir,sampler,outfile,savedir,mnpoints,mntolerance,mnefficiency,gridpoints,modules,valuesfile,priorsfile,like,short,ellmin,ellmax,nell,ss,sp,pp,ii,si,pi,ellmin,ellmax,nell,snz,lnz,savefile,area,ngshear,nglss,sige,gbperbin,nzinfile,nzdatablocks,infile,datablocks,gausscov,area,ngshear,nglss,sige)
              jobstring+="""mpirun -n 1 postprocess %s -o %s -p %s --no-plots
              """ % (outfile,config.pztestdir+test+'/out',testtype+'_spec_'+nofz[:-8]+'_'+nofz[:-8])
              if (ii2==10)&submit:
                print jobstring
                output,outputerr=p.communicate(input=jobstring)
                time.sleep(0.1)
                ii2=-1
                p = sp2.Popen('qsub', shell=True, bufsize=1, stdin=sp2.PIPE, stdout=sp2.PIPE, close_fds=True, cwd=config.pztestdir)
                output,input = p.stdout, p.stdin
                jobstring=jobstring0
        if submit&(ii2!=-1):
          print jobstring
          output,outputerr=p.communicate(input=jobstring)
          time.sleep(0.1)
        else:
          with open('cosmosis_pz_boot-'+str(boot)+'_cosmo-'+str(cosmo)+'.submit','w') as f:
            f.write(jobstring)
      else:
        for nofz in os.listdir(config.pztestdir+test+'/nofz'):
          if (pz0.name+'.fits.gz' in nofz):
            if 'spec' not in nofz:
              infile=config.pztestdir+test+'/out/spec_'+nofz[:-8]+'.fits.gz'
              nzinfile=config.pztestdir+test+'/nofz/'+nofz[:-8]+'.fits.gz'
              outfile=config.pztestdir+test+'/out/'+testtype+'_spec_'+nofz[:-8]+'_'+nofz[:-8]+'.txt'
              savedir="""''"""#config.pztestdir+test+'/out/spec_'+nofz[:-8]+'_'+nofz[:-8]
              savefile=config.pztestdir+test+'/out/spec_'+nofz[:-8]+'_'+nofz[:-8]+'.fits.gz'
              valuesfile=config.pztestdir+test+'/ini/spec_'+nofz[:-8]+'_'+nofz[:-8]+'_values.ini'
              if 'notomo' in nofz:
                make.values(params,vary,False,False,False,False,1,valuesfile)
              else:
                make.values(params,vary,False,False,False,False,bins,valuesfile)
              jobstring+="""mpirun -n %s cosmosis --mpi %stcp.ini -p runtime.sampler=%s output.filename=%s test.save_dir=%s multinest.live_points=%s multinest.tolerance=%s multinest.efficiency=%s grid.nsample_dimension=%s pipeline.modules=%s pipeline.values=%s pipeline.priors=%s pipeline.likelihoods=%s pipeline.shortcut=%s pk_to_cl.ell_min=%s pk_to_cl.ell_max=%s pk_to_cl.n_ell=%s pk_to_cl.shear-shear=%s pk_to_cl.shear-position=%s pk_to_cl.position-position=%s pk_to_cl.intrinsic-intrinsic=%s pk_to_cl.shear-intrinsic=%s pk_to_cl.position-intrinsic=%s save_c_ell_fits.ell_min=%s save_c_ell_fits.ell_max=%s save_c_ell_fits.n_ell=%s save_c_ell_fits.shear_nz_name=%s save_c_ell_fits.position_nz_name=%s save_c_ell_fits.filename=%s save_c_ell_fits.survey_area=%s save_c_ell_fits.number_density_shear_bin=%s save_c_ell_fits.number_density_lss_bin=%s save_c_ell_fits.sigma_e_bin=%s ggl_bias.perbin=%s fits_nz.nz_file=%s fits_nz.data_sets=%s 2pt_like.data_file=%s 2pt_like.data_sets=%s 2pt_like.gaussian_covariance=%s 2pt_like.survey_area=%s 2pt_like.number_density_shear_bin=%s 2pt_like.number_density_lss_bin=%s 2pt_like.sigma_e_bin=%s
              """ % (procs,config.pztestdir,sampler,outfile,savedir,mnpoints,mntolerance,mnefficiency,gridpoints,modules,valuesfile,priorsfile,like,short,ellmin,ellmax,nell,ss,sp,pp,ii,si,pi,ellmin,ellmax,nell,snz,lnz,savefile,area,ngshear,nglss,sige,gbperbin,nzinfile,nzdatablocks,infile,datablocks,gausscov,area,ngshear,nglss,sige)
              jobstring+="""mpirun -n 1 postprocess %s -o %s -p %s --no-plots
              """ % (outfile,config.pztestdir+test+'/out',testtype+'_spec_'+nofz[:-8]+'_'+nofz[:-8])
        if submit:
          print jobstring
          output,outputerr=p.communicate(input=jobstring)
          time.sleep(0.1)
        else:
          with open('cosmosis_pz_boot-'+str(boot)+'_cosmo-'+str(cosmo)+'.submit','w') as f:
            f.write(jobstring)
    else:
      if boot:
        ii2=-1
        for nofz in os.listdir(config.pztestdir+test+'/nofz'):
          if (pz0.name in nofz)&(pz0.name+'.fits.gz' not in nofz):
            infile=config.pztestdir+test+'/nofz/'+nofz[:-8]+'.fits.gz'
            nzinfile=config.pztestdir+test+'/nofz/'+nofz[:-8]+'.fits.gz'
            outfile=config.pztestdir+test+'/out/'+nofz[:-8]+'.txt'
            savedir="""''"""#config.pztestdir+test+'/out/'+nofz[:-8]
            savefile=config.pztestdir+test+'/out/'+nofz[:-8]+'.fits.gz'
            valuesfile=config.pztestdir+test+'/ini/'+nofz[:-8]+'_values.ini'
            if fillin&(nofz in os.listdir(config.pztestdir+test+'/out')):
              continue
            ii2+=1
            if 'notomo' in nofz:
              make.values(params,vary,False,False,False,False,1,valuesfile)
            else:
              make.values(params,vary,False,False,False,False,bins,valuesfile)
            jobstring+="""cosmosis %stcp.ini -p runtime.sampler=%s output.filename=%s test.save_dir=%s multinest.live_points=%s multinest.tolerance=%s multinest.efficiency=%s grid.nsample_dimension=%s pipeline.modules=%s pipeline.values=%s pipeline.priors=%s pipeline.likelihoods=%s pipeline.shortcut=%s pk_to_cl.ell_min=%s pk_to_cl.ell_max=%s pk_to_cl.n_ell=%s pk_to_cl.shear-shear=%s pk_to_cl.shear-position=%s pk_to_cl.position-position=%s pk_to_cl.intrinsic-intrinsic=%s pk_to_cl.shear-intrinsic=%s pk_to_cl.position-intrinsic=%s save_c_ell_fits.ell_min=%s save_c_ell_fits.ell_max=%s save_c_ell_fits.n_ell=%s save_c_ell_fits.shear_nz_name=%s save_c_ell_fits.position_nz_name=%s save_c_ell_fits.filename=%s save_c_ell_fits.survey_area=%s save_c_ell_fits.number_density_shear_bin=%s save_c_ell_fits.number_density_lss_bin=%s save_c_ell_fits.sigma_e_bin=%s ggl_bias.perbin=%s fits_nz.nz_file=%s fits_nz.data_sets=%s 2pt_like.data_file=%s 2pt_like.data_sets=%s 2pt_like.gaussian_covariance=%s 2pt_like.survey_area=%s 2pt_like.number_density_shear_bin=%s 2pt_like.number_density_lss_bin=%s 2pt_like.sigma_e_bin=%s
              """ % (config.pztestdir,sampler,outfile,savedir,mnpoints,mntolerance,mnefficiency,gridpoints,modules,valuesfile,priorsfile,like,short,ellmin,ellmax,nell,ss,sp,pp,ii,si,pi,ellmin,ellmax,nell,snz,lnz,savefile,area,ngshear,nglss,sige,gbperbin,nzinfile,nzdatablocks,infile,datablocks,gausscov,area,ngshear,nglss,sige)
            if (ii2==10)&submit:
              print jobstring
              output,outputerr=p.communicate(input=jobstring)
              time.sleep(0.1)
              ii2=-1
              p = sp2.Popen('qsub', shell=True, bufsize=1, stdin=sp2.PIPE, stdout=sp2.PIPE, close_fds=True, cwd=config.pztestdir)
              output,input = p.stdout, p.stdin
              jobstring=jobstring0
        if submit&(ii2!=-1):
          print jobstring
          output,outputerr=p.communicate(input=jobstring)
          time.sleep(0.1)
        else:
          with open('cosmosis_pz_boot-'+str(boot)+'_cosmo-'+str(cosmo)+'.submit','w') as f:
            f.write(jobstring)
      else:
        for nofz in os.listdir(config.pztestdir+test+'/nofz'):
          if (pz0.name+'.fits.gz' in nofz):
            infile=config.pztestdir+test+'/nofz/'+nofz[:-8]+'.fits.gz'
            nzinfile=config.pztestdir+test+'/nofz/'+nofz[:-8]+'.fits.gz'
            outfile=config.pztestdir+test+'/out/'+nofz[:-8]+'.txt'
            savedir=config.pztestdir+test+'/out/'+nofz[:-8]#"""''"""
            if testtype=='bao':
              savefile=config.pztestdir+test+'/out/'+nofz[:-8]+'_bao.fits.gz'
            savefile=config.pztestdir+test+'/out/'+nofz[:-8]+'.fits.gz'
            valuesfile=config.pztestdir+test+'/ini/'+nofz[:-8]+'_values.ini'
            if 'notomo' in nofz:
              make.values(params,vary,False,False,False,False,1,valuesfile)
            else:
              make.values(params,vary,False,False,False,False,bins,valuesfile)
            jobstring+="""cosmosis %stcp.ini -p runtime.sampler=%s output.filename=%s test.save_dir=%s multinest.live_points=%s multinest.tolerance=%s multinest.efficiency=%s grid.nsample_dimension=%s pipeline.modules=%s pipeline.values=%s pipeline.priors=%s pipeline.likelihoods=%s pipeline.shortcut=%s pk_to_cl.ell_min=%s pk_to_cl.ell_max=%s pk_to_cl.n_ell=%s pk_to_cl.shear-shear=%s pk_to_cl.shear-position=%s pk_to_cl.position-position=%s pk_to_cl.intrinsic-intrinsic=%s pk_to_cl.shear-intrinsic=%s pk_to_cl.position-intrinsic=%s save_c_ell_fits.ell_min=%s save_c_ell_fits.ell_max=%s save_c_ell_fits.n_ell=%s save_c_ell_fits.shear_nz_name=%s save_c_ell_fits.position_nz_name=%s save_c_ell_fits.filename=%s save_c_ell_fits.survey_area=%s save_c_ell_fits.number_density_shear_bin=%s save_c_ell_fits.number_density_lss_bin=%s save_c_ell_fits.sigma_e_bin=%s ggl_bias.perbin=%s fits_nz.nz_file=%s fits_nz.data_sets=%s 2pt_like.data_file=%s 2pt_like.data_sets=%s 2pt_like.gaussian_covariance=%s 2pt_like.survey_area=%s 2pt_like.number_density_shear_bin=%s 2pt_like.number_density_lss_bin=%s 2pt_like.sigma_e_bin=%s
              """ % (config.pztestdir,sampler,outfile,savedir,mnpoints,mntolerance,mnefficiency,gridpoints,modules,valuesfile,priorsfile,like,short,ellmin,ellmax,nell,ss,sp,pp,ii,si,pi,ellmin,ellmax,nell,snz,lnz,savefile,area,ngshear,nglss,sige,gbperbin,nzinfile,nzdatablocks,infile,datablocks,gausscov,area,ngshear,nglss,sige)
        if submit:
          print jobstring
          output,outputerr=p.communicate(input=jobstring)
          time.sleep(0.1)
        else:
          with open('cosmosis_pz_boot-'+str(boot)+'_cosmo-'+str(cosmo)+'.submit','w') as f:
            f.write(jobstring)

    return


  @staticmethod
  def submit_rho_leakage_test(
    deltaxi,
    optdict,
    test='rho_leakage',
    workdir='',
    procs=1,
    hr=3,
    cosmosisrootdir='',
    cosmosissource='source '+config.cosmosisnerscdir+'setup-cosmosis-nersc-edison', 
    inifile='cosmosis.ini',
    valuefile='values.ini',
    submit=False):
    """
    A wrapper to submit cosmosis runs specifically for delta xi+ systematic contamination. Could also make it optional which (xi vs cl) to use.

    Use:

    ....


    """

    def to_fits(theta,xip,xim,fileout,filein=workdir+'lsst_default.fits'):

      # From github.com/joezuntz/2point
      import twopoint

      # Setup xi extensions
      xipext = twopoint.SpectrumMeasurement(
        'xip', # hdu name
        (np.ones(len(theta)),np.ones(len(theta))), # tomographic bins
        (twopoint.Types.galaxy_shear_plus_real, twopoint.Types.galaxy_shear_plus_real), # type of 2pt statistic
        ('nofz', 'nofz'), # associated nofz
        "SAMPLE", # window function
        np.arange(len(theta)), # id
        xip, # value
        angle=theta, # theta value
        angle_unit='arcmin') # units

      ximext = twopoint.SpectrumMeasurement(
        'xim',
        (np.ones(len(theta)),np.ones(len(theta))), 
        (twopoint.Types.galaxy_shear_minus_real, twopoint.Types.galaxy_shear_minus_real), 
        ('nofz', 'nofz'), 
        "SAMPLE", 
        np.arange(len(theta)), 
        xim,
        angle=theta,
        angle_unit='arcmin') 

      # write to fits file
      from astropy.io import fits
      data=fits.open(filein)
      nofz=twopoint.NumberDensity.from_fits(data['nofz'])
      covmat=twopoint.CovarianceMatrixInfo.from_fits(data['covmat'])
      data=twopoint.TwoPointFile([xipext,ximext],[nofz],None,covmat)
      data.to_fits(fileout, clobber=True)

      return

    # Check for required cosmosis files
    if inifile not in os.listdir(workdir):
      print 'Missing ini file'
    if valuefile not in os.listdir(workdir):
      print 'Missing value file'

    if submit:
      import subprocess as sp

    if submit:
      # setup header of submit file and assign subprocess
      p = sp.Popen('qsub', shell=True, bufsize=1, stdin=sp.PIPE, stdout=sp.PIPE, close_fds=True, cwd=workdir)
      output,input = p.stdout, p.stdin

      jobstring0="""#!/bin/bash
      #PBS -l nodes=1:ppn=%s
      #PBS -l walltime=%s:00:00
      #PBS -N %s
      #PBS -o %s.log
      #PBS -j oe
      #PBS -m abe 
      #PBS -M michael.troxel@manchester.ac.uk
      module use /home/zuntz/modules/module-files
      module load python
      module use /etc/modulefiles/
      cd /home/troxel/cosmosis/
      source my-source
      cd %s
      """ % (str(procs),str(hr),test,test,workdir)
    else:
      # setup header of bash script
      jobstring0="""#!/bin/bash
      cd %s
      %s
      cd %s
      """ % (cosmosisrootdir,cosmosissource,workdir)

    # initialise final jobstring
    jobstring=jobstring0

    # call theory xip, xim for testing
    c0=_cosmosis(infile=workdir+'cosmosis.ini',fitsfile=workdir+'lsst_default.fits',values=workdir+'values_fixed.ini')
    c0.xi(1,1,theta=deltaxi['meanr'])

    # Check for and make output dir
    try:
      os.listdir(workdir+test)
    except:
      os.mkdir(workdir+test)

    # write modified xip, xim to twopoint fits file for cosmosis
    to_fits(deltaxi['meanr'],c0.xip+deltaxi['xi'],c0.xim,workdir+test+'/xi_plus_dxi.fits')

    # setup file paths
    infile=workdir+'cosmosis.ini'
    savedir="""''"""
    outfile=workdir+test+'/output.txt'
    nzinfile=workdir+test+'/xi_plus_dxi.fits'

    # setup data block names
    datablocks="""'xip xim'"""
    nzdatablocks='nofz'

    # setup cosmosis call
    jobstring+="""mpirun -n %s cosmosis --mpi %s -p output.filename=%s test.save_dir=%s fits_nz.nz_file=%s fits_nz.data_sets=%s 2pt_like.data_file=%s 2pt_like.data_sets=%s
    """ % (procs,infile,outfile,savedir,nzinfile,nzdatablocks,nzinfile,datablocks)
    jobstring+="""mpirun -n 1 postprocess %s -o %s 
    """ % (outfile,workdir+test)#--no-plots

    # submit job or print bash script
    if submit:
      print jobstring
      output,outputerr=p.communicate(input=jobstring)
      time.sleep(0.1)
    else:
      print jobstring
      with open('cosmosis_rho.submit','w') as f:
        f.write(jobstring)

    return

class make(object):

  @staticmethod
  def values(params,vary,ia,pz,mbias,planck,tomobins,sfile):
    """
    Writes values.ini files for cosmosis run submitted via run class.
    """

    n='\n'

    with open(sfile,'w') as f:
      f.write('[cosmological_parameters]'+n)
      for x in ['omega_m','h0','omega_b','sigma8_input','tau','n_s','A_s','omega_k','w','wa']:#,'omnuh2','massless_nu','massive_nu'
        if (vary.get(x)):
          f.write(x+' = '+str(params.get(x)[0])+' '+str(params.get(x)[1])+' '+str(params.get(x)[2])+n)
        else:
          f.write(x+' = '+str(params.get(x)[1])+n)
      f.write('[ggl_bias_vals]'+n)
      for x in ['b0','b1','b2','b3','b4','b5'][:tomobins]:
        if (vary.get(x)):
          f.write(x+' = '+str(params.get(x)[0])+' '+str(params.get(x)[1])+' '+str(params.get(x)[2])+n)
        else:
          f.write(x+' = '+str(params.get(x)[1])+n)
      if pz:
        f.write('\n[wl_photoz_errors]'+n)
        for x in ['bias_1','bias_2','bias_3','bias_4','bias_5','bias_6'][:tomobins]:
          if (vary.get(x)):
            f.write(x+' = '+str(params.get(x)[0])+' '+str(params.get(x)[1])+' '+str(params.get(x)[2])+n)
          else:
            f.write(x+' = '+str(params.get(x)[1])+n)
      if mbias:
        f.write('\n[shear_calibration_parameters]'+n)
        for x in ['m1','m2','m3','m4','m5','m6'][:tomobins]:
          if (vary.get(x)):
            f.write(x+' = '+str(params.get(x)[0])+' '+str(params.get(x)[1])+' '+str(params.get(x)[2])+n)
          else:
            f.write(x+' = '+str(params.get(x)[1])+n)
      if ia:
        f.write('\n[intrinsic_alignment_parameters]'+n)
        for x in ['A']:
          if (vary.get(x)):
            f.write(x+' = '+str(params.get(x)[0])+' '+str(params.get(x)[1])+' '+str(params.get(x)[2])+n)
          else:
            f.write(x+' = '+str(params.get(x)[1])+n)
        # f.write('\n[ia_z_field]'+n)
        # for x in ['alpha']:
        #   if (vary.get(x)):
        #     f.write(x+' = '+str(params.get(x)[0])+' '+str(params.get(x)[1])+' '+str(params.get(x)[2])+n)
        #   else:
        #     f.write(x+' = '+str(params.get(x)[1])+n)
      if planck:
        f.write('\n[planck]'+n)
        for x in ['a_planck']:
          if (vary.get(x)):
            f.write(x+' = '+str(params.get(x)[0])+' '+str(params.get(x)[1])+' '+str(params.get(x)[2])+n)
          else:
            f.write(x+' = '+str(params.get(x)[1])+n)

    return

  @staticmethod
  def priors(params,prior,ia,pz,mbias,planck,tomobins,sfile):
    """
    Writes priors.ini files for cosmosis run submitted via run class.
    """

    n='\n'
    cnt=0

    with open(sfile,'w') as f:
      f.write('[cosmological_parameters]'+n)
      for x in ['omega_m','h0','omega_b','sigma8_input','tau','n_s','A_s','omega_k','w','wa']:#,'omnuh2','massless_nu','massive_nu'
        if (prior.get(x)):
          cnt+=1
          f.write(x+' = gaussian '+str(params.get(x)[0])+' '+str(params.get(x)[1])+n)
      if pz:
        f.write('\n[wl_photoz_errors]'+n)
        for x in ['bias_1','bias_2','bias_3','bias_4','bias_5','bias_6'][:tomobins]:
          if (prior.get(x)):
            cnt+=1
            f.write(x+' = gaussian '+str(params.get(x)[0])+' '+str(params.get(x)[1])+n)
      if mbias:
        f.write('\n[shear_calibration_parameters]'+n)
        for x in ['m1','m2','m3','m4','m5','m6'][:tomobins]:
          if (prior.get(x)):
            cnt+=1
            f.write(x+' = gaussian '+str(params.get(x)[0])+' '+str(params.get(x)[1])+n)
      if ia:
        f.write('\n[intrinsic_alignment_parameters]'+n)
        for x in ['A']:
          if (prior.get(x)):
            cnt+=1
            f.write(x+' = gaussian '+str(params.get(x)[0])+' '+str(params.get(x)[1])+n)
        # f.write('\n[ia_z_field]'+n)
        # for x in ['alpha']:
        #   if (prior.get(x)):
        #     cnt+=1
        #     f.write(x+' = gaussian '+str(params.get(x)[0])+' '+str(params.get(x)[1])+n)
      if planck:
        f.write('\n[planck]'+n)
        for x in ['a_planck']:
          if (prior.get(x)):
            cnt+=1
            f.write(x+' = gaussian '+str(params.get(x)[0])+' '+str(params.get(x)[1])+n)

    return cnt

  @staticmethod
  def nofz(pz0,test,zmax=None,zbins=None):
    """
    Writes n(z) files in cosmosis format for photo-z spec validation testing.
    """

    test+='_'+str(pz0.tomo-1)
    if pz0.wt:
      test+='_weighted'

    if not os.path.exists(config.pztestdir+test):
      os.makedirs(config.pztestdir+test)
    if not os.path.exists(config.pztestdir+test+'/nofz'):
      os.makedirs(config.pztestdir+test+'/nofz')
    if not os.path.exists(config.pztestdir+test+'/out'):
      os.makedirs(config.pztestdir+test+'/out')
    if not os.path.exists(config.pztestdir+test+'/ini'):
      os.makedirs(config.pztestdir+test+'/ini')

    if zmax is None:
      mask=np.ones(len(pz0.bin)).astype(bool)
    else:
      mask=pz0.bin<zmax

    # if zbins is None:
    #   zz=pz0.bin[mask]
    #   pzz=pz0.pz[:,mask]
    # else:
    #   if zmax is None:
    #     print 'need zmax'
    #     return
    #   else:
    #     zz=np.linspace(np.min(pz0.bin),zmax,zbins)
    #     pzz=make.modify_pdf(pz0,nbins=zbins,zmax=zmax)

    make.fits(pz0.bin,pz0.pzlens[0,:],pz0.bin,pz0.pzsource[0,:],config.pztestdir+test+'/nofz/notomo_'+pz0.name+'.fits.gz')
    make.fits(pz0.bin,pz0.pzlens[1:,:],pz0.bin,pz0.pzsource[1:,:],config.pztestdir+test+'/nofz/'+pz0.name+'.fits.gz')

    if hasattr(pz0,'speclens')&hasattr(pz0,'specsource'):

      make.fits(pz0.bin,pz0.speclens[0,:],pz0.bin,pz0.specsource[0,:],config.pztestdir+test+'/nofz/spec_notomo_'+pz0.name+'.fits.gz')
      make.fits(pz0.bin,pz0.speclens[1:,:],pz0.bin,pz0.specsource[1:,:],config.pztestdir+test+'/nofz/spec_'+pz0.name+'.fits.gz')

    if hasattr(pz0,'boot')&hasattr(pz0,'bootspeclens')&hasattr(pz0,'bootspecsource'):

      for j in xrange(pz0.boot):

        make.fits(pz0.bin,pz0.bootspeclens[0,j,:],pz0.bin,pz0.bootspecsource[0,j,:],config.pztestdir+test+'/nofz/spec_notomo_'+pz0.name+'_'+str(j)+'.fits.gz')
        make.fits(pz0.bin,pz0.bootspeclens[1:,j,:],pz0.bin,pz0.bootspecsource[1:,j,:],config.pztestdir+test+'/nofz/spec_'+pz0.name+'_'+str(j)+'.fits.gz')
        make.fits(pz0.bin,pz0.bootpzlens[0,j,:],pz0.bin,pz0.bootpzsource[0,j,:],config.pztestdir+test+'/nofz/notomo_'+pz0.name+'_'+str(j)+'.fits.gz')
        make.fits(pz0.bin,pz0.bootpzlens[1:,j,:],pz0.bin,pz0.bootpzsource[1:,j,:],config.pztestdir+test+'/nofz/'+pz0.name+'_'+str(j)+'.fits.gz')

    return

  @staticmethod
  def fits(zl,pzl,zs,pzs,filename):

    try:
      os.remove(filename)
      print 'Removing file ',filename
    except OSError:
      pass

    fits=fio.FITS(filename,'rw')

    hdr={'extname':'NZ_POSITION','NZDATA':True}
    out=np.empty(np.shape(zl),dtype=[('Z_MID','f8')])
    out['Z_MID']=zl
    fits.write(out,header=hdr)
    fits[-1].insert_column('Z_LOW',np.abs(zl-(zl[1]-zl[0])/2.))
    fits[-1].insert_column('Z_HIGH',zl+(zl[1]-zl[0])/2.)
    if 'notomo' in filename:
      fits[-1].insert_column('BIN1',pzl)
    else:
      for i in range(len(pzl)):
        fits[-1].insert_column('BIN'+str(i+1),pzl[i,:])

    hdr={'extname':'NZ_SHEAR','NZDATA':True}
    out=np.empty(np.shape(zs),dtype=[('Z_MID','f8')])
    out['Z_MID']=zs
    fits.write(out,header=hdr)
    fits[-1].insert_column('Z_LOW',np.abs(zs-(zs[1]-zs[0])/2.))
    fits[-1].insert_column('Z_HIGH',zs+(zs[1]-zs[0])/2.)
    if 'notomo' in filename:
      fits[-1].insert_column('BIN1',pzs)
    else:
      for i in range(len(pzs)):
        fits[-1].insert_column('BIN'+str(i+1),pzs[i,:])

    fits.close()

    return


  @staticmethod
  def modify_pdf(pz0,nbins=200,zmax=3.0):

    from scipy import array

    def extrap1d(i):
      xs = i.x
      ys = i.y
      
      def x0(x):
          if x < xs[0]:
              return ys[0]+(x-xs[0])*(ys[1]-ys[0])/(xs[1]-xs[0])
          elif x > xs[-1]:
              return ys[-1]+(x-xs[-1])*(ys[-1]-ys[-2])/(xs[-1]-xs[-2])
          else:
              return i(x)
      
      def func(xs):
          return array(map(x0, array(xs)))
      
      return func

    tmp=np.zeros((pz0.tomo,nbins))
    for i in range(len(pz0.pz)):

      f=scipy.interpolate.interp1d(pz0.bin,pz0.pz[i],kind='cubic',fill_value=0.)
      f2=extrap1d(f)
      tmp[i]=f2(np.linspace(np.min(pz0.bin),zmax,nbins))

    return tmp


class _cosmosis(object):

  def __init__(self,infile='/home/troxel/destest/params.ini',fitsfile=None,values=None):
    from cosmosis.runtime.config import Inifile
    from cosmosis.runtime.pipeline import LikelihoodPipeline

    ini=Inifile(infile)
    if fitsfile is not None:
      ini.set('fits_nz', 'nz_file', fitsfile)
      ini.set('2pt_like', 'data_file', fitsfile)
    if values is not None:
      ini.set('pipeline', 'values', values)
    ini.set('pipeline','modules',ini.get('pipeline','modules').replace('2pt_like',''))
    ini.set('runtime','sampler','test')
    print ini.get('pipeline', 'values')
    self.pipeline=LikelihoodPipeline(ini)
    self.data=self.pipeline.run_parameters([])

  def cls(self,i,j,ell=None,interpout=False):

    ell0=self.data['shear_cl','ell']
    cl0=self.data['shear_cl','bin_'+str(i)+'_'+str(j)]

    if ell is None:
      self.ell=ell0
      self.cl=cl0
    else:
      f=scipy.interpolate.interp1d(ell0,cl0)
      self.ell=ell
      self.cl=f(ell)

    if interpout:
      return f
    else:
      return

  def xi(self,i,j,theta=None):

    theta0=self.data['shear_xi','theta']
    xip0=self.data['shear_xi','xiplus_'+str(i)+'_'+str(j)]
    xim0=self.data['shear_xi','ximinus_'+str(i)+'_'+str(j)]

    if theta is None:
      self.theta=theta0/np.pi*180.*60.
      self.xip=xip0
      self.xim=xim0
    else:
      theta=theta*np.pi/180./60.
      f=scipy.interpolate.interp1d(theta0,xip0)
      f2=scipy.interpolate.interp1d(theta0,xim0)
      self.theta=theta
      self.xip=f(theta)
      self.xim=f2(theta)

    return

  def xiobs(self,bandpowers):

    f=scipy.interpolate.interp1d(self.theta,self.xip)
    f2=scipy.interpolate.interp1d(self.theta,self.xim)
    def func(t,f,i):
      return bandpowers.window_theta_geometric(t,i)*f(t)

    self.xipobs=np.zeros(bandpowers.nt)
    self.ximobs=np.zeros(bandpowers.nt)
    for i in range(bandpowers.nt):
      self.xipobs[i]=scipy.integrate.quad(func,bandpowers.tmin[i],bandpowers.tmax[i],args=(f,i))[0]
      self.ximobs[i]=scipy.integrate.quad(func,bandpowers.tmin[i],bandpowers.tmax[i],args=(f2,i))[0]

    return
