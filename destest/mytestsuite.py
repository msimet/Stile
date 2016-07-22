# cd /global/cscratch1/sd/troxel/destest
# source /project/projectdirs/cmb/modules/hpcports_NERSC.sh
# source /scratch2/scratchdirs/zuntz/stack/setup

import numpy as np
import os
if "NERSC_HOST" not in os.environ:
  import kmeans_radec as km
else:
  print 'No kmeans'

import src.catalog as catalog
import src.config as config
import src.fig as fig
import src.lin as lin
import src.sys_split as sys_split
import src.corr as corr
import src.field as field
import src.pz as pz
import sys
import fitsio as fio
import healpy as hp
import src.cosmo as cosmo

import time
from multiprocessing import Pool

# Shear tests

def summary_tests(cat,mask):
  lin.summary_stats.i3_flags_dist(cat,mask=mask)
  lin.summary_stats.e1_psf_stats(cat,mask=mask)
  return

def linear_tests(cat,cols,flagcols,mask):
  lin.hist.hist_tests(cols,cat,mask=mask)
  lin.hist.hist_2D_tests(cols,cols,cat,mask=mask)
  lin.footprint.hexbin_tests(cols,cat,mask=mask)
  lin.footprint.footprint_tests(cat,[],mask=mask,bins=100,label='All')
  lin.footprint.footprint_tests(cat,flagcols,mask=mask,bins=100,label='')
  return

def tile_tests(cat,vals,mask):
  lin.summary_stats.tile_stats(cat,vals,mask=mask)
  lin.hist.tile_tests(vals,cat)
  lin.hist.tile_tests_2D(vals,vals,cat)
  lin.footprint.tile_tests(vals,cat,mask=mask)
  return

def field_tests(epochcat,mask):
  field.field.footprint(epochcat,mask=mask)
  field.field.whisker(epochcat,mask=mask)
  field.field.whisker_chip(i3epoch,mask=epochmask)
  return

def split_tests(cat,lenscat,cols,mask):
  sys_split.split.cat_splits_lin(cols,cat,mask=mask)
  sys_split.split.cat_splits_2pt(cols,cat,lenscat,mask=mask)
  return

def corr_tests(cat,mask):
  corr.xi_2pt.xi_2pt(cat,corr='GG',maska=mask,plot=True)
  corr.xi_2pt.xi_2pt(cat,corr='GG',ga='psf',maska=mask,plot=True)
  corr.xi_2pt.xi_2pt(cat,cat,corr='GG',gb='psf',maska=mask,plot=True)
  return

def single_tests(epochcat,epochmask,cat,mask,lenscat):
  vals=['e1','e2','psf1','psf2','psffwhm']
  summary_tests(cat,mask)
  tile_tests(cat,vals,mask)
  cols=['psffwhm','rgp','rad','psf1','psf2','snr','e1','e2','evals','iter','ra_off','dec_off','flux','chi2pix','invfluxfrac','resmin','resmax','pos','psfpos','airmass','fwhm','maglimit','skybrite','skysigma','dpsf']
  flagcols=['info','error']
  linear_tests(cat,cols,flagcols,mask)
  cols=['psffwhm','rgp','rad','psf1','psf2','snr','e1','e2','evals','iter','ra_off','dec_off','flux','chi2pix','invfluxfrac','resmin','resmax','psfpos','airmass','fwhm','maglimit','skybrite','skysigma','dpsf']
  split_tests(cat,lenscat,cols,mask)
  field_tests(epochcat,epochmask)
  corr_tests(cat,mask)
  return

# def pair_tests(cat,cat2,mask,mask2,match_col):
#   vals=['e1','e2','psf1','psf2','psffwhm']
#   lin.hist.tile_tests_2D(vals,vals,cat,cat2=cat2)
#   cols=['psffwhm','rgp','rad','psf1','psf2','snr','e1','e2','evals','iter','ra_off','dec_off','flux','chi2pix','invfluxfrac','resmin','resmax','psfpos','airmass','fwhm','maglimit','skybrite','skysigma','dpsf']
#   cols=['psffwhm','rgp','skysigma','dpsf']
#   lin.hist.hist_2D_tests(cols,cols,cat,cat2=cat2,mask=mask,mask2=mask2,match_col=match_col)
#   return

# import fitsio as fio
# tmp=fio.FITS('filename')[-1].read()
# tmp=tmp['HPIX'][tmp['FRACGOOD']>.75]
# array[mask]

if __name__ == '__main__':

  if int(sys.argv[1])==0:
    catalog.CatalogMethods.download_cat_desdm('*',name='gold',table='NSEVILLA.Y1A1_GOLD_1_0_1',dir='/share/des/disc2/y1/gold_v101/',order=False,start=129)
    sys.exit()
  elif int(sys.argv[1])==1:
    catalog.CatalogMethods.download_cat_desdm('coadd_objects_id,mag_auto_g,mag_auto_i,mag_auto_r,mag_auto_z,mag_auto_y,magerr_auto_g,magerr_auto_i,magerr_auto_r,magerr_auto_z,magerr_auto_y,weights,z_spec',name='spec_cat',table='nsevilla.y1a1_train_valid_12_11_15_gold',dir='/home/troxel/',order=False,num=0)
    sys.exit()
  elif int(sys.argv[1])==2:
    import easyaccess as ea
    connection=ea.connect()
    cursor=connection.cursor()
    query="""SELECT * FROM coaddtile"""
    connection.query_and_save(query,'coadd_tiles2.fits')
    sys.exit()
  elif int(sys.argv[1])==3:
    field.field.build_special_points(int(sys.argv[2]))
    sys.exit()
  elif int(sys.argv[1])==4:
    cols=['stamp','nexp','chi2pix','psffwhm','coadd','like','flux','rgp','dec','evals','rad','dec_off','ra_off','fluxfrac','psf1','psf2','modmax','modmin','ra','resmax','tile','maskfrac','snr','resmin','e1','e2','iter','pz','r','g','i','z']
    i3=catalog.CatalogStore('y1_i3_r_v1',cattype='i3',cols=cols,cutfunc=catalog.CatalogMethods.i3_cuts(),catfile='/share/des/disc2/y1/im3shape/single_band/r/y1v1/combined_r.fits',release='y1')
    sys_split.split_methods.load_maps(i3)
    rm10=catalog.CatalogStore('y1_rm10',cutfunc=catalog.CatalogMethods.default_rm_cuts(),cattype='gal',cols=['coadd','ra','dec','zp'],catfile=config.redmagicdir+'y1a1_gold_1.0-pre2_run_redmapper_v6.4.4_redmagic_0.5-10.fit',release='y1')

    # cols=['coadd','expnum','psf1_exp','psf2_exp','ccd','row','col','e1','e2'      


    # epochmask=np.in1d(i3epoch.coadd,i3.coadd,assume_unique=False)

    vals=['e1','e2','e','psf1','psf2','psffwhm']
    #lin.summary_stats.i3_flags_dist(i3)
    lin.summary_stats.e1_psf_stats(i3)

    cols=['chi2pix','psffwhm','e','coadd','nlike','flux','rgp','dec','evals','rad','dec_off','ra_off','fluxfrac','psf1','psf2','modmax','modmin','ra','resmax','snr','resmin','e1','e2','pz','r','g','i','z','airmass','fwhm','maglimit','skybrite','skysigma','maskfrac']
    lin.hist.hist_tests(cols,i3)
    lin.hist.hist_2D_tests(cols,cols,i3)
    lin.footprint.hexbin_tests(cols,i3)
    lin.footprint.footprint_tests(i3,[],bins=100,label='All')

    lin.summary_stats.tile_stats(i3,vals)
    lin.hist.tile_tests(vals,i3)
    lin.hist.tile_tests_2D(vals,vals,i3)
    lin.footprint.tile_tests(vals,i3)

    # field.field.footprint(i3epoch,mask=epochmask)
    # field.field.whisker(i3epoch,mask=epochmask)
    # field.field.whisker_chip(i3epoch,mask=epochmask)

    sys_split.split.cat_splits_lin(cols,i3)
    sys_split.split.cat_splits_2pt(cols,i3,rm10)

    sys.exit()
  elif int(sys.argv[1])==5:
    field.field.build_special_points_fits()
    sys.exit()
  elif int(sys.argv[1])==6:
    import fitsio as fio
    import healpy as hp
    # maskpix=fio.FITS('/share/des/disc2/y1/redmagicv6.4.4/y1a1_gold_1.0-pre2_run_redmapper_v6.4.4_redmagic_1.0_vlim_zmask.fit')[-1].read()['HPIX']
    #maskpix=fio.FITS('/share/des/disc2/y1/redmagicv6.4.9/y1a1_gold_1.0.2-full_redmapper_v6.4.9_redmagic_highdens_0.5_vlim_zmask.fit')[-1].read()
    #maskpix=maskpix['HPIX']((maskpix['FRACGOOD']>0.5)&(maskpix['']))
    maskpix=np.load('wfirstvipersmask.npy')
    # maskpix=fio.FITS('/share/des/disc2/y1/gold_v101/y1a1_gold_1.0.1_wide_footprint_4096.fit')[-1].read()['I']
    # maskpix=maskpix[maskpix>0].astype(int)
    maskpix=hp.ring2nest(4096,maskpix)
    catalog.CatalogMethods.create_random_cat(5000000,maskpix,label='cfhtvipers_large_')
    sys.exit()
  elif int(sys.argv[1])==7:

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    print rank,size

    rmdens=catalog.CatalogStore('y1_rm_highdens',cattype='gal',cols=['coadd','ra','dec','zp','e1','e2','lum','info','error'],catfile=config.redmagicdir+'y1a1_gold_1.0.2b-full_redmapper_v6.4.11_redmagic_highdens_0.5-10_e.fit',release='y1')
    mask=(rmdens.info==0)
    catalog.CatalogMethods.match_cat(rmdens,mask)
    ran=fio.FITS(config.redmagicdir+'y1a1_gold_1.0.2b-full_redmapper_v6.4.11_redmagic_highdens_0.5-10_randoms.fit')[-1].read()

    rmdens.ran_ra=ran['RA']
    rmdens.ran_dec=ran['DEC']
    # rmdens.ra=rmdens.ra/180.*np.pi
    # rmdens.dec=rmdens.dec/180.*np.pi
    # rmdens.ran_ra=ran['RA']/180.*np.pi
    # rmdens.ran_dec=ran['DEC']/180.*np.pi
    rmdens.ran_z=ran['Z']
    rmdens.regs=np.load(config.redmagicdir+'highdens_regs.npy')[mask]
    rmdens.ran_regs=np.load(config.redmagicdir+'highdens_ran_regs.npy')
    mask=np.random.choice(np.arange(len(rmdens.ran_ra)),2000000,replace=False)
    rmdens.ran_ra=rmdens.ran_ra[mask]
    rmdens.ran_dec=rmdens.ran_dec[mask]
    rmdens.ran_z=rmdens.ran_z[mask]
    rmdens.ran_regs=rmdens.ran_regs[mask]
    rmdens.num_regs=110
    rmdens.c1=np.zeros(len(rmdens.coadd))
    rmdens.c2=np.zeros(len(rmdens.coadd))
    rmdens.m=np.zeros(len(rmdens.coadd))
    rmdens.w=np.ones(len(rmdens.coadd))
    rmdens.bs=False
    rmdens.wt=False



    r,gp,gx,ee,xx,gperr,gxerr,eeerr,xxerr=corr.xi_2pt.ia_estimatorb(rmdens,rmdens,dlos=60.,rbins=10,rmin=.5,rmax=100.,logr=True,lum=.0,comm=comm,rank=rank,size=size,output=True,label='rm_highdens')
    if rank==0:

      print r
      print gp
      print gperr
      print gx
      print gxerr
      print ee
      print eeerr
      print xx
      print xxerr

      fig.plot_methods.plot_IA_lin((r[1:]+r[:-1])/2.,[gp,gx],[gperr,gxerr],'ge_highdens')
      fig.plot_methods.plot_IA((r[1:]+r[:-1])/2.,[ee,xx],[eeerr,xxerr],'ee_highdens')

    rmlum=catalog.CatalogStore('y1_rm_highlum',cattype='gal',cols=['coadd','ra','dec','zp','e1','e2','lum','info','error'],catfile=config.redmagicdir+'y1a1_gold_1.0.2b-full_redmapper_v6.4.11_redmagic_highlum_1.0-04_e.fit',release='y1')
    mask=(rmlum.info==0)
    catalog.CatalogMethods.match_cat(rmlum,mask)
    ran=fio.FITS(config.redmagicdir+'y1a1_gold_1.0.2b-full_redmapper_v6.4.11_redmagic_highlum_1.0-04_randoms.fit')[-1].read()

    rmlum.ra=rmlum.ra/180.*np.pi
    rmlum.dec=rmlum.dec/180.*np.pi
    rmlum.ran_ra=ran['RA']/180.*np.pi
    rmlum.ran_dec=ran['DEC']/180.*np.pi
    rmlum.ran_z=ran['Z']
    rmlum.e2=-rmlum.e2
    rmlum.regs=np.load(config.redmagicdir+'highlum_regs.npy')[mask]
    rmlum.ran_regs=np.load(config.redmagicdir+'highlum_ran_regs.npy')
    mask=np.random.choice(np.arange(len(rmlum.ran_ra)),2000000,replace=False)
    rmlum.ran_ra=rmlum.ran_ra[mask]
    rmlum.ran_dec=rmlum.ran_dec[mask]
    rmlum.ran_z=rmlum.ran_z[mask]    
    rmlum.ran_regs=rmlum.ran_regs[mask]
    rmlum.num_regs=110
    rmlum.bs=False
    rmlum.wt=False

    r,gp,gx,ee,xx,gperr,gxerr,eeerr,xxerr,tmp=corr.xi_2pt.ia_estimatorb(rmdens,rmdens,dlos=60.,rbins=10,rmin=.5,rmax=100.,logr=True,lum=.0,comm=comm,rank=rank,size=size,output=True,label='rm_highlum')
    if rank==0:

      print r
      print gp
      print gperr
      print gx
      print gxerr
      print ee
      print eeerr
      print xx
      print xxerr

      fig.plot_methods.plot_IA_lin((r[1:]+r[:-1])/2.,[gp,gx],[gperr,gxerr],'ge_highlum')
      fig.plot_methods.plot_IA((r[1:]+r[:-1])/2.,[ee,xx],[eeerr,xxerr],'ee_highlum')


  elif int(sys.argv[1])==8:

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    print rank,size

    import fitsio as fio
    import numpy.random as rand
    tmp=fio.FITS('ms_megazlike_Est_Sma_est_sma_0_0.fits')[-1].read()
    ia=catalog.CatalogStore('joachimi_ia',setup=False)
    ia.e1=tmp['e_1']
    ia.e2=tmp['e_2']
    ia.zp=tmp['redshift']
    ia.dec=tmp['y_pos[rad]']/np.pi*180.
    ia.ra=tmp['x_pos[rad]']/np.pi*180.
    ia.coadd=np.arange(len(ia.ra))
    ia.ran_ra=rand.choice(1000000,200000)*2./1000000.
    ia.ran_dec=rand.choice(1000000,200000)*2./1000000.
    ia.num_regs=1
    ia.regs=np.zeros(len(ia.ra))
    ia.ran_regs=np.zeros(len(ia.ra))

    r,gp,gx,ee,xx,gperr,gxerr,eeerr,xxerr,tmp=corr.xi_2pt.ia_estimatorb(ia,ia,dlos=100.,rbins=10,rmin=.1,rmax=100.,logr=True,comm=comm,rank=rank,size=size,output=True)
    if rank==0:

      print r
      print gp
      print gperr
      print gx
      print gxerr
      print ee
      print eeerr
      print xx
      print xxerr

      fig.plot_methods.plot_IA((r[1:]+r[:-1])/2.,[gp,gx],[gperr,gxerr],'ge')
      fig.plot_methods.plot_IA((r[1:]+r[:-1])/2.,[ee,xx],[eeerr,xxerr],'ee')

      # print tmp

  elif int(sys.argv[1])==9:

    ada=catalog.PZStore('ada',setup=True,pztype='ada',filetype='h5',file=config.pzdir+'Y1A1_GOLD101_Y1A1trainValid_14.12.2015.validsY1A1.25215.out.DES.pdf.hdf5')
    hwe=catalog.PZStore('hwe',setup=True,pztype='hwe',filetype='h5',file=config.pzdir+'PDF_MMRAU_standard_output.hdf5')
    bpz=catalog.PZStore('bpz',setup=True,pztype='bpz',filetype='h5',file=config.pzdir+'BPZ_v1_probs_DEC15_trainvalid.hdf5')
    dnf=catalog.PZStore('dnf',setup=True,pztype='dnf',filetype='h5',file=config.pzdir+'dnf_validation.hdf5')

    cnt=0
    for i,pz0 in enumerate([bpz,dnf]):
      for binning in [3,6]:
        for wt in [True,False]:
          for ztype in ['mean','peak']:
            for stack in ['pdf','mc']:
              if cnt==int(sys.argv[2]):
                pz0.wt=wt
                pz.pz_methods.build_nofz_bins(pz0,pzlow=0.2,pzhigh=1.3,cat=None,bins=binning,split=ztype,nztype=stack,pzmask=None,catmask=None,spec=True,boot=200)
                cosmosis.make.nofz(pz0,'y1_v1_spec_validation_'+ztype+'_'+stack)          
              cnt+=1
  
  elif int(sys.argv[1])==10:

    import treecorr as t

    w1=np.genfromtxt('match_W1.txt',names=True)
    w4=np.genfromtxt('match_W4.txt',names=True)
    mask1=(w1['fitclass']==0)&(w1['star_flag']==0)&(w1['MASK']<=1)&(w1['weight']>0)
    mask4=(w4['fitclass']==0)&(w4['star_flag']==0)&(w4['MASK']<=1)&(w4['weight']>0)
    mask=np.hstack((mask1,mask4))
    cfht=catalog.CatalogStore('cfhtvipers',cattype='i3',release='y1',setup=False)
    cfht.ra=np.hstack((w1['ALPHA_J2000'],w4['ALPHA_J2000']))
    cfht.dec=np.hstack((w1['DELTA_J2000'],w4['DELTA_J2000']))
    cfht.coadd=np.arange(len(cfht.ra))
    cfht.e1=np.hstack((w1['e1'],w4['e1']))
    cfht.e2=np.hstack((w1['e2'],w4['e2']))
    cfht.c1=np.zeros(len(cfht.e1))
    cfht.c2=np.hstack((w1['c2'],w4['c2']))
    cfht.m=np.hstack((w1['m'],w4['m']))
    cfht.w=np.hstack((w1['weight'],w4['weight']))
    cfht.zp=np.hstack((w1['W1_SPECTRO_PDR1zspec'],w4['W4_SPECTRO_PDR1zspec']))
    cfht.bs=True
    cfht.wt=True
    import fitsio as fio
    tmp=fio.FITS('cfhtvipers_random.fits.gz')[-1].read()
    cfht.ran_ra=tmp['ra']
    cfht.ran_dec=tmp['dec']
    cfht.ran_regs=np.load('cfht_rm_ran_regs.npy')[:,1]
    cfht.regs=np.load('cfht_rm_regs.npy')[:,1]
    cfht.regs=cfht.regs[np.in1d(np.load('cfht_rm_regs.npy')[:,0],cfht.coadd,assume_unique=True)]
    cfht.e1[~mask]=0.
    cfht.e2[~mask]=0.
    cfht.w[~mask]=0.
    cfht.c2[~mask]=0.
    cfht.c1[~mask]=0.
    cfht.m[~mask]=0.

    # cfht.ra=cfht.ra/180.*np.pi
    # cfht.dec=cfht.dec/180.*np.pi
    # cfht.ran_ra=cfht.ran_ra/180.*np.pi
    # cfht.ran_dec=cfht.ran_dec/180.*np.pi
    cfht.num_regs=50
    cfht.lum=10.*np.ones(len(cfht.coadd))

    # cfht.ran_ra=cfht.ran_ra[:200000]
    # cfht.ran_dec=cfht.ran_dec[:200000]
    # cfht.ran_regs=cfht.ran_regs[:200000]

    w1=np.genfromtxt('L6.dat')
    w4=np.genfromtxt('RL6.dat')
    cfht=catalog.CatalogStore('sdss',cattype='i3',release='y1',setup=False)
    cfht.ra=w1[:,0]
    cfht.dec=w1[:,1]
    cfht.pix=hp.ang2pix(524288, np.pi/2.-np.radians(cfht.dec),np.radians(cfht.ra), nest=True)
    cfht.coadd=np.arange(len(cfht.ra))
    cfht.e1=w1[:,10]
    cfht.e2=-w1[:,11]
    cfht.m=np.zeros(len(cfht.ra))
    cfht.c1=np.zeros(len(cfht.ra))
    cfht.c2=np.zeros(len(cfht.ra))
    cfht.zp=w1[:,2]
    cfht.w=np.ones(len(cfht.ra))

    cfht.bs=True
    cfht.wt=False
    cfht.ran_ra=w4[:,0]
    cfht.ran_dec=w4[:,1]
    cfht.ran_pix=hp.ang2pix(524288, np.pi/2.-np.radians(cfht.ran_dec),np.radians(cfht.ran_ra), nest=True)
    cfht.ran_zp=w4[:,2]
    cfht.ran_regs=w4[:,16]
    cfht.regs=w1[:,16]

    # cfht.ra=cfht.ra/180.*np.pi
    # cfht.dec=cfht.dec/180.*np.pi
    # cfht.ran_ra=cfht.ran_ra/180.*np.pi
    # cfht.ran_dec=cfht.ran_dec/180.*np.pi
    cfht.num_regs=50
    cfht.lum=10.*np.ones(len(cfht.coadd))

    

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    r,gp,gx,ee,xx,gperr,gxerr,eeerr,xxerr=corr.xi_2pt.ia_estimatorb(cfht,cfht,dlos=60.,rbins=10,rmin=.3,rmax=60.,logr=True,lum=.0,comm=comm,rank=rank,size=size,output=True)
    if rank==0:

      print r
      print gp
      print gperr
      print gx
      print gxerr
      print ee
      print eeerr
      print xx
      print xxerr

      fig.plot_methods.plot_IA_lin(r,[gp*120,gx*120],[gperr*120,gxerr*120],'ge')
      fig.plot_methods.plot_IA(r,[ee,xx],[eeerr,xxerr],'ee')

  elif int(sys.argv[1])==11:

    import matplotlib
    matplotlib.use ('agg')
    import matplotlib.pyplot as plt
    plt.style.use('/home/troxel/SVA1/SVA1StyleSheet.mplstyle')
    import matplotlib.cm as cm
    from matplotlib.colors import LogNorm
    import matplotlib.gridspec as gridspec
    from matplotlib.ticker import MultipleLocator, FormatStrFormatter
    import pylab
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.grid_search import RandomizedSearchCV
    from time import time
    from operator import itemgetter
    from scipy.stats import randint as sp_randint

    from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
    from sklearn.datasets import load_digits
    from sklearn.ensemble import RandomForestClassifier
    n_iter_search = 20
    parameters = {'max_depth':[None],
                  "max_features": [0.5, 0.7, 0.9, 1.0],
                  "min_samples_split": [2, 5, 10, 20],
                  "min_samples_leaf": [2, 5, 10, 20]}

    def report(grid_scores, n_top=3):
      top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
      for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
          score.mean_validation_score,
          np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")

    cols=['coadd','info','rgp','psf1','psf2','tile','snr','e1','e2']
    sim=catalog.CatalogStore('y1_sim_v1',cattype='i3',cols=cols,catdir='/share/des/disc2/y1/nbc/v1/results/disc/main/',release='y1')
    truth=catalog.CatalogStore('y1_truth_v1',cattype='truth',catdir='/share/des/disc2/y1/nbc/v1/truth/',release='y1')

    catalog.CatalogMethods.remove_duplicates(sim)
    catalog.CatalogMethods.remove_duplicates(truth)

    x,y=catalog.CatalogMethods.sort2(sim.coadd,truth.coadd)
    catalog.CatalogMethods.match_cat(sim,x)
    catalog.CatalogMethods.match_cat(truth,y)
    sim.zp=truth.zp

    mask=(sim.info==0)
    catalog.CatalogMethods.match_cat(sim,mask)
    catalog.CatalogMethods.match_cat(truth,mask)

    mask=~(np.isnan(sim.psf1)|np.isnan(sim.psf2)|np.isnan(sim.snr))
    catalog.CatalogMethods.match_cat(sim,mask)
    catalog.CatalogMethods.match_cat(truth,mask)

    tiles=np.genfromtxt('/share/des/disc2/y1/nbc/v1/tiles.txt',dtype=None)[np.load('/share/des/disc2/y1/nbc/v1/halftiles.npy')]
    mask=np.in1d(sim.tile,tiles,assume_unique=False)

    # RFR = RandomForestRegressor(n_estimators=2000, n_jobs=-1)

    # rf = RandomizedSearchCV(RFR, parameters, n_jobs=1, verbose=1, n_iter=n_iter_search, cv=5)

    # rf.fit(np.vstack((sim.e1[mask],sim.psf1[mask],sim.e2[mask],sim.psf2[mask],sim.snr[mask],sim.rgp[mask],sim.psffwhm[mask])).T,np.vstack((truth.e1[mask]+truth.ee1[mask],truth.e2[mask]+truth.ee2[mask])).T)
    # report(rf.grid_scores_)

    if int(sys.argv[2])==0:

      rf = RandomForestRegressor(n_estimators=1000, n_jobs=-1,min_samples_split=100)
      rf.fit(np.vstack((sim.e1[mask],sim.e2[mask],sim.snr[mask],sim.rgp[mask],sim.psf1[mask],sim.psf2[mask])).T,np.vstack((sim.e1[mask]-truth.e1[mask],sim.e2[mask]-truth.e2[mask])).T)

      a=rf.predict(np.vstack((sim.e1[~mask],sim.e2[~mask],sim.snr[~mask],sim.rgp[~mask],sim.psf1[~mask],sim.psf2[~mask])).T)
      np.save('rf_out1a.npy',np.vstack((sim.coadd[~mask],a[:,0],a[:,1])).T)

    elif int(sys.argv[2])==3:

      rf = RandomForestRegressor(n_estimators=2000, n_jobs=-1,max_depth=10)
      rf.fit(np.vstack((sim.e1[mask],sim.e2[mask],sim.snr[mask],sim.rgp[mask],sim.psf1[mask],sim.psf2[mask])).T,np.vstack((sim.e1[mask]-truth.e1[mask],sim.e2[mask]-truth.e2[mask])).T)

      a=rf.predict(np.vstack((sim.e1[~mask],sim.e2[~mask],sim.snr[~mask],sim.rgp[~mask],sim.psf1[~mask],sim.psf2[~mask])).T)
      np.save('rf_out4a.npy',np.vstack((sim.coadd[~mask],a[:,0],a[:,1])).T)

    # plt.hist2d(truth.e1[~mask]+truth.ee1[~mask],a[:,0],bins=100)
    # plt.plot([-1,1],[-1,1])
    # plt.axis('equal')
    # plt.gca().set_aspect('equal', 'box')
    # plt.savefig('tmp1.png',bbox_inches='tight')
    # plt.close()
    # plt.hist2d(truth.e2[~mask]+truth.ee2[~mask],a[:,1],bins=100)
    # plt.plot([-1,1],[-1,1])
    # plt.gca().set_aspect('equal', 'box')
    # plt.savefig('tmp2.png',bbox_inches='tight')
    # plt.close()

  elif int(sys.argv[1])==12:

    #bp=corr.bandpowers(load=True)
    i3=catalog.CatalogStore('buzzard',cattype='buzzard',cols=None,catdir='/share/des/disc2/buzzard_v1.1/y1/observed_cat/',release='y1',maxrows=300000000)
    mask=np.random.choice(np.arange(len(i3.coadd)),100000000,replace=False)
    catalog.CatalogMethods.match_cat(i3,mask)
    i3.e2=-i3.e2
    i3.tbins=500
    i3.slop=1.
    theta,out,err,chi2=corr.xi_2pt.xi_2pt(i3,corr='GG',maska=None,plot=True)
    np.save('theta2.npy',theta)
    np.save('outxip2.npy',out[0])
    np.save('outxim2.npy',out[1])

  elif int(sys.argv[1])==13:

    #bp=corr.bandpowers(load=True)
    cols=['coadd','info','dec','ra','e1','e2','zp','rgp','snr','psf1','psf2','psffwhm']
    i3=catalog.CatalogStore('y1_i3_sv_v1',cattype='i3',cols=cols,catdir='/share/des/disc2/y1/im3shape/single_band/r/y1v1/complete/main/',release='y1')
    catalog.CatalogMethods.match_cat(i3,(i3.info==0)&(i3.rgp>1.13)&(i3.snr>12)&(i3.snr<200)&(i3.rgp<3)&(~(np.isnan(i3.psf1)|np.isnan(i3.psf2)|np.isnan(i3.snr)|np.isnan(i3.psffwhm))))
    catalog.CatalogMethods.remove_duplicates(i3)
    nbc=np.load('/home/troxel/destest/i3nbcv1.npy')
    a=np.argsort(nbc[:,0])
    mask=np.diff(nbc[a,0])
    mask=mask==0
    mask=~mask
    mask=a[mask]
    nbc=nbc[mask]
    x,y=catalog.CatalogMethods.sort2(nbc[:,0],i3.coadd)
    catalog.CatalogMethods.match_cat(i3,y)
    i3.m=nbc[x,1]
    i3.c1=nbc[x,2]
    i3.c2=nbc[x,3]
    i3.w=nbc[x,4]

    i3.tbins=500
    i3.slop=1.

    theta,out,err,chi2=corr.xi_2pt.xi_2pt(i3)
    np.save('bp_theta_i3_notomo.npy',theta)
    np.save('bp_xip_i3_notomo.npy',out[0])
    np.save('bp_xim_i3_notomo.npy',out[1])
    np.save('bp_xiperr_i3_notomo.npy',err[0])
    np.save('bp_ximerr_i3_notomo.npy',err[1])

    theta,out,err,chi2=corr.xi_2pt.xi_2pt_tomo(i3,[.3,.6,.9,1.2],catb=i3)
    np.save('bp_theta_i3_tomo.npy',theta)
    np.save('bp_xip_i3_tomo.npy',out[0])
    np.save('bp_xim_i3_tomo.npy',out[1])
    np.save('bp_xiperr_i3_tomo.npy',err[0])
    np.save('bp_ximerr_i3_tomo.npy',err[1])

  elif int(sys.argv[1])==14:

    cols=['coadd','info','dec','ra','e1','e2','zp','rgp','snr','psf1','psf2','psffwhm']
    i3=catalog.CatalogStore('y1_i3_sv_v1',cattype='i3',cols=cols,catdir='/share/des/disc2/y1/im3shape/single_band/r/y1v1/complete/main/',release='y1')
    catalog.CatalogMethods.match_cat(i3,(i3.info==0)&(i3.rgp>1.13)&(i3.snr>12)&(i3.snr<200)&(i3.rgp<3)&(~(np.isnan(i3.psf1)|np.isnan(i3.psf2)|np.isnan(i3.snr)|np.isnan(i3.psffwhm))))
    catalog.CatalogMethods.remove_duplicates(i3)
    nbc=np.load('/home/troxel/destest/i3nbcv1.npy')
    a=np.argsort(nbc[:,0])
    mask=np.diff(nbc[a,0])
    mask=mask==0
    mask=~mask
    mask=a[mask]
    nbc=nbc[mask]
    x,y=catalog.CatalogMethods.sort2(nbc[:,0],i3.coadd)
    catalog.CatalogMethods.match_cat(i3,y)
    i3.m=nbc[x,1]
    i3.c1=nbc[x,2]
    i3.c2=nbc[x,3]
    i3.w=nbc[x,4]

    i3.tbins=500
    i3.slop=1.
    i3.bs=True
    i3.wt=True

    theta,out,err,chi2=corr.xi_2pt.xi_2pt(i3)
    np.save('bp_theta_i3_notomo_bs.npy',theta)
    np.save('bp_xip_i3_notomo_bs.npy',out[0])
    np.save('bp_xim_i3_notomo_bs.npy',out[1])
    np.save('bp_xiperr_i3_notomo_bs.npy',err[0])
    np.save('bp_ximerr_i3_notomo_bs.npy',err[1])

    theta,out,err,chi2=corr.xi_2pt.xi_2pt_tomo(i3,[.3,.6,.9,1.2],catb=i3)
    np.save('bp_theta_i3_tomo_bs.npy',theta)
    np.save('bp_xip_i3_tomo_bs.npy',out[0])
    np.save('bp_xim_i3_tomo_bs.npy',out[1])
    np.save('bp_xiperr_i3_tomo_bs.npy',err[0])
    np.save('bp_ximerr_i3_tomo_bs.npy',err[1])

  elif int(sys.argv[1])==15:

    #bp=corr.bandpowers(load=True)
    cols=['coadd','info','dec','ra','e1','e2','tile','zp','rgp','snr','psf1','psf2','psffwhm']
    i3i=catalog.CatalogStore('y1_i3_sv_v1i',cattype='i3',cols=cols,catfile='/share/des/disc2/y1/im3shape/single_band/r/y1v1/combined_i.fits',release='y1')
    i3r=catalog.CatalogStore('y1_i3_sv_v1r',cattype='i3',cols=cols,catfile='/share/des/disc2/y1/im3shape/single_band/r/y1v1/combined_r.fits',release='y1')
    a=np.argsort(i3i.coadd)
    mask=np.diff(i3i.coadd[a])
    mask=mask==0
    mask=~mask
    mask=a[mask]
    catalog.CatalogMethods.match_cat(i3i,mask)
    a=np.argsort(i3r.coadd)
    mask=np.diff(i3r.coadd[a])
    mask=mask==0
    mask=~mask
    mask=a[mask]
    catalog.CatalogMethods.match_cat(i3r,mask)

    catalog.CatalogMethods.match_cat(i3r,i3r.info==0)
    catalog.CatalogMethods.match_cat(i3i,i3i.info==0)

    x,y=catalog.CatalogMethods.sort2(i3i.coadd,i3r.coadd)
    catalog.CatalogMethods.match_cat(i3i,x)
    catalog.CatalogMethods.match_cat(i3r,y)

    i3r.tbins=500
    i3r.slop=1.
    i3i.tbins=500
    i3i.slop=1.

    theta,out,err,chi2=corr.xi_2pt.xi_2pt_tomo(i3r,[.3,.6,.9,1.2],catb=i3i)
    np.save('thetadataritomo.npy',theta)
    np.save('outxipdataritomo.npy',out[0])
    np.save('outximdataritomo.npy',out[1])
    np.save('outxiperrdataritomo.npy',err[0])
    np.save('outximerrdataritomo.npy',err[1])

  elif int(sys.argv[1])==16:

    #bp=corr.bandpowers(load=True)
    cols=['coadd','info','dec','ra','e1','e2','tile','zp']
    i3r=catalog.CatalogStore('y1_i3_sv_v1r',cattype='i3',cols=cols,catfile='/share/des/disc2/y1/im3shape/single_band/r/y1v1/combined_r.fits',release='y1')
    a=np.argsort(i3r.coadd)
    mask=np.diff(i3r.coadd[a])
    mask=mask==0
    mask=~mask
    mask=a[mask]

    catalog.CatalogMethods.match_cat(i3r,mask)
    catalog.CatalogMethods.match_cat(i3r,i3r.info==0)

    i3r.tbins=1000
    i3r.slop=1.

    theta,out,err,chi2=corr.xi_2pt.xi_2pt(i3r)
    np.save('thetadatar1000.npy',theta)
    np.save('outxipdatar1000.npy',out[0])
    np.save('outximdatar1000.npy',out[1])
    np.save('outxiperrdatar1000.npy',err[0])
    np.save('outximerrdatar1000.npy',err[1])

  elif int(sys.argv[1])==17:

    cols=['stamp','nexp','chi2pix','psffwhm','coadd','info','error','like','flux','rgp','dec','evals','rad','dec_off','ra_off','fluxfrac','psf1','psf2','hsmpsf1','hsmpsf2','modmax','modmin','ra','resmax','tile','maskfrac','snr','resmin','e1','e2','iter','bflux','dflux','flux','cov11','cov22','zp','g','r','i','z','bfrac']
    i3=catalog.CatalogStore('y1_i3_sv_v1',cattype='i3',cols=cols,catdir='/share/des/disc2/y1/im3shape/single_band/r/y1v1/complete/main/',release='y1')
    catalog.CatalogMethods.remove_duplicates(i3)

    rm=catalog.CatalogStore('y1_rm_highdens',cattype='gal',cols=['coadd','ra','dec','zp'],catfile=config.redmagicdir+'y1a1_gold_1.0.2b-full_redmapper_v6.4.11_redmagic_highdens_0.5-10.fit',release='y1',ranfile=config.redmagicdir+'y1a1_gold_1.0.2b-full_redmapper_v6.4.11_redmagic_highdens_0.5-10_randoms.fit')

    nbc=np.load('/home/troxel/destest/i3nbcv1.npy')
    a=np.argsort(nbc[:,0])
    mask=np.diff(nbc[a,0])
    mask=mask==0
    mask=~mask
    mask=a[mask]
    nbc=nbc[mask]

    x,y=catalog.CatalogMethods.sort2(nbc[:,0],i3.coadd)
    catalog.CatalogMethods.match_cat(i3,y)
    i3.m=nbc[x,1]
    i3.c1=nbc[x,2]
    i3.c2=nbc[x,3]
    i3.w=nbc[x,4]
    i3.w/=np.sum(i3.w)

    lin.summary_stats.i3_flags_vals_check(i3)
    lin.footprint.footprint_tests(i3,['info','error'])

    mask=(i3.info==0)&(i3.rgp>1.13)&(i3.snr>12)&(i3.snr<200)&(i3.rgp<3)&(~(np.isnan(i3.psf1)|np.isnan(i3.psf2)|np.isnan(i3.snr)|np.isnan(i3.psffwhm)))&(i3.g<99)&(i3.r<99)&(i3.i<99)&(i3.z<99)
    catalog.CatalogMethods.match_cat(i3,mask)

    lin.summary_stats.i3_flags_vals_check(i3)

    cols=['chi2pix','psffwhm','nlike','flux','rgp','dec','evals','rad','dec_off','ra_off','invfluxfrac','psf1','psf2','hsmpsf1','hsmpsf2','modmax','modmin','ra','resmax','maskfrac','snr','resmin','e1','e2','flux','zp','g','r','i','z','pos','e','psfe','dpsf','psfpos','hsmpsfe','hsmdpsf','hsmpsfpos']

    i3.wt=False
    i3.bs=False
    lin.summary_stats.val_stats(i3,cols=cols)

    lin.hist.hist_tests(i3,cols=cols)
    lin.hist.hist_2D_tests(i3,colsx=cols,colsy=cols)
    lin.footprint.hexbin_tests(i3,cols=cols)
    lin.footprint.footprint_tests(cat,[],label='All')

  elif int(sys.argv[1])==18:

    cols=['stamp','nexp','chi2pix','psffwhm','coadd','info','error','like','flux','rgp','dec','evals','rad','dec_off','ra_off','fluxfrac','psf1','psf2','hsmpsf1','hsmpsf2','modmax','modmin','ra','resmax','tile','maskfrac','snr','resmin','e1','e2','iter','bflux','dflux','flux','cov11','cov22','zp','g','r','i','z','bfrac']
    i3=catalog.CatalogStore('y1_i3_sv_v1',cattype='i3',cols=cols,catdir='/share/des/disc2/y1/im3shape/single_band/r/y1v1/complete/main/',release='y1')
    catalog.CatalogMethods.remove_duplicates(i3)

    rm=catalog.CatalogStore('y1_rm_highdens',cattype='gal',cols=['coadd','ra','dec','zp'],catfile=config.redmagicdir+'y1a1_gold_1.0.2b-full_redmapper_v6.4.11_redmagic_highdens_0.5-10.fit',release='y1',ranfile=config.redmagicdir+'y1a1_gold_1.0.2b-full_redmapper_v6.4.11_redmagic_highdens_0.5-10_randoms.fit')

    nbc=np.load('/home/troxel/destest/i3nbcv1.npy')
    a=np.argsort(nbc[:,0])
    mask=np.diff(nbc[a,0])
    mask=mask==0
    mask=~mask
    mask=a[mask]
    nbc=nbc[mask]

    x,y=catalog.CatalogMethods.sort2(nbc[:,0],i3.coadd)
    catalog.CatalogMethods.match_cat(i3,y)
    i3.m=nbc[x,1]
    i3.c1=nbc[x,2]
    i3.c2=nbc[x,3]
    i3.w=nbc[x,4]
    i3.w/=np.sum(i3.w)

    lin.summary_stats.i3_flags_vals_check(i3)
    lin.footprint.footprint_tests(i3,['info','error'])

    mask=(i3.info==0)&(i3.rgp>1.13)&(i3.snr>12)&(i3.snr<200)&(i3.rgp<3)&(~(np.isnan(i3.psf1)|np.isnan(i3.psf2)|np.isnan(i3.snr)|np.isnan(i3.psffwhm)))&(i3.g<99)&(i3.r<99)&(i3.i<99)&(i3.z<99)
    catalog.CatalogMethods.match_cat(i3,mask)

    lin.summary_stats.i3_flags_vals_check(i3)

    cols=['chi2pix','psffwhm','nlike','flux','rgp','dec','evals','rad','dec_off','ra_off','invfluxfrac','psf1','psf2','hsmpsf1','hsmpsf2','modmax','modmin','ra','resmax','maskfrac','snr','resmin','e1','e2','flux','zp','g','r','i','z','pos','e','psfe','dpsf','psfpos','hsmpsfe','hsmdpsf','hsmpsfpos']

    i3.wt=True
    i3.bs=True
    lin.summary_stats.val_stats(i3)
    sys_split.split.cat_splits_lin_e(i3,cols=cols)
    sys_split.split.cat_splits_lin_full(i3,cols=cols)
    sys_split.split.cat_splits_2pt(i3,rm,cols=cols)

  elif int(sys.argv[1])==19:

    cols=['stamp','nexp','chi2pix','psffwhm','coadd','info','error','like','flux','rgp','dec','evals','rad','dec_off','ra_off','fluxfrac','psf1','psf2','hsmpsf1','hsmpsf2','modmax','modmin','ra','resmax','tile','maskfrac','snr','resmin','e1','e2','iter','bflux','dflux','flux','cov11','cov22','zp','g','r','i','z','bfrac']
    i3=catalog.CatalogStore('y1_i3_sv_v1',cattype='i3',cols=cols,catdir='/share/des/disc2/y1/im3shape/single_band/r/y1v1/complete/main/',release='y1')
    catalog.CatalogMethods.remove_duplicates(i3)

    rm=catalog.CatalogStore('y1_rm_highdens',cattype='gal',cols=['coadd','ra','dec','zp'],catfile=config.redmagicdir+'y1a1_gold_1.0.2b-full_redmapper_v6.4.11_redmagic_highdens_0.5-10.fit',release='y1',ranfile=config.redmagicdir+'y1a1_gold_1.0.2b-full_redmapper_v6.4.11_redmagic_highdens_0.5-10_randoms.fit')

    nbc=np.load('/home/troxel/destest/i3nbcv1.npy')
    a=np.argsort(nbc[:,0])
    mask=np.diff(nbc[a,0])
    mask=mask==0
    mask=~mask
    mask=a[mask]
    nbc=nbc[mask]

    x,y=catalog.CatalogMethods.sort2(nbc[:,0],i3.coadd)
    catalog.CatalogMethods.match_cat(i3,y)
    i3.m=nbc[x,1]
    i3.c1=nbc[x,2]
    i3.c2=nbc[x,3]
    i3.w=nbc[x,4]
    i3.w/=np.sum(i3.w)

    lin.summary_stats.i3_flags_vals_check(i3)
    lin.footprint.footprint_tests(i3,['info','error'])

    mask=(i3.info==0)&(i3.rgp>1.13)&(i3.snr>12)&(i3.snr<200)&(i3.rgp<3)&(~(np.isnan(i3.psf1)|np.isnan(i3.psf2)|np.isnan(i3.snr)|np.isnan(i3.psffwhm)))&(i3.g<99)&(i3.r<99)&(i3.i<99)&(i3.z<99)
    catalog.CatalogMethods.match_cat(i3,mask)

    lin.summary_stats.i3_flags_vals_check(i3)

    cols=['chi2pix','psffwhm','nlike','flux','rgp','dec','evals','rad','dec_off','ra_off','invfluxfrac','psf1','psf2','hsmpsf1','hsmpsf2','modmax','modmin','ra','resmax','maskfrac','snr','resmin','e1','e2','flux','zp','g','r','i','z','pos','e','psfe','dpsf','psfpos','hsmpsfe','hsmdpsf','hsmpsfpos']

    i3.wt=False
    i3.bs=False
    sys_split.split.cat_splits_lin_e(i3,cols=cols)
    sys_split.split.cat_splits_lin_full(i3,cols=cols)
    sys_split.split.cat_splits_2pt(i3,rm,cols=cols)

  elif int(sys.argv[1])==20:

    cols=['stamp','nexp','chi2pix','psffwhm','coadd','info','error','like','flux','rgp','dec','evals','rad','dec_off','ra_off','fluxfrac','psf1','psf2','hsmpsf1','hsmpsf2','modmax','modmin','ra','resmax','tile','maskfrac','snr','resmin','e1','e2','iter','bflux','dflux','flux','cov11','cov22','zp','g','r','i','z','bfrac']
    i3=catalog.CatalogStore('y1_i3_sv_v1',cattype='i3',cols=cols,catdir='/share/des/disc2/y1/im3shape/single_band/r/y1v1/complete/main/',release='y1')
    catalog.CatalogMethods.remove_duplicates(i3)

    rm=catalog.CatalogStore('y1_rm_highdens',cattype='gal',cols=['coadd','ra','dec','zp'],catfile=config.redmagicdir+'y1a1_gold_1.0.2b-full_redmapper_v6.4.11_redmagic_highdens_0.5-10.fit',release='y1',ranfile=config.redmagicdir+'y1a1_gold_1.0.2b-full_redmapper_v6.4.11_redmagic_highdens_0.5-10_randoms.fit')

    nbc=np.load('/home/troxel/destest/i3nbcv1.npy')
    a=np.argsort(nbc[:,0])
    mask=np.diff(nbc[a,0])
    mask=mask==0
    mask=~mask
    mask=a[mask]
    nbc=nbc[mask]

    x,y=catalog.CatalogMethods.sort2(nbc[:,0],i3.coadd)
    catalog.CatalogMethods.match_cat(i3,y)
    i3.m=nbc[x,1]
    i3.c1=nbc[x,2]
    i3.c2=nbc[x,3]
    i3.w=nbc[x,4]
    i3.w/=np.sum(i3.w)

    lin.summary_stats.i3_flags_vals_check(i3)
    lin.footprint.footprint_tests(i3,['info','error'])

    mask=(i3.info==0)&(i3.rgp>1.13)&(i3.snr>12)&(i3.snr<200)&(i3.rgp<3)&(~(np.isnan(i3.psf1)|np.isnan(i3.psf2)|np.isnan(i3.snr)|np.isnan(i3.psffwhm)))&(i3.g<99)&(i3.r<99)&(i3.i<99)&(i3.z<99)
    catalog.CatalogMethods.match_cat(i3,mask)

    lin.summary_stats.i3_flags_vals_check(i3)

    cols=['chi2pix','psffwhm','nlike','flux','rgp','dec','evals','rad','dec_off','ra_off','invfluxfrac','psf1','psf2','hsmpsf1','hsmpsf2','modmax','modmin','ra','resmax','maskfrac','snr','resmin','e1','e2','flux','zp','g','r','i','z','pos','e','psfe','dpsf','psfpos','hsmpsfe','hsmdpsf','hsmpsfpos']

    lin.summary_stats.tile_stats(i3,cols=cols)
    lin.hist.tile_tests(i3)
    lin.hist.tile_tests_2D(i3)
    lin.footprint.tile_tests(i3)

  elif int(sys.argv[1])==21:

    p=None

    # Select columns to read from catalog (no cols option indicates to read all columns from appropriate dict in config.py)
    cols=['stamp','nexp','chi2pix','psffwhm','coadd','info','error','like','rgp','dec','evals','rad','dec_off','ra_off','fluxfrac','psf1','psf2','hsmpsf1','hsmpsf2','modmax','modmin','ra','resmax','tile','maskfrac','snr','resmin','e1','e2','iter','bflux','dflux','flux','cov11','cov22','zp','g','r','i','z','bfrac']
    # Build CatalogStore object for shape catalog - could include cutfunc option that points to a function to, for example, not read objects with error flags set. We want to look at the error distribution, so we don't do this here. p!=None will set up the arrays in shared memory for multiprocessing.Pool() use in other functions.
    i3=catalog.CatalogStore('y1_i3_sv_v1',cattype='i3',cols=cols,catdir='/share/des/disc2/y1/im3shape/single_band/r/y1v1/complete/main/',release='y1',p=p)
    # Remove duplicate unique ids (if necessary)
    catalog.CatalogMethods.remove_duplicates(i3)

    # # Read in galaxy catalog
    # rm=catalog.CatalogStore('y1_rm_highdens',cattype='gal',cols=['coadd','ra','dec','zp'],catfile=config.redmagicdir+'y1a1_gold_1.0.2b-full_redmapper_v6.4.11_redmagic_highdens_0.5-10.fit',release='y1',ranfile=config.redmagicdir+'y1a1_gold_1.0.2b-full_redmapper_v6.4.11_redmagic_highdens_0.5-10_randoms.fit')

    print 'done with catalog stuff'

    # Pre-cut summary of error properties
    lin.summary_stats.i3_flags_vals_check(i3)
    # Plots location of flagged failures
    #lin.footprint.footprint_tests(i3,['info','error'],p=True)

    print 'done with error stuff'

    # Cut bad objects from catalog
    mask=(i3.info==0)&(i3.rgp>1.13)&(i3.snr>12)&(i3.snr<200)&(i3.rgp<3)&(~(np.isnan(i3.psf1)|np.isnan(i3.psf2)|np.isnan(i3.snr)|np.isnan(i3.psffwhm)))&(i3.g<99)&(i3.r<99)&(i3.i<99)&(i3.z<99)
    catalog.CatalogMethods.match_cat(i3,mask)

    nbc=np.load('/home/troxel/destest/i3nbcv1.npy')
    a=np.argsort(nbc[:,0])
    mask=np.diff(nbc[a,0])
    mask=mask==0
    mask=~mask
    mask=a[mask]
    nbc=nbc[mask]

    x,y=catalog.CatalogMethods.sort2(nbc[:,0],i3.coadd)
    catalog.CatalogMethods.match_cat(i3,y)
    i3.m1=nbc[x,1]
    i3.m2=nbc[x,1]
    i3.c1=nbc[x,2]
    i3.c2=nbc[x,3]
    i3.w=nbc[x,4]
    i3.w/=np.sum(i3.w)

    # Post-cut summary of error properties
    lin.summary_stats.i3_flags_vals_check(i3,label='b')

    # Optionally specify sub-set of columns for the following functions
    cols=['chi2pix','psffwhm','nlike','rgp','dec','evals','rad','dec_off','ra_off','invfluxfrac','psf1','psf2','hsmpsf1','hsmpsf2','modmax','modmin','ra','resmax','maskfrac','snr','resmin','e1','e2','flux','cov11','cov22','zp','g','r','i','z','gr','ri','iz','pos','e','psfe','dpsf','psfpos','hsmpsfe','hsmdpsf','hsmpsfpos','m1','c1','c2','w']

    print 'done with vals check stuff'

    # Check summary statistics of catalog values (default all)
    i3.wt=True
    i3.bs=True
    lin.summary_stats.val_stats(i3)

    print 'done with vals stuff'
    # p!=None (default None) will spawn a multiprocessing Pool() that work is distributed over
    # Produce object density plot across survey footprint (default all)
    #lin.footprint.footprint_tests(i3,[],label='All')

    # Produce histograms of catalog columns (default all)
    lin.hist.hist_tests(i3,cols=cols,p=p)

    print 'done with hist stuff'
    # Produce plots of mean values of columns across survey footprint (default all)
    lin.footprint.hexbin_tests(i3,cols=cols,p=p)

    lin.hist.hist_2D_tests(i3,colsx=cols,colsy=cols,p=p)

    print 'done with footprint stuff'

  elif int(sys.argv[1])==22:


    p=None

    # Select columns to read from catalog (no cols option indicates to read all columns from appropriate dict in config.py)
    cols=['stamp','nexp','chi2pix','psffwhm','coadd','info','error','like','rgp','dec','evals','rad','dec_off','ra_off','fluxfrac','psf1','psf2','hsmpsf1','hsmpsf2','modmax','modmin','ra','resmax','tile','maskfrac','snr','resmin','e1','e2','iter','bflux','dflux','flux','cov11','cov22','zp','g','r','i','z','bfrac']
    # Build CatalogStore object for shape catalog - could include cutfunc option that points to a function to, for example, not read objects with error flags set. We want to look at the error distribution, so we don't do this here. p!=None will set up the arrays in shared memory for multiprocessing.Pool() use in other functions.
    i3=catalog.CatalogStore('y1_i3_sv_v1',cattype='i3',cols=cols,catdir='/share/des/disc2/y1/im3shape/single_band/r/y1v1/complete/main/',release='y1',p=p)
    # Remove duplicate unique ids (if necessary)
    catalog.CatalogMethods.remove_duplicates(i3)

    # # Read in galaxy catalog
    # rm=catalog.CatalogStore('y1_rm_highdens',cattype='gal',cols=['coadd','ra','dec','zp'],catfile=config.redmagicdir+'y1a1_gold_1.0.2b-full_redmapper_v6.4.11_redmagic_highdens_0.5-10.fit',release='y1',ranfile=config.redmagicdir+'y1a1_gold_1.0.2b-full_redmapper_v6.4.11_redmagic_highdens_0.5-10_randoms.fit')

    print 'done with catalog stuff'

    # Pre-cut summary of error properties
    lin.summary_stats.i3_flags_vals_check(i3)
    # Plots location of flagged failures
    #lin.footprint.footprint_tests(i3,['info','error'],p=True)

    print 'done with error stuff'

    # Cut bad objects from catalog
    mask=(i3.info==0)&(i3.rgp>1.13)&(i3.snr>12)&(i3.snr<200)&(i3.rgp<3)&(~(np.isnan(i3.psf1)|np.isnan(i3.psf2)|np.isnan(i3.snr)|np.isnan(i3.psffwhm)))&(i3.g<99)&(i3.r<99)&(i3.i<99)&(i3.z<99)
    catalog.CatalogMethods.match_cat(i3,mask)

    nbc=np.load('/home/troxel/destest/i3nbcv1.npy')
    a=np.argsort(nbc[:,0])
    mask=np.diff(nbc[a,0])
    mask=mask==0
    mask=~mask
    mask=a[mask]
    nbc=nbc[mask]

    x,y=catalog.CatalogMethods.sort2(nbc[:,0],i3.coadd)
    catalog.CatalogMethods.match_cat(i3,y)
    i3.m1=nbc[x,1]
    i3.m2=nbc[x,1]
    i3.c1=nbc[x,2]
    i3.c2=nbc[x,3]
    i3.w=nbc[x,4]
    i3.w/=np.sum(i3.w)

    # Optionally specify sub-set of columns for the following functions
    cols=['chi2pix','psffwhm','nlike','rgp','dec','evals','rad','dec_off','ra_off','invfluxfrac','psf1','psf2','hsmpsf1','hsmpsf2','modmax','modmin','ra','resmax','maskfrac','snr','resmin','e1','e2','flux','cov11','cov22','zp','g','r','i','z','gr','ri','iz','pos','e','psfe','dpsf','psfpos','hsmpsfe','hsmdpsf','hsmpsfpos','m1','c1','c2','w']

    print cols

    i3.wt=True
    i3.bs=True
    # Produce plots and statistics on linear correlation of ellipticity with catalog columns
    sys_split.split.cat_splits_lin_e(i3,cols=cols,p=p)
    # Produce plots and statistics on linear correlation of non-ellipticity catalog columns
    sys_split.split.cat_splits_lin_full(i3,cols=cols,p=p)

  elif int(sys.argv[1])==23:

    p=None

    # Select columns to read from catalog (no cols option indicates to read all columns from appropriate dict in config.py)
    cols=['stamp','nexp','chi2pix','psffwhm','coadd','info','error','like','rgp','dec','evals','rad','dec_off','ra_off','fluxfrac','psf1','psf2','hsmpsf1','hsmpsf2','modmax','modmin','ra','resmax','tile','maskfrac','snr','resmin','e1','e2','iter','bflux','dflux','flux','cov11','cov22','zp','g','r','i','z','bfrac']
    # Build CatalogStore object for shape catalog - could include cutfunc option that points to a function to, for example, not read objects with error flags set. We want to look at the error distribution, so we don't do this here. p!=None will set up the arrays in shared memory for multiprocessing.Pool() use in other functions.
    i3=catalog.CatalogStore('y1_i3_sv_v1',cattype='i3',cols=cols,catdir='/share/des/disc2/y1/im3shape/single_band/r/y1v1/complete/main/',release='y1',p=p)
    # Remove duplicate unique ids (if necessary)
    catalog.CatalogMethods.remove_duplicates(i3)

    # # Read in galaxy catalog
    # rm=catalog.CatalogStore('y1_rm_highdens',cattype='gal',cols=['coadd','ra','dec','zp'],catfile=config.redmagicdir+'y1a1_gold_1.0.2b-full_redmapper_v6.4.11_redmagic_highdens_0.5-10.fit',release='y1',ranfile=config.redmagicdir+'y1a1_gold_1.0.2b-full_redmapper_v6.4.11_redmagic_highdens_0.5-10_randoms.fit')

    print 'done with catalog stuff'

    # Pre-cut summary of error properties
    lin.summary_stats.i3_flags_vals_check(i3)
    # Plots location of flagged failures
    #lin.footprint.footprint_tests(i3,['info','error'],p=True)

    print 'done with error stuff'

    # Cut bad objects from catalog
    mask=(i3.info==0)&(i3.rgp>1.13)&(i3.snr>12)&(i3.snr<200)&(i3.rgp<3)&(~(np.isnan(i3.psf1)|np.isnan(i3.psf2)|np.isnan(i3.snr)|np.isnan(i3.psffwhm)))&(i3.g<99)&(i3.r<99)&(i3.i<99)&(i3.z<99)
    catalog.CatalogMethods.match_cat(i3,mask)

    nbc=np.load('/home/troxel/destest/i3nbcv1.npy')
    a=np.argsort(nbc[:,0])
    mask=np.diff(nbc[a,0])
    mask=mask==0
    mask=~mask
    mask=a[mask]
    nbc=nbc[mask]

    x,y=catalog.CatalogMethods.sort2(nbc[:,0],i3.coadd)
    catalog.CatalogMethods.match_cat(i3,y)
    i3.m1=nbc[x,1]
    i3.m2=nbc[x,1]
    i3.c1=nbc[x,2]
    i3.c2=nbc[x,3]
    i3.w=nbc[x,4]
    i3.w/=np.sum(i3.w)

    # Optionally specify sub-set of columns for the following functions
    cols=['chi2pix','psffwhm','nlike','rgp','dec','evals','rad','dec_off','ra_off','invfluxfrac','psf1','psf2','hsmpsf1','hsmpsf2','modmax','modmin','ra','resmax','maskfrac','snr','resmin','e1','e2','flux','cov11','cov22','zp','g','r','i','z','gr','ri','iz','pos','e','psfe','dpsf','psfpos','hsmpsfe','hsmdpsf','hsmpsfpos','m1','c1','c2','w']

    # Compile summary statistics per tile of catalog values (default all)
    lin.summary_stats.tile_stats(i3,cols=cols,p=p)
    # Produce histograms of catalog columns per tile - number of tiles small enough that this doesn't bother paralellising (default all)
    lin.hist.tile_tests(i3)
    lin.hist.tile_tests_2D(i3)
    #Produce plots of mean values of columns across survey footprint by tile(default all)
    lin.footprint.tile_hexbin_tests(i3,p=p)

    print 'done with tile stuff'

    # # Show the relative disagreement between 2pt statistics when using only galaxies split into each half of a given quantity
    # #sys_split.split.cat_splits_2pt(i3,rm,cols=cols)

  elif int(sys.argv[1])==24:

    p=None

    # Select columns to read from catalog (no cols option indicates to read all columns from appropriate dict in config.py)
    cols=['stamp','nexp','chi2pix','psffwhm','coadd','info','error','like','rgp','dec','evals','rad','dec_off','ra_off','fluxfrac','psf1','psf2','hsmpsf1','hsmpsf2','modmax','modmin','ra','resmax','tile','maskfrac','snr','resmin','e1','e2','iter','bflux','dflux','flux','cov11','cov22','zp','g','r','i','z','bfrac']
    # Build CatalogStore object for shape catalog - could include cutfunc option that points to a function to, for example, not read objects with error flags set. We want to look at the error distribution, so we don't do this here. p!=None will set up the arrays in shared memory for multiprocessing.Pool() use in other functions.
    i3=catalog.CatalogStore('y1_i3_sv_v1',cattype='i3',cols=cols,catdir='/share/des/disc2/y1/im3shape/single_band/r/y1v1/complete/main/',release='y1',p=p)
    # Remove duplicate unique ids (if necessary)
    catalog.CatalogMethods.remove_duplicates(i3)

    # # Read in galaxy catalog
    rm=catalog.CatalogStore('y1_rm_highdens',cattype='gal',cols=['coadd','ra','dec','zp'],catfile=config.redmagicdir+'y1a1_gold_1.0.2b-full_redmapper_v6.4.11_redmagic_highdens_0.5-10.fit',release='y1',ranfile=config.redmagicdir+'y1a1_gold_1.0.2b-full_redmapper_v6.4.11_redmagic_highdens_0.5-10_randoms.fit')

    print 'done with catalog stuff'

    # Pre-cut summary of error properties
    lin.summary_stats.i3_flags_vals_check(i3)
    # Plots location of flagged failures
    #lin.footprint.footprint_tests(i3,['info','error'],p=True)

    print 'done with error stuff'

    # Cut bad objects from catalog
    mask=(i3.info==0)&(i3.rgp>1.13)&(i3.snr>12)&(i3.snr<200)&(i3.rgp<3)&(~(np.isnan(i3.psf1)|np.isnan(i3.psf2)|np.isnan(i3.snr)|np.isnan(i3.psffwhm)))&(i3.g<99)&(i3.r<99)&(i3.i<99)&(i3.z<99)
    catalog.CatalogMethods.match_cat(i3,mask)

    nbc=np.load('/home/troxel/destest/i3nbcv1.npy')
    a=np.argsort(nbc[:,0])
    mask=np.diff(nbc[a,0])
    mask=mask==0
    mask=~mask
    mask=a[mask]
    nbc=nbc[mask]

    x,y=catalog.CatalogMethods.sort2(nbc[:,0],i3.coadd)
    catalog.CatalogMethods.match_cat(i3,y)
    i3.m1=nbc[x,1]
    i3.m2=nbc[x,1]
    i3.c1=nbc[x,2]
    i3.c2=nbc[x,3]
    i3.w=nbc[x,4]
    i3.w/=np.sum(i3.w)

    print 'done with vals stuff'
    # p!=None (default None) will spawn a multiprocessing Pool() that work is distributed over

    cols=['snr','rgp','ra','dec','psf1','psf2','psffwhm','hsmpsf1','hsmpsf2','psfe','dpsf','psfpos','hsmpsfe','hsmdpsf','hsmpsfpos','g','r','i','z','chi2pix','nlike','gr','ri','iz']

    # # Show the relative disagreement between 2pt statistics when using only galaxies split into each half of a given quantity
    i3.bs=False
    i3.wt=False
    i3.pzrw=True
    sys_split.split.cat_splits_2pt(i3,rm,cols=cols)

  elif int(sys.argv[1])==25:
    p=None

    # Select columns to read from catalog (no cols option indicates to read all columns from appropriate dict in config.py)
    cols=['stamp','nexp','chi2pix','psffwhm','coadd','info','error','like','rgp','dec','evals','rad','dec_off','ra_off','fluxfrac','psf1','psf2','hsmpsf1','hsmpsf2','modmax','modmin','ra','resmax','tile','maskfrac','snr','resmin','e1','e2','iter','bflux','dflux','flux','cov11','cov22','zp','g','r','i','z','bfrac']
    # Build CatalogStore object for shape catalog - could include cutfunc option that points to a function to, for example, not read objects with error flags set. We want to look at the error distribution, so we don't do this here. p!=None will set up the arrays in shared memory for multiprocessing.Pool() use in other functions.
    i3=catalog.CatalogStore('y1_i3_sv_v1',cattype='i3',cols=cols,catdir='/share/des/disc2/y1/im3shape/single_band/r/y1v1/complete/main/',release='y1',p=p)

i3old=catalog.CatalogStore('y1_i3_sv_v3',cattype='i3',cols=cols,catdir='/project/projectdirs/des/wl/desdata/wlpipe/im3shape_y1a1_v3/bord/main/',release='y1',p=p,tiles=np.genfromtxt('/project/projectdirs/des/wl/desdata/wlpipe/im3shape_y1a1_v3/both.txt',dtype=None)[:200])
i3new=catalog.CatalogStore('y1_i3_sv_v1',cattype='i3',cols=cols,catdir='/project/projectdirs/des/wl/desdata/wlpipe/im3shape_y1a1_v1/nbc/main/',release='y1',p=p,tiles=np.genfromtxt('/project/projectdirs/des/wl/desdata/wlpipe/im3shape_y1a1_v3/both.txt',dtype=None)[:200])

catalog.CatalogMethods.remove_duplicates(i3old)
catalog.CatalogMethods.remove_duplicates(i3new)

maskold=(i3old.info==0)&(i3old.rgp>1.13)&(i3old.snr>12)&(i3old.snr<200)&(i3old.rgp<3)&(~(np.isnan(i3old.psf1)|np.isnan(i3old.psf2)|np.isnan(i3old.snr)|np.isnan(i3old.psffwhm)))&(i3old.g<99)&(i3old.r<99)&(i3old.i<99)&(i3old.z<99)
masknew=(i3new.info==0)&(i3new.rgp>1.13)&(i3new.snr>12)&(i3new.snr<200)&(i3new.rgp<3)&(~(np.isnan(i3new.psf1)|np.isnan(i3new.psf2)|np.isnan(i3new.snr)|np.isnan(i3new.psffwhm)))&(i3new.g<99)&(i3new.r<99)&(i3new.i<99)&(i3new.z<99)

catalog.CatalogMethods.match_cat(i3new,masknew)
catalog.CatalogMethods.match_cat(i3old,maskold)

lin.summary_stats.val_stats(i3old)
lin.summary_stats.val_stats(i3new)

lin.hist.hist_tests(i3new,cat2=i3old)

sys_split.split.cat_splits_lin_e(i3old)
sys_split.split.cat_splits_lin_e(i3new)



i3old=catalog.CatalogStore('y1_i3_sv_v1',cattype='i3',cols=cols,catdir='/share/des/y1/meds-boxsize-test/box32/main/',release='y1',p=p)
i48=catalog.CatalogStore('y1_i3_sv_v1',cattype='i3',cols=cols,catdir='/share/des/y1/meds-boxsize-test/box48/main/',release='y1',p=p)

catalog.CatalogMethods.remove_duplicates(i32)
catalog.CatalogMethods.remove_duplicates(i48)

x,y=catalog.CatalogMethods.sort2(i32.coadd,i48.coadd)
catalog.CatalogMethods.match_cat(i32,x)
catalog.CatalogMethods.match_cat(i48,y)

de1=i32.e1-i48.e1
de2=i32.e2-i48.e2
dpsf1=i32.psf1-i48.psf1
dpsf2=i32.psf2-i48.psf2

m32=i32.stamp==32
m48=i32.stamp==48
m64=i32.stamp==64
m96=i32.stamp==96

a,b=catalog.CatalogMethods.sort2(i32.coadd[i32.stamp==48],i48.coadd[i48.stamp==48])

print np.mean(i32.e1[m32]),np.mean(i48.e1[m32])
print np.mean(i32.e1[m48]),np.mean(i48.e1[m48])
print np.mean(i32.e1[m64]),np.mean(i48.e1[m64])
print np.mean(i32.e1[m96]),np.mean(i48.e1[m96])

print np.mean(i32.e2[m32]),np.mean(i48.e2[m32])
print np.mean(i32.e2[m48]),np.mean(i48.e2[m48])
print np.mean(i32.e2[m64]),np.mean(i48.e2[m64])
print np.mean(i32.e2[m96]),np.mean(i48.e2[m96])

mask=(i32.info==0)&(i32.rgp>1.13)&(i32.snr>12)&(i32.snr<200)&(i32.rgp<3)&(~(np.isnan(i32.psf1)|np.isnan(i32.psf2)|np.isnan(i32.snr)|np.isnan(i32.psffwhm)))&(i32.g<99)&(i32.r<99)&(i32.i<99)&(i32.z<99)
mask2=(i48.info==0)&(i48.rgp>1.13)&(i48.snr>12)&(i48.snr<200)&(i48.rgp<3)&(~(np.isnan(i48.psf1)|np.isnan(i48.psf2)|np.isnan(i48.snr)|np.isnan(i48.psffwhm)))&(i48.g<99)&(i48.r<99)&(i48.i<99)&(i48.z<99)

print np.mean(i32.e1[m32&mask]),np.mean(i48.e1[m32&mask])
print np.mean(i32.e1[m48&mask]),np.mean(i48.e1[m48&mask])
print np.mean(i32.e1[m64&mask]),np.mean(i48.e1[m64&mask])
print np.mean(i32.e1[m96&mask]),np.mean(i48.e1[m96&mask])

print np.mean(i32.e2[m32&mask]),np.mean(i48.e2[m32&mask])
print np.mean(i32.e2[m48&mask]),np.mean(i48.e2[m48&mask])
print np.mean(i32.e2[m64&mask]),np.mean(i48.e2[m64&mask])
print np.mean(i32.e2[m96&mask]),np.mean(i48.e2[m96&mask])

print np.mean(i32.e1[m32&mask]-i48.e1[m32&mask])
print np.mean(i32.e1[m48&mask]-i48.e1[m48&mask])
print np.mean(i32.e1[m64&mask]-i48.e1[m64&mask])
print np.mean(i32.e1[m96&mask]-i48.e1[m96&mask])

print np.mean(i32.e2[m32&mask]-i48.e2[m32&mask])
print np.mean(i32.e2[m48&mask]-i48.e2[m48&mask])
print np.mean(i32.e2[m64&mask]-i48.e2[m64&mask])
print np.mean(i32.e2[m96&mask]-i48.e2[m96&mask])


print np.mean(i32.psf1[m32&mask]),np.mean(i48.psf1[m32&mask])
print np.mean(i32.psf1[m48&mask]),np.mean(i48.psf1[m48&mask])
print np.mean(i32.psf1[m64&mask]),np.mean(i48.psf1[m64&mask])
print np.mean(i32.psf1[m96&mask]),np.mean(i48.psf1[m96&mask])

print np.mean(i32.psf2[m32&mask]),np.mean(i48.psf2[m32&mask])
print np.mean(i32.psf2[m48&mask]),np.mean(i48.psf2[m48&mask])
print np.mean(i32.psf2[m64&mask]),np.mean(i48.psf2[m64&mask])
print np.mean(i32.psf2[m96&mask]),np.mean(i48.psf2[m96&mask])

sys_split.split.cat_splits_lin_full(i32,cols=['de1','de2','psf1','psf2','dpsf1','dpsf2'],mask=m32&mask)
sys_split.split.cat_splits_lin_full(i32,cols=['de1','de2','psf1','psf2','dpsf1','dpsf2'],mask=m64&mask)
sys_split.split.cat_splits_lin_full(i32,cols=['de1','de2','psf1','psf2','dpsf1','dpsf2'],mask=m64&mask)
sys_split.split.cat_splits_lin_full(i32,cols=['de1','de2','psf1','psf2','dpsf1','dpsf2'],mask=m96&mask)

sys_split.split.cat_splits_lin_full(i48,cols=['de1','de2','psf1','psf2','dpsf1','dpsf2'],mask=m32&mask)
sys_split.split.cat_splits_lin_full(i48,cols=['de1','de2','psf1','psf2','dpsf1','dpsf2'],mask=m64&mask)
sys_split.split.cat_splits_lin_full(i48,cols=['de1','de2','psf1','psf2','dpsf1','dpsf2'],mask=m64&mask)
sys_split.split.cat_splits_lin_full(i48,cols=['de1','de2','psf1','psf2','dpsf1','dpsf2'],mask=m96&mask)

sigma_clip(i32.e1[m32&mask]-i48.e1[m32&mask], weights=None, niter=4, nsig=4, get_err=True)
sigma_clip(i32.e1[m48&mask]-i48.e1[m48&mask], weights=None, niter=4, nsig=4, get_err=True)
sigma_clip(i32.e1[m64&mask]-i48.e1[m64&mask], weights=None, niter=4, nsig=4, get_err=True)
sigma_clip(i32.e1[m96&mask]-i48.e1[m96&mask], weights=None, niter=4, nsig=4, get_err=True)

sigma_clip(i32.e2[m32&mask]-i48.e2[m32&mask], weights=None, niter=4, nsig=4, get_err=True)
sigma_clip(i32.e2[m48&mask]-i48.e2[m48&mask], weights=None, niter=4, nsig=4, get_err=True)
sigma_clip(i32.e2[m64&mask]-i48.e2[m64&mask], weights=None, niter=4, nsig=4, get_err=True)
sigma_clip(i32.e2[m96&mask]-i48.e2[m96&mask], weights=None, niter=4, nsig=4, get_err=True)


def sigma_clip(arrin, weights=None, niter=4, nsig=4, get_err=False, get_indices=False, extra={},verbose=False, silent=False, **ignored_kw):
    arr = np.array(arrin, ndmin=1, copy=False)
    if len(arr.shape) > 1:
        raise ValueError("only 1-dimensional arrays suppored, "
                         "got %s" % (arr.shape,))
    if weights is not None:
        weights = np.array(weights, ndmin=1, copy=False)
        if weights.size != arr.size:
            raise ValueError("weights should have same size as "
                             "array, got "
                             "%s %s" % (arr.size,weights.size))
    indices = np.arange( arr.size )
    nold = arr.size
    tarr, tweights = _get_sigma_clip_subset(arr, indices, weights=weights)
    m,e,s=_get_sigma_clip_stats(tarr, weights=tweights)
    if verbose:
        _print_sigma_clip_stats(0, indices.size, m, s)
    for i in xrange(1,niter+1):
        w, = np.where( (np.abs(tarr - m)) < nsig*s)
        if (w.size == 0):
            if (not silent):
                stderr.write("nsig too small. Everything clipped on "
                             "iteration %d\n" % (i+1))
            break
        if w.size == nold:
            break
        indices = indices[w]
        tarr, tweights = _get_sigma_clip_subset(arr, indices, weights=weights)
        nold = w.size
        m,e,s=_get_sigma_clip_stats(tarr, weights=tweights)
        if verbose:
            _print_sigma_clip_stats(i, indices.size, m, s)
    res=[]
    res.append(m)
    res.append(s)
    if get_err:
        res.append(e)
    if get_indices:
        res.append(indices)
    extra['indices'] = indices
    return res

def _get_sigma_clip_subset(arr, indices, weights=None):
    tarr = arr[indices]
    if weights is not None:
        tweights=weights[indices]
    else:
        tweights=None
    return tarr, tweights

def _get_sigma_clip_stats(arr, weights=None):
    if weights is not None:
        m,e,s=wmom(arr, weights, calcerr=True, sdev=True)
    else:
        m = arr.mean()
        s = arr.std()
        e = s/np.sqrt(arr.shape[0])
    return m,e,s

def _print_sigma_clip_stats(iter, nuse, mean, stdev):
    mess='iter: %d  nuse: %d mean: %10.3g stdev: %10.3g'
    print mess % (iter, nuse, mean, stdev)

    # Remove duplicate unique ids (if necessary)
    catalog.CatalogMethods.remove_duplicates(i3)

    # # Read in galaxy catalog
    rm=catalog.CatalogStore('y1_rm_highdens',cattype='gal',cols=['coadd','ra','dec','zp'],catfile=config.redmagicdir+'y1a1_gold_1.0.2b-full_redmapper_v6.4.11_redmagic_highdens_0.5-10.fit',release='y1',ranfile=config.redmagicdir+'y1a1_gold_1.0.2b-full_redmapper_v6.4.11_redmagic_highdens_0.5-10_randoms.fit')

    print 'done with catalog stuff'

    # Pre-cut summary of error properties
    lin.summary_stats.i3_flags_vals_check(i3)
    # Plots location of flagged failures
    #lin.footprint.footprint_tests(i3,['info','error'],p=True)

    print 'done with error stuff'

    # Cut bad objects from catalog
    mask=(i3.info==0)&(i3.rgp>1.13)&(i3.snr>12)&(i3.snr<200)&(i3.rgp<3)&(~(np.isnan(i3.psf1)|np.isnan(i3.psf2)|np.isnan(i3.snr)|np.isnan(i3.psffwhm)))&(i3.g<99)&(i3.r<99)&(i3.i<99)&(i3.z<99)
    catalog.CatalogMethods.match_cat(i3,mask)

    nbc=np.load('/home/troxel/destest/i3nbcv1.npy')
    a=np.argsort(nbc[:,0])
    mask=np.diff(nbc[a,0])
    mask=mask==0
    mask=~mask
    mask=a[mask]
    nbc=nbc[mask]

    x,y=catalog.CatalogMethods.sort2(nbc[:,0],i3.coadd)
    catalog.CatalogMethods.match_cat(i3,y)
    i3.m1=nbc[x,1]
    i3.m2=nbc[x,1]
    i3.c1=nbc[x,2]
    i3.c2=nbc[x,3]
    i3.w=nbc[x,4]
    i3.w/=np.sum(i3.w)

    # # Check summary statistics of catalog values (default all)
    i3.wt=True
    i3.bs=True
    i3.pzrw=True

    print 'done with vals stuff'
    # p!=None (default None) will spawn a multiprocessing Pool() that work is distributed over
    cols=['snr','rgp','ra','dec','psf1','psf2','psffwhm','hsmpsf1','hsmpsf2','psfe','dpsf','psfpos','hsmpsfe','hsmdpsf','hsmpsfpos','g','r','i','z','chi2pix','nlike','gr','ri','iz']

    # # Show the relative disagreement between 2pt statistics when using only galaxies split into each half of a given quantity
    sys_split.split.cat_splits_2pt(i3,rm,cols=cols)

elif int(sys.argv[1])==26:

    import time
    time0=time.time()
    import numpy as np
    import src.catalog as catalog
    import src.corr as corr
    c=np.genfromtxt('partW4.txt',names=['ra', 'dec','e1','e2','weight','fitclass','Z_B','m','c2','star_flag'])
    c=np.genfromtxt('/share/des/disc2/cfhtlens/ellipfull_mask0_WLpass_Wall_Zcut.tsv',names=['ra', 'dec','z','e1','e2','weight','m','c2','field'],dtype=None)
    cc=catalog.CatalogStore('cfhtlens',setup=False,cattype='i3')
    cc.coadd=np.arange(len(c))
    cc.ra=c['ra']
    cc.dec=c['dec']
    cc.e1=c['e1']
    cc.e2=c['e2']
    cc.c2=c['c2']
    cc.w=c['weight']
    cc.m1=c['m']
    cc.m2=c['m']
    cc.c1=np.zeros(len(c))
    cc.bs=True
    cc.wt=True
    cc.sep=[.8,350.]
    cc.tbins=21
    cc.slop=0.1
    theta,out,err,chi2=corr.xi_2pt.xi_2pt(cc,corr='GG',plot=True)
    #theta,out,err,chi2=corr.xi_2pt.xi_2pt(cc,catb=cc,corr='NG',plot=True,ran=False)
    print time.time()-time0

elif int(sys.argv[1])==27:

    import fitsio as fio

    data0=fio.FITS('uband_raw.fits')[-1].read()
    gold=fio.FITS('gold_d04.fits')[-1].read()

    info_cuts =[
        """data0['flux_auto']<=0""",
        """data0['flux_psf']<=0""",
        """data0['flags']>=4""",
        """1.0857362*(data0['fluxerr_psf']/data0['flux_psf']) >= 0.5""",
        """data0['xwin_image']<15""",
        """data0['xwin_image']>2018""",
        """data0['ywin_image']<15""",
        """data0['ywin_image']>4066"""
    ]

    info = np.zeros(len(data0), dtype=np.int64)
    for i,cut in enumerate(info_cuts):
      mask=eval(cut).astype(int)
      print i,cut,np.sum(mask)
      j=1<<i
      flags=mask*j
      info|=flags

    from sklearn.neighbors import KDTree as kdtree
    tree=kdtree(np.vstack((data0['ra'],data0['dec'])).T[info<2**4], leaf_size=2)
    match=tree.query(np.vstack((gold['ra'],gold['dec'])).T, k=1, return_distance=True, sort_results=True)

    mask=np.where(info<2**4)[0]
    coadd=-1*np.ones(len(data0)).astype(int)
    coadd[mask[match[1].reshape(len(match[1]))[match[0].reshape(len(match[0]))<1./60./60.]]]=gold[np.where(match[0].reshape(len(match[0]))<1./60./60.)[0]]

    coadd0=coadd

    spec=fio.FITS('/home/troxel/spec_cat_0.fits.gz')[-1].read()

    tree=kdtree(np.vstack((data0['ra'],data0['dec'])).T[info<2**4], leaf_size=2)
    match0=tree.query_radius(np.vstack((data0['ra'],data0['dec'])).T[info<2**4], r=2./60./60., return_distance=True, count_only=False,sort_results=True)

    num=np.ones(np.max(coadd))
    for j in range(50):
      print j
      for i in range(len(match0[0])):
        if i%10000==0:
          print i
        if coadd[i]==-1:
          if len(match0[0][i])>j:
            if data0['expnum'][match0[0][i][j]]!=data0['expnum'][i]:
              if coadd[match0[0][i][j]]!=-1:
                num[coadd[match0[0][i][j]]-1]+=1
                if num[coadd[match0[0][i][j]]-1]<=10:
                  coadd[i]=coadd[match0[0][i][j]]

    num=0

    # coadd=np.load('coadd.npy')
    coadd0=coadd[coadd!=-1]
    # info=np.load('info.npy')
    uc=np.unique(coadd)
    uc=uc[uc!=-1]

    image=np.ones(len(coadd0), dtype=data0.dtype.descr + [('info',int)]+[('coadd_objects_id',int)]+[('z_spec','f8')]+[('mag_auto_g','f8')]+[('magerr_auto_g','f8')]+[('mag_auto_r','f8')]+[('magerr_auto_r','f8')]+[('mag_auto_i','f8')]+[('magerr_auto_i','f8')]+[('mag_auto_z','f8')]+[('magerr_auto_z','f8')]+[('mag_auto_y','f8')]+[('magerr_auto_y','f8')]+[('modest_class',int)]+[('flags_gold',int)]+[('flags_badregion',int)]+[('hpix',int)]+[('desdm_zp','f8')])

    image2=np.ones(len(uc), dtype=[('mag_auto_u_mean','f8')]+[('mag_auto_u_median','f8')]+[('magerr_auto_u','f8')] + [('info',int)]+[('coadd_objects_id',int)]+[('z_spec','f8')]+[('mag_auto_g','f8')]+[('magerr_auto_g','f8')]+[('mag_auto_r','f8')]+[('magerr_auto_r','f8')]+[('mag_auto_i','f8')]+[('magerr_auto_i','f8')]+[('mag_auto_z','f8')]+[('magerr_auto_z','f8')]+[('mag_auto_y','f8')]+[('magerr_auto_y','f8')]+[('modest_class',int)]+[('flags_gold',int)]+[('flags_badregion',int)]+[('hpix',int)]+[('desdm_zp','f8')])

    for name in image.dtype.names:
      image[name]*=-9999

    for name in image2.dtype.names:
      image2[name]*=-9999

    for i,x in enumerate(uc):
      if x==-1:
        continue
      if i%40!=int(sys.argv[2]):
        continue
      print i,len(uc)
      mask=gold['coadd_objects_id']==x
      mask2=np.in1d(coadd0,x,assume_unique=False)
      mask3=spec['coadd_objects_id']==x
      mask4=info[mask2]==0
      image2['coadd_objects_id'][i]=x
      if np.sum(mask4>0):
        image2['mag_auto_u_mean'][i]=np.mean((data0['mag_auto_u'][mask2])[mask4])
        image2['mag_auto_u_median'][i]=np.median((data0['mag_auto_u'][mask2])[mask4])
        image2['magerr_auto_u'][i]=np.mean((data0['magerr_auto_u'][mask2])[mask4])/np.sqrt(np.sum(mask4))
        image2['info'][i]=0
      else:
        image2['info'][i]=1
      if np.sum(mask3)>0:
        image['z_spec'][mask2]=spec['z_spec'][mask3]
        image2['z_spec'][i]=spec['z_spec'][mask3]
      else:
        image['z_spec'][mask2]=-9999
        image2['z_spec'][i]=-9999
      for name in data0.dtype.names:
        image[name][mask2]=data0[name][mask2]
        image['coadd_objects_id'][mask2]=coadd0[mask2]
        image['info'][mask2]=info[mask2]
      for name in gold.dtype.names:
        if name!='coadd_objects_id':
          if name in image.dtype.names:
            image[name][mask2]=gold[name][mask]
          if name in image2.dtype.names:
            image2[name][i]=gold[name][mask]

    fits = fio.FITS('y3_u_match_'+str(sys.argv[2])+'.fits','rw')
    fits.write(image,clobber=True)
    fits.close()

    fits = fio.FITS('y3_u_match2_'+str(sys.argv[2])+'.fits','rw')
    fits.write(image2,clobber=True)
    fits.close()

elif int(sys.argv[1])==28:

    cfht=fio.FITS('/share/des/disc2/cfhtlens/cfht_vipers_shape_match.fits')[-1].read()
    # nb still need to exclude z_flag==0,1 objects
    cfht=cfht[((cfht['z_flag']==2)|(cfht['z_flag']==3)|(cfht['z_flag']==4)|(cfht['z_flag']==9))&(cfht['z_spec']>0)&(cfht['weight']>0)]

    vipers1=fio.FITS('/share/des/disc2/cfhtlens/VIPERS_W1_SPECTRO_PDR1.fits.gz')[-1].read()
    vipers4=fio.FITS('/share/des/disc2/cfhtlens/VIPERS_W4_SPECTRO_PDR1.fits.gz')[-1].read()
    vipers=np.append(vipers1,vipers4)
    vipers['zflg']=np.floor(vipers['zflg'])
    vipers=vipers[((vipers['zflg']==2)|(vipers['zflg']==3)|(vipers['zflg']==4)|(vipers['zflg']==9))&(vipers['zspec']>0)]

    mask=np.load('wfirstvipersmask.npy')
    #limit to joint photo-spec mask
    import healpy as hp
    cpix=hp.ang2pix(4096, np.pi/2.-np.radians(cfht['dec']),np.radians(cfht['ra']), nest=False)
    vpix=hp.ang2pix(4096, np.pi/2.-np.radians(vipers['delta']),np.radians(vipers['alpha']), nest=False)
    cmask=np.in1d(cpix,mask,assume_unique=False)
    vmask=np.in1d(vpix,mask,assume_unique=False)
    cfht=cfht[cmask]
    vipers=vipers[vmask]

    import scipy.special
    csr=.5-.5*scipy.special.erf(10.8*(.44-vipers['zspec']))
    vipersweight=(1./vipers['tsr'])*(1./vipers['ssr'])#*(1./csr)
    vipers=vipers[(~np.isinf(vipersweight))&(vipersweight>0)]
    vipersweight=vipersweight[(~np.isinf(vipersweight))&(vipersweight>0)]

    h,z=np.histogram(vipers['zspec'],weights=vipersweight,bins=50)
    z=(z[:-1]+z[1:])/2.
    import scipy.interpolate as interp
    f=interp.interp1d(z, h/np.sum(h), kind='cubic',bounds_error=False,fill_value=0.)

    ran=fio.FITS('cfhtvipers_large_random.fits.gz')[-1].read()
    # ranz=np.random.choice(np.arange(1000000)*2./1000000., size=len(ran), replace=True, p=f(np.arange(1000000)*2./1000000.)/np.sum(f(np.arange(1000000)*2./1000000.)))

    import kmeans_radec as km
    km0 = km.kmeans_sample(np.vstack((vipers['alpha'],vipers['delta'])).T, 50, maxiter=100, tol=1.0e-5)
    vipersreg=km0.labels
    cfhtreg=km0.find_nearest(np.vstack((cfht['ra'],cfht['dec'])).T)
    ranreg=km0.find_nearest(np.vstack((ran['ra'],ran['dec'])).T)

    # cfhtr=chi(cfht['z_spec'])
    # vipersr=chi(vipers['zspec'])
    # ranr=chi(ranz)

    # np.save('cfhtr.npy',cfhtr)
    # np.save('vipersr.npy',vipersr)
    # np.save('ranr.npy',ranr)

    cfhtr=np.load('cfhtr.npy')
    vipersr=np.load('vipersr.npy')
    ranr=np.load('ranr.npy')

    ranmask=np.random.choice(np.arange(len(ran)),1000000,replace=False)
    ran=ran[ranmask]
    ranr=ranr[ranmask]
    ranreg=ranreg[ranmask]

    r=np.zeros((51,10))
    ngxi=np.zeros((51,10))
    ngxiwt=np.zeros((51,10))
    ngxinp=np.zeros((51,10))
    rgxi=np.zeros((51,10))
    rgxiwt=np.zeros((51,10))
    rgxinp=np.zeros((51,10))
    nkxi=np.zeros((51,10))
    nkxiwt=np.zeros((51,10))
    nkxinp=np.zeros((51,10))
    rkxi=np.zeros((51,10))
    rkxiwt=np.zeros((51,10))
    rkxinp=np.zeros((51,10))
    ggxip=np.zeros((51,10))
    ggxim=np.zeros((51,10))
    ggxiwt=np.zeros((51,10))
    ggxinp=np.zeros((51,10))
    kkxi=np.zeros((51,10))
    kkxiwt=np.zeros((51,10))
    kkxinp=np.zeros((51,10))
    rrxiwt=np.zeros((51,10))


    import treecorr as t
    dlos=60.
    maxlos=np.max([np.max(cfhtr),np.max(vipersr),np.max(ranr)])#4200.
    minlos=np.min([np.min(cfhtr),np.min(vipersr),np.min(ranr)])#200.
    for reg in range(50):
      print 'reg',reg
      gg = t.GGCorrelation(nbins=10, min_sep=.6, max_sep=100.,bin_slop=.1,verbose=0)
      kk = t.KKCorrelation(nbins=10, min_sep=.6, max_sep=100.,bin_slop=.1,verbose=0)
      ng = t.NGCorrelation(nbins=10, min_sep=.6, max_sep=100.,bin_slop=.1,verbose=0)
      nk = t.NKCorrelation(nbins=10, min_sep=.6, max_sep=100.,bin_slop=.1,verbose=0)
      rg = t.NGCorrelation(nbins=10, min_sep=.6, max_sep=100.,bin_slop=.1,verbose=0)
      rk = t.NKCorrelation(nbins=10, min_sep=.6, max_sep=100.,bin_slop=.1,verbose=0)
      rr = t.NNCorrelation(nbins=10, min_sep=.6, max_sep=100.,bin_slop=.1,verbose=0)
      for i in range(1000):
        if i%100==0:
          print i
        rc=(maxlos-minlos)/1000*i+minlos
        maskc=(np.abs(cfhtr-rc)<dlos)&(cfhtreg==reg)
        maskv=(np.abs(vipersr-rc)<(maxlos-minlos)/1000/2.)&(vipersreg==reg)
        maskr=(np.abs(ranr-rc)<(maxlos-minlos)/1000/2.)&(ranreg==reg)
        if np.sum(maskc)!=0:
          cg=t.Catalog(g1=cfht['e1'][maskc], g2=cfht['e2'][maskc]-cfht['c2'][maskc], w=cfht['weight'][maskc], ra=cfht['ra'][maskc], dec=cfht['dec'][maskc],r=cfhtr[maskc], ra_units='deg', dec_units='deg')
          ck=t.Catalog(k=(1.+cfht['m'][maskc]), w=cfht['weight'][maskc], ra=cfht['ra'][maskc], dec=cfht['dec'][maskc],r=cfhtr[maskc], ra_units='deg', dec_units='deg')
        if np.sum(maskv)!=0:
          vn=t.Catalog(ra=vipers['alpha'][maskv], dec=vipers['delta'][maskv], w=vipersweight[maskv],r=vipersr[maskv], ra_units='deg', dec_units='deg')
        if np.sum(maskr)!=0:
          vr=t.Catalog(ra=ran['ra'][maskr], dec=ran['dec'][maskr],r=ranr[maskr], ra_units='deg', dec_units='deg')
        if (np.sum(maskc)!=0):
          gg.process_cross(cg,cg,metric='Rperp')
          kk.process_cross(ck,ck,metric='Rperp')
        if (np.sum(maskc)!=0)&(np.sum(maskv)!=0):
          ng.process_cross(vn,cg,metric='Rperp')
          nk.process_cross(vn,ck,metric='Rperp')
        if np.sum(maskr)!=0:
          if (np.sum(maskc)!=0):
            rg.process_cross(vr,cg,metric='Rperp')
            rk.process_cross(vr,ck,metric='Rperp')
          rr.process_cross(vr,vr,metric='Rperp')

      r[0,:]+=ng.meanr
      ngxi[0,:]+=ng.xi
      ngxim[0,:]+=ng.xi_im
      ngxiwt[0,:]+=ng.weight
      ngxinp[0,:]+=ng.npairs
      rgxi[0,:]+=rg.xi
      rgxim[0,:]+=rg.xi_im
      rgxiwt[0,:]+=rg.weight
      rgxinp[0,:]+=rg.npairs
      nkxi[0,:]+=nk.xi
      nkxiwt[0,:]+=nk.weight
      nkxinp[0,:]+=nk.npairs
      rkxi[0,:]+=rk.xi
      rkxiwt[0,:]+=rk.weight
      rkxinp[0,:]+=rk.npairs
      ggxip[0,:]+=gg.xip
      ggxim[0,:]+=gg.xim
      ggxiwt[0,:]+=gg.weight
      ggxinp[0,:]+=gg.npairs
      kkxi[0,:]+=kk.xi
      kkxiwt[0,:]+=kk.weight
      kkxinp[0,:]+=kk.npairs
      rrxiwt[0,:]+=rr.weight

      r[reg+1,:]=ng.meanr
      ngxi[reg+1,:]=ng.xi
      ngxim[reg+1,:]=ng.xi_im
      ngxiwt[reg+1,:]=ng.weight
      ngxinp[reg+1,:]=ng.npairs
      rgxi[reg+1,:]=rg.xi
      rgxim[reg+1,:]=rg.xi_im
      rgxiwt[reg+1,:]=rg.weight
      rgxinp[reg+1,:]=rg.npairs
      nkxi[reg+1,:]=nk.xi
      nkxiwt[reg+1,:]=nk.weight
      nkxinp[reg+1,:]=nk.npairs
      rkxi[reg+1,:]=rk.xi
      rkxiwt[reg+1,:]=rk.weight
      rkxinp[reg+1,:]=rk.npairs
      ggxip[reg+1,:]=gg.xip
      ggxim[reg+1,:]=gg.xim
      ggxiwt[reg+1,:]=gg.weight
      ggxinp[reg+1,:]=gg.npairs
      kkxi[reg+1,:]=kk.xi
      kkxiwt[reg+1,:]=kk.weight
      kkxinp[reg+1,:]=kk.npairs
      rrxiwt[reg+1,:]=rr.weight

    np.save('cv_r_60.npy',r)
    np.save('cv_ng_xi_60.npy',ngxi)
    np.save('cv_ng_xim_60.npy',ngxim)
    np.save('cv_ng_xi_wt_60.npy',ngxiwt)
    np.save('cv_ng_xi_np_60.npy',ngxinp)
    np.save('cv_rg_xi_60.npy',rgxi)
    np.save('cv_rg_xim_60.npy',rgxim)
    np.save('cv_rg_xi_wt_60.npy',rgxiwt)
    np.save('cv_rg_xi_np_60.npy',rgxinp)
    np.save('cv_nk_xi_60.npy',nkxi)
    np.save('cv_nk_xi_wt_60.npy',nkxiwt)
    np.save('cv_nk_xi_np_60.npy',nkxinp)
    np.save('cv_rk_xi_60.npy',rkxi)
    np.save('cv_rk_xi_wt_60.npy',rkxiwt)
    np.save('cv_rk_xi_np_60.npy',rkxinp)
    np.save('cv_gg_xip_60.npy',ggxip)
    np.save('cv_gg_xim_60.npy',ggxim)
    np.save('cv_gg_xi_wt_60.npy',ggxiwt)
    np.save('cv_gg_xi_np_60.npy',ggxinp)
    np.save('cv_kk_xi_60.npy',kkxi)
    np.save('cv_kk_xi_wt_60.npy',kkxiwt)
    np.save('cv_kk_xi_np_60.npy',kkxinp)
    np.save('cv_rr_xi_wt_60.npy',rrxiwt)


elif int(sys.argv[1])==29:

cfht=fio.FITS('cfht_vipers_shape_match.fits')[-1].read()
# nb still need to exclude z_flag==0,1 objects
cfht=cfht[((cfht['z_flag']==2)|(cfht['z_flag']==3)|(cfht['z_flag']==4)|(cfht['z_flag']==9))&(cfht['z_spec']>0)&(cfht['weight']>0)]

vipers1=fio.FITS('VIPERS_W1_SPECTRO_PDR1.fits.gz')[-1].read()
vipers4=fio.FITS('VIPERS_W4_SPECTRO_PDR1.fits.gz')[-1].read()
vipers=np.append(vipers1,vipers4)
vipers['zflg']=np.floor(vipers['zflg'])
vipers=vipers[((vipers['zflg']==2)|(vipers['zflg']==3)|(vipers['zflg']==4)|(vipers['zflg']==9))&(vipers['zspec']>0)]

mask=np.load('wfirstvipersmask.npy')
#limit to joint photo-spec mask
import healpy as hp
cpix=hp.ang2pix(4096, np.pi/2.-np.radians(cfht['dec']),np.radians(cfht['ra']), nest=False)
vpix=hp.ang2pix(4096, np.pi/2.-np.radians(vipers['delta']),np.radians(vipers['alpha']), nest=False)
cmask=np.in1d(cpix,mask,assume_unique=False)
vmask=np.in1d(vpix,mask,assume_unique=False)
cfht=cfht[cmask]
vipers=vipers[vmask]

import scipy.special
csr=.5-.5*scipy.special.erf(10.8*(.44-vipers['zspec']))
vipersweight=(1./vipers['tsr'])*(1./vipers['ssr'])#*(1./csr)
vipers=vipers[(~np.isinf(vipersweight))&(vipersweight>0)]
vipersweight=vipersweight[(~np.isinf(vipersweight))&(vipersweight>0)]

h,z=np.histogram(vipers['zspec'],weights=vipersweight,bins=50)
z=(z[:-1]+z[1:])/2.
import scipy.interpolate as interp
f=interp.interp1d(z, h/np.sum(h), kind='cubic',bounds_error=False,fill_value=0.)

ran=fio.FITS('cfhtvipers_large_random.fits.gz')[-1].read()
# ranz=np.random.choice(np.arange(1000000)*2./1000000., size=len(ran), replace=True, p=f(np.arange(1000000)*2./1000000.)/np.sum(f(np.arange(1000000)*2./1000000.)))

# import kmeans_radec as km
# km0 = km.kmeans_sample(np.vstack((vipers['alpha'],vipers['delta'])).T, 50, maxiter=100, tol=1.0e-5)
# vipersreg=km0.labels
# cfhtreg=km0.find_nearest(np.vstack((cfht['ra'],cfht['dec'])).T)
# ranreg=km0.find_nearest(np.vstack((ran['ra'],ran['dec'])).T)

# cfhtr=chi(cfht['z_spec'])
# vipersr=chi(vipers['zspec'])
# ranr=chi(ranz)

# np.save('cfhtr.npy',cfhtr)
# np.save('vipersr.npy',vipersr)
# np.save('ranr.npy',ranr)

cfhtr=np.load('cfhtr.npy')
vipersr=np.load('vipersr.npy')
ranr=np.load('ranr.npy')

ranmask=np.random.choice(np.arange(len(ran)),1000000,replace=False)
ran=ran[ranmask]
ranr=ranr[ranmask]
# ranreg=ranreg[ranmask]

import treecorr

# cg=treecorr.Catalog(file_name='cg.fits',num=-1,ra_col='ra',dec_col='dec',r_col='r',w_col='w',g1_col='g1',g2_col='g2',ra_units='deg', dec_units='deg')
# vn=treecorr.Catalog(file_name='vn.fits',num=-1,ra_col='ra',dec_col='dec',r_col='r',w_col='w',ra_units='deg', dec_units='deg')
# de = treecorr.NGCorrelation(nbins=10, min_sep=.6, max_sep=60, min_rpar = 100, max_rpar = 6000, bin_slop=0.01, verbose=0)
# de2 = treecorr.NGCorrelation(nbins=10, min_sep=.6, max_sep=60, min_rpar = -60, max_rpar = 60, bin_slop=0.01, verbose=0)
# de.process(vn,cg,metric='Rperp')
# de2.process(vn,cg,metric='Rperp')

cg=treecorr.Catalog(g1=-cfht['e1'], g2=-(cfht['e2']-cfht['c2']), w=cfht['weight'], ra=cfht['ra'], dec=cfht['dec'],r=cfhtr, ra_units='deg', dec_units='deg')
ck=treecorr.Catalog(k=(1.+cfht['m']), w=cfht['weight'], ra=cfht['ra'], dec=cfht['dec'],r=cfhtr, ra_units='deg', dec_units='deg')
vn=treecorr.Catalog(ra=vipers['alpha'], dec=vipers['delta'], w=vipersweight,r=vipersr, ra_units='deg', dec_units='deg')
vr=treecorr.Catalog(ra=ran['ra'], dec=ran['dec'],r=ranr, ra_units='deg', dec_units='deg')

de = treecorr.NGCorrelation(nbins=10, min_sep=.6, max_sep=60, min_rpar = 100, max_rpar = 6000, bin_slop=0.01, verbose=0)
dm = treecorr.NKCorrelation(nbins=10, min_sep=.6, max_sep=60, min_rpar = 100, max_rpar = 6000, bin_slop=0.01, verbose=0)
re = treecorr.NGCorrelation(nbins=10, min_sep=.6, max_sep=60, min_rpar = 100, max_rpar = 6000, bin_slop=0.01, verbose=0)
rm = treecorr.NKCorrelation(nbins=10, min_sep=.6, max_sep=60, min_rpar = 100, max_rpar = 6000, bin_slop=0.01, verbose=0)
de2 = treecorr.NGCorrelation(nbins=10, min_sep=.6, max_sep=60, min_rpar = -60, max_rpar = 60, bin_slop=0.01, verbose=0)
dm2 = treecorr.NKCorrelation(nbins=10, min_sep=.6, max_sep=60, min_rpar = -60, max_rpar = 60, bin_slop=0.01, verbose=0)
re2 = treecorr.NGCorrelation(nbins=10, min_sep=.6, max_sep=60, min_rpar = -60, max_rpar = 60, bin_slop=0.01, verbose=0)
rm2 = treecorr.NKCorrelation(nbins=10, min_sep=.6, max_sep=60, min_rpar = -60, max_rpar = 60, bin_slop=0.01, verbose=0)

de.process(vn,cg,metric='Rperp')
dm.process(vn,ck,metric='Rperp')
re.process(vr,cg,metric='Rperp')
rm.process(vr,ck,metric='Rperp')
de2.process(vn,cg,metric='Rperp')
dm2.process(vn,ck,metric='Rperp')
re2.process(vr,cg,metric='Rperp')
rm2.process(vr,ck,metric='Rperp')

xip=de2.xi/dm2.xi-re2.xi_im/rm2.xi
xim=de2.xi_im/dm2.xi-re2.xi/rm2.xi
varxi=np.sqrt(de.varxi)
varxi2=np.sqrt(de2.varxi)
print xip*120,varxi
print xim*120*.1,varxi2


    # r=np.zeros((51,10))
    # ngxi=np.zeros((51,10))
    # ngxiwt=np.zeros((51,10))
    # ngxinp=np.zeros((51,10))
    # rgxi=np.zeros((51,10))
    # rgxiwt=np.zeros((51,10))
    # rgxinp=np.zeros((51,10))
    # nkxi=np.zeros((51,10))
    # nkxiwt=np.zeros((51,10))
    # nkxinp=np.zeros((51,10))
    # rkxi=np.zeros((51,10))
    # rkxiwt=np.zeros((51,10))
    # rkxinp=np.zeros((51,10))
    # ggxip=np.zeros((51,10))
    # ggxim=np.zeros((51,10))
    # ggxiwt=np.zeros((51,10))
    # ggxinp=np.zeros((51,10))
    # kkxi=np.zeros((51,10))
    # kkxiwt=np.zeros((51,10))
    # kkxinp=np.zeros((51,10))
    # rrxiwt=np.zeros((51,10))

    # import treecorr as t
    # dlos=100.
    # maxlos=np.max([np.max(cfhtr),np.max(vipersr),np.max(ranr)])#4200.
    # minlos=np.min([np.min(cfhtr),np.min(vipersr),np.min(ranr)])#200.
    # for reg in range(50):
    #   print 'reg',reg
    #   gg = t.GGCorrelation(nbins=10, min_sep=.6, max_sep=100.,bin_slop=.1,verbose=0)
    #   kk = t.KKCorrelation(nbins=10, min_sep=.6, max_sep=100.,bin_slop=.1,verbose=0)
    #   ng = t.NGCorrelation(nbins=10, min_sep=.6, max_sep=100.,bin_slop=.1,verbose=0)
    #   nk = t.NKCorrelation(nbins=10, min_sep=.6, max_sep=100.,bin_slop=.1,verbose=0)
    #   rg = t.NGCorrelation(nbins=10, min_sep=.6, max_sep=100.,bin_slop=.1,verbose=0)
    #   rk = t.NKCorrelation(nbins=10, min_sep=.6, max_sep=100.,bin_slop=.1,verbose=0)
    #   rr = t.NNCorrelation(nbins=10, min_sep=.6, max_sep=100.,bin_slop=.1,verbose=0)
    #   for i in range(1000):
    #     if i%100==0:
    #       print i
    #     rc=(maxlos-minlos)/1000*i+minlos
    #     maskc=(np.abs(cfhtr-rc)<dlos)&(cfhtreg==reg)
    #     maskv=(np.abs(vipersr-rc)<(maxlos-minlos)/1000/2.)&(vipersreg==reg)
    #     maskr=(np.abs(ranr-rc)<(maxlos-minlos)/1000/2.)&(ranreg==reg)
    #     if np.sum(maskc)!=0:
    #       cg=t.Catalog(g1=cfht['e1'][maskc], g2=cfht['e2'][maskc]-cfht['c2'][maskc], w=cfht['weight'][maskc], ra=cfht['ra'][maskc], dec=cfht['dec'][maskc],r=cfhtr[maskc], ra_units='deg', dec_units='deg')
    #       ck=t.Catalog(k=(1.+cfht['m'][maskc]), w=cfht['weight'][maskc], ra=cfht['ra'][maskc], dec=cfht['dec'][maskc],r=cfhtr[maskc], ra_units='deg', dec_units='deg')
    #     if np.sum(maskv)!=0:
    #       vn=t.Catalog(ra=vipers['alpha'][maskv], dec=vipers['delta'][maskv], w=vipersweight[maskv],r=vipersr[maskv], ra_units='deg', dec_units='deg')
    #     if np.sum(maskr)!=0:
    #       vr=t.Catalog(ra=ran['ra'][maskr], dec=ran['dec'][maskr],r=ranr[maskr], ra_units='deg', dec_units='deg')
    #     if (np.sum(maskc)!=0):
    #       gg.process_cross(cg,cg,metric='Rperp')
    #       kk.process_cross(ck,ck,metric='Rperp')
    #     if (np.sum(maskc)!=0)&(np.sum(maskv)!=0):
    #       ng.process_cross(vn,cg,metric='Rperp')
    #       nk.process_cross(vn,ck,metric='Rperp')
    #     if np.sum(maskr)!=0:
    #       if (np.sum(maskc)!=0):
    #         rg.process_cross(vr,cg,metric='Rperp')
    #         rk.process_cross(vr,ck,metric='Rperp')
    #       rr.process_cross(vr,vr,metric='Rperp')

    #   r[0,:]+=ng.meanr
    #   ngxi[0,:]+=ng.xi
    #   ngxim[0,:]+=ng.xi_im
    #   ngxiwt[0,:]+=ng.weight
    #   ngxinp[0,:]+=ng.npairs
    #   rgxi[0,:]+=rg.xi
    #   rgxim[0,:]+=rg.xi_im
    #   rgxiwt[0,:]+=rg.weight
    #   rgxinp[0,:]+=rg.npairs
    #   nkxi[0,:]+=nk.xi
    #   nkxiwt[0,:]+=nk.weight
    #   nkxinp[0,:]+=nk.npairs
    #   rkxi[0,:]+=rk.xi
    #   rkxiwt[0,:]+=rk.weight
    #   rkxinp[0,:]+=rk.npairs
    #   ggxip[0,:]+=gg.xip
    #   ggxim[0,:]+=gg.xim
    #   ggxiwt[0,:]+=gg.weight
    #   ggxinp[0,:]+=gg.npairs
    #   kkxi[0,:]+=kk.xi
    #   kkxiwt[0,:]+=kk.weight
    #   kkxinp[0,:]+=kk.npairs
    #   rrxiwt[0,:]+=rr.weight

    #   r[reg+1,:]=ng.meanr
    #   ngxi[reg+1,:]=ng.xi
    #   ngxim[reg+1,:]=ng.xi_im
    #   ngxiwt[reg+1,:]=ng.weight
    #   ngxinp[reg+1,:]=ng.npairs
    #   rgxi[reg+1,:]=rg.xi
    #   rgxim[reg+1,:]=rg.xi_im
    #   rgxiwt[reg+1,:]=rg.weight
    #   rgxinp[reg+1,:]=rg.npairs
    #   nkxi[reg+1,:]=nk.xi
    #   nkxiwt[reg+1,:]=nk.weight
    #   nkxinp[reg+1,:]=nk.npairs
    #   rkxi[reg+1,:]=rk.xi
    #   rkxiwt[reg+1,:]=rk.weight
    #   rkxinp[reg+1,:]=rk.npairs
    #   ggxip[reg+1,:]=gg.xip
    #   ggxim[reg+1,:]=gg.xim
    #   ggxiwt[reg+1,:]=gg.weight
    #   ggxinp[reg+1,:]=gg.npairs
    #   kkxi[reg+1,:]=kk.xi
    #   kkxiwt[reg+1,:]=kk.weight
    #   kkxinp[reg+1,:]=kk.npairs
    #   rrxiwt[reg+1,:]=rr.weight

    # np.save('cv_r_100.npy',r)
    # np.save('cv_ng_xi_100.npy',ngxi)
    # np.save('cv_ng_xim_100.npy',ngxim)
    # np.save('cv_ng_xi_wt_100.npy',ngxiwt)
    # np.save('cv_ng_xi_np_100.npy',ngxinp)
    # np.save('cv_rg_xi_100.npy',rgxi)
    # np.save('cv_rg_xim_100.npy',rgxim)
    # np.save('cv_rg_xi_wt_100.npy',rgxiwt)
    # np.save('cv_rg_xi_np_100.npy',rgxinp)
    # np.save('cv_nk_xi_100.npy',nkxi)
    # np.save('cv_nk_xi_wt_100.npy',nkxiwt)
    # np.save('cv_nk_xi_np_100.npy',nkxinp)
    # np.save('cv_rk_xi_100.npy',rkxi)
    # np.save('cv_rk_xi_wt_100.npy',rkxiwt)
    # np.save('cv_rk_xi_np_100.npy',rkxinp)
    # np.save('cv_gg_xip_100.npy',ggxip)
    # np.save('cv_gg_xim_100.npy',ggxim)
    # np.save('cv_gg_xi_wt_100.npy',ggxiwt)
    # np.save('cv_gg_xi_np_100.npy',ggxinp)
    # np.save('cv_kk_xi_100.npy',kkxi)
    # np.save('cv_kk_xi_wt_100.npy',kkxiwt)
    # np.save('cv_kk_xi_np_100.npy',kkxinp)
    # np.save('cv_rr_xi_wt_100.npy',rrxiwt)

elif int(sys.argv[1])==30:

def chi(z,omegam=0.27,H=100):
  # import cosmology  
  # c0=cosmology.Cosmo(H0=H,omega_m=omegam)
  # return c0.Dc(0.,z)
  from astropy import cosmology
  from astropy.cosmology import FlatLambdaCDM
  cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
  return cosmo.comoving_distance(z).value

cfht=np.genfromtxt('L6.dat',names=['ra','dec','z','4','5','6','7','8','9','10','e1','e2','13','14','15','16','reg'])
ran=np.genfromtxt('RL6.dat',names=['ra','dec','z','4','5','6','7','8','9','10','e1','e2','13','14','15','16','reg'])

cfhtr=chi(cfht['z'])
ranr=chi(ran['z'])

import treecorr



cated=treecorr.Catalog(g1=cfht['e1'], g2=-cfht['e2'], ra=cfht['ra'], dec=cfht['dec'], r=cfhtr, ra_units='deg', dec_units='deg')
catrd=treecorr.Catalog(ra=ran['ra'], dec=ran['dec'], r=ranr, ra_units='deg', dec_units='deg')
de2 = treecorr.NGCorrelation(nbins=10, min_sep=.3, max_sep=60, min_rpar = -60, max_rpar = 60, bin_slop=0.01, verbose=0)
re2 = treecorr.NGCorrelation(nbins=10, min_sep=.3, max_sep=60, min_rpar = -60, max_rpar = 60, bin_slop=0.01, verbose=0)
de2.process(cated,cated,metric='Rperp')
re2.process(catrd,cated,metric='Rperp')
xip,xim,varxi=de2.calculateXi(rg=re2)
print xip*120,varxi
print xim*120*.1,varxi

r=np.exp(de2.meanlogr)
fig.plot_methods.plot_IA(r,[xip*np.sqrt(r)*120,xim*np.sqrt(r)*120],[varxi*np.sqrt(r)*120,varxi*np.sqrt(r)*120],'sdss')



comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

ms=catalog.CatalogStore('ms',setup=False)
tmp=fio.FITS('ms_megazlike_Est_Sma_est_sma_0_0.fits')[-1].read()

ms.ra=tmp['x_pos[rad]']
ms.dec=tmp['y_pos[rad]']
ms.coadd=np.arange(len(ms.ra))
ms.r=tmp['comoving_distance[Mpc/h]']
ms.e1=tmp['e_1']
ms.e2=-tmp['e_2']

cated=treecorr.Catalog(g1=ms.e1, g2=ms.e1, ra=ms.ra, dec=ms.dec, r=ms.r, ra_units='rad', dec_units='rad')

de = treecorr.NGCorrelation(nbins=12, min_sep=.4, max_sep=100, min_rpar = -10, max_rpar = 10, bin_slop=0., verbose=0)

de.process(cated,cated,metric='Rperp')
de.xi*200

[ -1.09632292e-02,  -6.25599431e-03,  -3.65967894e-03,  -1.53056308e-03,  -8.01473195e-04,  -6.79558919e-04,  -7.74421378e-04,  -4.52932660e-04,  -2.37235557e-04,  -5.86310354e-05,  -3.64357847e-04,  -6.40552955e-04]

plt.errorbar(b_10[:,0],b_10[:,3],yerr=b_10[:,6],ls='',marker='.',label='bj')
plt.errorbar(t_10[:,0]*1.1,-t_10[:,1],ls='',marker='.',label='treecorr')
plt.errorbar(p_10[:,0]*1.2,p_10[:,1],ls='',marker='.',label='python')
plt.xscale('log')
plt.xlabel(r'$R_{perp}$')
plt.ylabel(r'$w_{g+}$')
plt.legend(loc='upper right',ncol=2, frameon=True,prop={'size':12})
plt.savefig('sdss.png')
plt.close()

plt.errorbar(b_10[:,0],b_10[:,4],yerr=b_10[:,6],ls='',marker='.',label='bj')
plt.errorbar(t_10[:,0]*1.1,t_10[:,2],ls='',marker='.',label='treecorr')
plt.errorbar(p_10[:,0]*1.2,p_10[:,2],ls='',marker='.',label='python')
plt.xscale('log')
plt.xlabel(r'$R_{perp}$')
plt.ylabel(r'$w_{gx}$')
plt.legend(loc='upper right',ncol=2, frameon=True,prop={'size':12})
plt.savefig('wgx.png')
plt.close()

plt.errorbar(b_10[:,0],b_10[:,7],ls='',marker='.',label='bj')
plt.errorbar(t_10[:,0]*1.1,t_10[:,3],ls='',marker='.',label='treecorr')
plt.errorbar(p_10[:,0]*1.2,p_10[:,3],ls='',marker='.',label='python')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$R_{perp}$')
plt.ylabel('npairs')
plt.legend(loc='upper left',ncol=2, frameon=True,prop={'size':12})
plt.savefig('npairs.png')
plt.close()

    # regnum=0

    # r=np.zeros((regnum+1,10))
    # ngxi=np.zeros((regnum+1,10))
    # ngxim=np.zeros((regnum+1,10))
    # ngxiwt=np.zeros((regnum+1,10))
    # ngxinp=np.zeros((regnum+1,10))
    # rgxi=np.zeros((regnum+1,10))
    # rgxim=np.zeros((regnum+1,10))
    # rgxiwt=np.zeros((regnum+1,10))
    # rgxinp=np.zeros((regnum+1,10))
    # nkxi=np.zeros((regnum+1,10))
    # nkxiwt=np.zeros((regnum+1,10))
    # nkxinp=np.zeros((regnum+1,10))
    # rkxi=np.zeros((regnum+1,10))
    # rkxiwt=np.zeros((regnum+1,10))
    # rkxinp=np.zeros((regnum+1,10))
    # ggxip=np.zeros((regnum+1,10))
    # ggxim=np.zeros((regnum+1,10))
    # ggxiwt=np.zeros((regnum+1,10))
    # ggxinp=np.zeros((regnum+1,10))
    # kkxi=np.zeros((regnum+1,10))
    # kkxiwt=np.zeros((regnum+1,10))
    # kkxinp=np.zeros((regnum+1,10))
    # rrxiwt=np.zeros((regnum+1,10))
    # import treecorr as t
    # dlos=60.
    # maxlos=np.max([np.max(cfhtr),np.max(vipersr),np.max(ranr)])#4200.
    # minlos=np.min([np.min(cfhtr),np.min(vipersr),np.min(ranr)])#200.
    # # for reg in range(50):
    # #   print 'reg',reg
    # reg=0
    # gg = t.GGCorrelation(nbins=10, min_sep=.6, max_sep=60.,bin_slop=0.,verbose=0)
    # kk = t.KKCorrelation(nbins=10, min_sep=.6, max_sep=60.,bin_slop=0.,verbose=0)
    # ng = t.NGCorrelation(nbins=10, min_sep=.6, max_sep=60.,bin_slop=0.,verbose=0)
    # nk = t.NKCorrelation(nbins=10, min_sep=.6, max_sep=60.,bin_slop=0.,verbose=0)
    # rg = t.NGCorrelation(nbins=10, min_sep=.6, max_sep=60.,bin_slop=0.,verbose=0)
    # rk = t.NKCorrelation(nbins=10, min_sep=.6, max_sep=60.,bin_slop=0.,verbose=0)
    # rr = t.NNCorrelation(nbins=10, min_sep=.6, max_sep=60.,bin_slop=0.,verbose=0)
    # for i in range(len(ran)):
    #   if i%100==0:
    #     print i
    #   rc=(maxlos-minlos)/1000*i+minlos
    #   maskc=(np.abs(cfhtr-vipersr[i])<dlos)#(np.abs(cfhtr-rc)<dlos)
    #   maskv=i#(np.abs(vipersr-rc)<(maxlos-minlos)/1000/2.)
    #   maskr=i#(np.abs(ranr-rc)<(maxlos-minlos)/1000/2.)
    #   maskr2=(np.abs(ranr-ranr[i])<dlos)#(np.abs(ranr-rc)<(maxlos-minlos)/1000/2.)
    #   if np.sum(maskc)!=0:
    #     cg=t.Catalog(g1=cfht['e1'][maskc], g2=cfht['e2'][maskc], ra=cfht['ra'][maskc], dec=cfht['dec'][maskc],r=cfhtr[maskc], ra_units='deg', dec_units='deg')
    #   if np.sum(maskv)!=0:
    #     vn=t.Catalog(ra=np.atleast_1d(vipers['ra'][maskv]), dec=np.atleast_1d(vipers['dec'][maskv]),r=np.atleast_1d(vipersr[maskv]), ra_units='deg', dec_units='deg')
    #   if np.sum(maskr)!=0:
    #     vr=t.Catalog(ra=np.atleast_1d(ran['ra'][maskr]), dec=np.atleast_1d(ran['dec'][maskr]),r=np.atleast_1d(ranr[maskr]), ra_units='deg', dec_units='deg')
    #     vr2=t.Catalog(ra=np.atleast_1d(ran['ra'][maskr2]), dec=np.atleast_1d(ran['dec'][maskr2]),r=np.atleast_1d(ranr[maskr2]), ra_units='deg', dec_units='deg')
    #   if (np.sum(maskc)!=0):
    #     gg.process_cross(cg,cg,metric='Euclidean')#Rperp
    #   if (np.sum(maskc)!=0)&(np.sum(maskv)!=0):
    #     ng.process_cross(vn,cg,metric='Euclidean')
    #   if np.sum(maskr)!=0:
    #     if (np.sum(maskc)!=0):
    #       rg.process_cross(vr,cg,metric='Euclidean')
    #     rr.process_cross(vr,vr2,metric='Euclidean')
    #   r[0,:]+=ng.meanr
    #   ngxi[0,:]+=ng.xi
    #   ngxim[0,:]+=ng.xi_im
    #   ngxiwt[0,:]+=ng.weight
    #   ngxinp[0,:]+=ng.npairs
    #   rgxi[0,:]+=rg.xi
    #   rgxim[0,:]+=rg.xi_im
    #   rgxiwt[0,:]+=rg.weight
    #   rgxinp[0,:]+=rg.npairs
    #   ggxip[0,:]+=gg.xip
    #   ggxim[0,:]+=gg.xim
    #   ggxiwt[0,:]+=gg.weight
    #   ggxinp[0,:]+=gg.npairs
    #   rrxiwt[0,:]+=rr.weight
    #   # r[reg+1,:]=ng.meanr
    #   # ngxi[reg+1,:]=ng.xi
    #   # ngxim[reg+1,:]=ng.xi_im
    #   # ngxiwt[reg+1,:]=ng.weight
    #   # ngxinp[reg+1,:]=ng.npairs
    #   # rgxi[reg+1,:]=rg.xi
    #   # rgxim[reg+1,:]=rg.xi_im
    #   # rgxiwt[reg+1,:]=rg.weight
    #   # rgxinp[reg+1,:]=rg.npairs
    #   # ggxip[reg+1,:]=gg.xip
    #   # ggxim[reg+1,:]=gg.xim
    #   # ggxiwt[reg+1,:]=gg.weight
    #   # ggxinp[reg+1,:]=gg.npairs
    #   # rrxiwt[reg+1,:]=rr.weight

    # np.save('cv_r_60.npy',r)
    # np.save('cv_ng_xi_60.npy',ngxi)
    # np.save('cv_ng_xim_60.npy',ngxim)
    # np.save('cv_ng_xi_wt_60.npy',ngxiwt)
    # np.save('cv_ng_xi_np_60.npy',ngxinp)
    # np.save('cv_rg_xi_60.npy',rgxi)
    # np.save('cv_rg_xim_60.npy',rgxim)
    # np.save('cv_rg_xi_wt_60.npy',rgxiwt)
    # np.save('cv_rg_xi_np_60.npy',rgxinp)
    # np.save('cv_nk_xi_60.npy',nkxi)
    # np.save('cv_nk_xi_wt_60.npy',nkxiwt)
    # np.save('cv_nk_xi_np_60.npy',nkxinp)
    # np.save('cv_rk_xi_60.npy',rkxi)
    # np.save('cv_rk_xi_wt_60.npy',rkxiwt)
    # np.save('cv_rk_xi_np_60.npy',rkxinp)
    # np.save('cv_gg_xip_60.npy',ggxip)
    # np.save('cv_gg_xim_60.npy',ggxim)
    # np.save('cv_gg_xi_wt_60.npy',ggxiwt)
    # np.save('cv_gg_xi_np_60.npy',ggxinp)
    # np.save('cv_kk_xi_60.npy',kkxi)
    # np.save('cv_kk_xi_wt_60.npy',kkxiwt)
    # np.save('cv_kk_xi_np_60.npy',kkxinp)
    # np.save('cv_rr_xi_wt_60.npy',rrxiwt)

elif int(sys.argv[1])==31:

    field.field.loop_epoch_stuff(int(sys.argv[2]),maxloop=1)

elif int(sys.argv[1])==32:

    p=None

    cols=['stamp','nexp','chi2pix','psffwhm','coadd','info','error','like','rgp','dec','evals','rad','dec_off','ra_off','fluxfrac','psf1','psf2','hsmpsf1','hsmpsf2','modmax','modmin','ra','resmax','tile','maskfrac','snr','resmin','e1','e2','iter','bflux','dflux','flux','cov11','cov22','zp','g','r','i','z','bfrac']
    i30=catalog.CatalogStore('y1_i3_even_v1',cattype='i3',cols=cols,catdir='/project/projectdirs/des/wl/desdata/wlpipe/im3shape_y1a1_v2/even/main/',release='y1',p=p)
    catalog.CatalogMethods.remove_duplicates(i30)

    print 'done with catalog stuff'

    lin.summary_stats.i3_flags_vals_check(i30)

    print 'done with error stuff'

    mask=(i30.info==0)&(i30.rgp>1.13)&(i30.snr>12)&(i30.snr<200)&(i30.rgp<3)&(~(np.isnan(i30.psf1)|np.isnan(i30.psf2)|np.isnan(i30.snr)|np.isnan(i30.psffwhm)))&(i30.g<99)&(i30.r<99)&(i30.i<99)&(i30.z<99)
    catalog.CatalogMethods.match_cat(i30.mask)

    # nbc=np.load('/home/troxel/destest/i3nbcv1.npy')
    # a=np.argsort(nbc[:,0])
    # mask=np.diff(nbc[a,0])
    # mask=mask==0
    # mask=~mask
    # mask=a[mask]
    # nbc=nbc[mask]

    # x,y=catalog.CatalogMethods.sort2(nbc[:,0],i3.coadd)
    # catalog.CatalogMethods.match_cat(i3,y)
    # i3.m1=nbc[x,1]
    # i3.m2=nbc[x,1]
    # i3.c1=nbc[x,2]
    # i3.c2=nbc[x,3]
    # i3.w=nbc[x,4]
    # i3.w/=np.sum(i3.w)

    i30.wt=False
    i30.bs=False

    # Post-cut summary of error properties
    lin.summary_stats.i3_flags_vals_check(i30,label='b')

    # Optionally specify sub-set of columns for the following functions
    cols=['chi2pix','psffwhm','nlike','rgp','dec','evals','rad','dec_off','ra_off','invfluxfrac','psf1','psf2','hsmpsf1','hsmpsf2','modmax','modmin','ra','resmax','maskfrac','snr','resmin','e1','e2','flux','cov11','cov22','zp','g','r','i','z','gr','ri','iz','pos','e','psfe','dpsf','psfpos','hsmpsfe','hsmdpsf','hsmpsfpos']

    print 'done with vals check stuff'

    lin.summary_stats.val_stats(i30,cols=cols)

    print 'done with vals stuff'
    lin.hist.hist_tests(i30,cols=cols,p=p)

    print 'done with hist stuff'
    lin.hist.hist_2D_tests(i30,colsx=cols,colsy=cols,p=p)

    print 'done with footprint stuff'
    sys_split.split.cat_splits_lin_e(i30,cols=cols,p=p)
    sys_split.split.cat_splits_lin_full(i30,cols=cols,p=p)

elif int(sys.argv[1])==33:

    p=None

    cols=['stamp','nexp','chi2pix','psffwhm','coadd','info','error','like','rgp','dec','evals','rad','dec_off','ra_off','fluxfrac','psf1','psf2','hsmpsf1','hsmpsf2','modmax','modmin','ra','resmax','tile','maskfrac','snr','resmin','e1','e2','iter','bflux','dflux','flux','cov11','cov22','zp','g','r','i','z','bfrac']
    i31=catalog.CatalogStore('y1_i3_odd_v1',cattype='i3',cols=cols,catdir='/project/projectdirs/des/wl/desdata/wlpipe/im3shape_y1a1_v2/odd/main/',release='y1',p=p,maxiter=1)
    catalog.CatalogMethods.remove_duplicates(i31)

    print 'done with catalog stuff'

    lin.summary_stats.i3_flags_vals_check(i31)

    print 'done with error stuff'

    mask=(i31.info==0)&(i31.rgp>1.13)&(i31.snr>12)&(i31.snr<200)&(i31.rgp<3)&(~(np.isnan(i31.psf1)|np.isnan(i31.psf2)|np.isnan(i31.snr)|np.isnan(i31.psffwhm)))&(i31.g<99)&(i31.r<99)&(i31.i<99)&(i31.z<99)
    catalog.CatalogMethods.match_cat(i31,mask)

    # nbc=np.load('/home/troxel/destest/i3nbcv1.npy')
    # a=np.argsort(nbc[:,0])
    # mask=np.diff(nbc[a,0])
    # mask=mask==0
    # mask=~mask
    # mask=a[mask]
    # nbc=nbc[mask]

    # x,y=catalog.CatalogMethods.sort2(nbc[:,0],i3.coadd)
    # catalog.CatalogMethods.match_cat(i3,y)
    # i3.m1=nbc[x,1]
    # i3.m2=nbc[x,1]
    # i3.c1=nbc[x,2]
    # i3.c2=nbc[x,3]
    # i3.w=nbc[x,4]
    # i3.w/=np.sum(i3.w)

    i31.wt=False
    i31.bs=False

    # Post-cut summary of error properties
    lin.summary_stats.i3_flags_vals_check(i31,label='b')

    # Optionally specify sub-set of columns for the following functions
    cols=['chi2pix','psffwhm','nlike','rgp','dec','evals','rad','dec_off','ra_off','invfluxfrac','psf1','psf2','hsmpsf1','hsmpsf2','modmax','modmin','ra','resmax','maskfrac','snr','resmin','e1','e2','flux','cov11','cov22','zp','g','r','i','z','gr','ri','iz','pos','e','psfe','dpsf','psfpos','hsmpsfe','hsmdpsf','hsmpsfpos']

    print 'done with vals check stuff'

    lin.summary_stats.val_stats(i31,cols=cols)

    print 'done with vals stuff'
    lin.hist.hist_tests(i31,cols=cols,p=p)

    lin.hist.hist_2D_tests(i31,colsx=cols,colsy=cols,p=p)

    print 'done with footprint stuff'
    sys_split.split.cat_splits_lin_e(i31,cols=cols,p=p)
    sys_split.split.cat_splits_lin_full(i31,cols=cols,p=p)


elif int(sys.argv[1])==34:

    # cd /global/cscratch1/sd/troxel/destest
    # source /project/projectdirs/cmb/modules/hpcports_NERSC.sh
    # source /scratch2/scratchdirs/zuntz/stack/setup

    config.tests=['all']

    p=None
    cols=['box_size','coadd_run','flags','fofid','id','mask_frac','mcal_flags','mcal_g','mcal_g_1m','mcal_g_1m_psf','mcal_g_1p','mcal_g_1p_psf','mcal_g_2m','mcal_g_2m_psf','mcal_g_2p','mcal_g_2p_psf','mcal_gpsf','mcal_pars','mcal_pars_1m','mcal_pars_1m_psf','mcal_pars_1p','mcal_pars_1p_psf','mcal_pars_2m','mcal_pars_2m_psf','mcal_pars_2p','mcal_pars_2p_psf','mcal_s2n_r','mcal_s2n_r_1m','mcal_s2n_r_1m_psf','mcal_s2n_r_1p','mcal_s2n_r_1p_psf','mcal_s2n_r_2m','mcal_s2n_r_2m_psf','mcal_s2n_r_2p','mcal_s2n_r_2p_psf','mcal_T','mcal_T_1m','mcal_T_1m_psf','mcal_T_1p','mcal_T_1p_psf','mcal_T_2m','mcal_T_2m_psf','mcal_T_2p','mcal_T_2p_psf','mcal_T_err','mcal_T_err_1m','mcal_T_err_1m_psf','mcal_T_err_1p','mcal_T_err_1p_psf','mcal_T_err_2m','mcal_T_err_2m_psf','mcal_T_err_2p','mcal_T_err_2p_psf','mcal_T_r','mcal_T_r_1m','mcal_T_r_1m_psf','mcal_T_r_1p','mcal_T_r_1p_psf','mcal_T_r_2m','mcal_T_r_2m_psf','mcal_T_r_2p','mcal_T_r_2p_psf','mcal_Tpsf','nimage_tot','nimage_use','number','obj_flags','psf_flags','psf_flux','psf_flux_err','psf_flux_s2n','psf_mag','psfrec_g','psfrec_T','time_last_fit','gauss_max_logsb','gauss_logsb']
    ng=catalog.CatalogStore('ngmix_metacal_v1',cattype='ng',cols=cols,catdir='/global/cscratch1/sd/troxel/ngmix/',release='y1',p=p,cutfunc=catalog.CatalogMethods.final_null_cuts_id(),hdu=1,maxrows=1000000)
    catalog.CatalogMethods.remove_duplicates(ng)

    tmp=fio.FITS('/home/zuntz/gold/y1a1_ra_dec_flags_mag_v2.fits')[-1].read()
    x,y=catalog.CatalogMethods.sort2(tmp['COADD_OBJECTS_ID'],ng.coadd)

    print 'done with catalog stuff'

    lin.summary_stats.i3_flags_vals_check(ng,flags=['flags'])

    print 'done with error stuff'

    catalog.CatalogMethods.match_cat(ng,y)
    ng.g=tmp['MAG_AUTO_G'][x]
    ng.r=tmp['MAG_AUTO_R'][x]
    ng.i=tmp['MAG_AUTO_I'][x]
    ng.z=tmp['MAG_AUTO_Z'][x]
    ng.ri=ng.r-ng.i
    ng.gr=ng.g-ng.r
    ng.ra=tmp['RA'][x]
    ng.dec=tmp['DEC'][x]
    ng.zp=tmp['DESDM_ZP'][x]

    import healpy as hp
    goldmask=hp.read_map(config.golddir+'y1a1_gold_1.0.2_wide_footprint_4096.fit.gz')
    badregmask=hp.read_map(config.golddir+'y1a1_gold_1.0.2_wide_badmask_4096.fit.gz')

    pix=hp.ang2pix(4096, np.pi/2.-np.radians(ng.dec),np.radians(ng.ra), nest=False)

    mask=(ng.flags==0)&(tmp['MODEST'][x]==1)&(tmp['FLAGS_R'][x]==0)&((goldmask[pix] >=1)==True)&(badregmask[pix]==0)&(ng.mcal_s2n_r>10)&(ng.g<50)&(ng.r<50)&(ng.i<50)&(ng.z<50)
    catalog.CatalogMethods.match_cat(ng,mask)

    ng.wt=False
    ng.bs=False

    # Post-cut summary of error properties
    lin.summary_stats.i3_flags_vals_check(ng,flags=['flags'],label='b')

    print 'done with vals check stuff'

    lin.summary_stats.val_stats(ng)

    print 'done with vals stuff'

    cols=['box_size','coadd_run','flags','fofid','coadd','mask_frac','mcal_flags','mcal_g_1','mcal_g_1m_1','mcal_g_1m_psf_1','mcal_g_1p_1','mcal_g_1p_psf_1','mcal_g_2m_1','mcal_g_2m_psf_1','mcal_g_2p_1','mcal_g_2p_psf_1','mcal_gpsf_1','mcal_g_2','mcal_g_1m_2','mcal_g_1m_psf_2','mcal_g_1p_2','mcal_g_1p_psf_2','mcal_g_2m_2','mcal_g_2m_psf_2','mcal_g_2p_2','mcal_g_2p_psf_2','mcal_gpsf_2','mcal_pars_1','mcal_pars_2','mcal_pars_1m_1','mcal_pars_1m_psf_1','mcal_pars_1p_1','mcal_pars_1p_psf_1','mcal_pars_2m_1','mcal_pars_2m_psf_1','mcal_pars_2p_1','mcal_pars_2p_psf_1','mcal_pars_1m_2','mcal_pars_1m_psf_2','mcal_pars_1p_2','mcal_pars_1p_psf_2','mcal_pars_2m_2','mcal_pars_2m_psf_2','mcal_pars_2p_2','mcal_pars_2p_psf_2','mcal_pars_1m_3','mcal_pars_1m_psf_3','mcal_pars_1p_3','mcal_pars_1p_psf_3','mcal_pars_2m_3','mcal_pars_2m_psf_3','mcal_pars_2p_3','mcal_pars_2p_psf_3','mcal_pars_1m_4','mcal_pars_1m_psf_4','mcal_pars_1p_4','mcal_pars_1p_psf_4','mcal_pars_2m_4','mcal_pars_2m_psf_4','mcal_pars_2p_4','mcal_pars_2p_psf_4','mcal_pars_1m_5','mcal_pars_1m_psf_5','mcal_pars_1p_5','mcal_pars_1p_psf_5','mcal_pars_2m_5','mcal_pars_2m_psf_5','mcal_pars_2p_5','mcal_pars_2p_psf_5','mcal_pars_1m_6','mcal_pars_1m_psf_6','mcal_pars_1p_6','mcal_pars_1p_psf_6','mcal_pars_2m_6','mcal_pars_2m_psf_6','mcal_pars_2p_6','mcal_pars_2p_psf_6','mcal_s2n_r','mcal_s2n_r_1m','mcal_s2n_r_1m_psf','mcal_s2n_r_1p','mcal_s2n_r_1p_psf','mcal_s2n_r_2m','mcal_s2n_r_2m_psf','mcal_s2n_r_2p','mcal_s2n_r_2p_psf','mcal_T','mcal_T_1m','mcal_T_1m_psf','mcal_T_1p','mcal_T_1p_psf','mcal_T_2m','mcal_T_2m_psf','mcal_T_2p','mcal_T_2p_psf','mcal_T_err','mcal_T_err_1m','mcal_T_err_1m_psf','mcal_T_err_1p','mcal_T_err_1p_psf','mcal_T_err_2m','mcal_T_err_2m_psf','mcal_T_err_2p','mcal_T_err_2p_psf','mcal_T_r','mcal_T_r_1m','mcal_T_r_1m_psf','mcal_T_r_1p','mcal_T_r_1p_psf','mcal_T_r_2m','mcal_T_r_2m_psf','mcal_T_r_2p','mcal_T_r_2p_psf','mcal_Tpsf','nimage_tot','nimage_use','number','obj_flags','psf_flags','psf_flux','psf_flux_err','psf_flux_s2n','psf_mag','psfrec_g_1','psfrec_g_2','psfrec_T','time_last_fit']

    lin.hist.hist_tests(ng,p=p)

    #lin.hist.hist_2D_tests(ng,p=p)

    print 'done with footprint stuff'
    sys_split.split.cat_splits_lin_e(ng,p=p)
    #sys_split.split.cat_splits_lin_full(ng,p=p)

    rm=catalog.CatalogStore('y1_rm_highdens',cattype='gal',cols=-1,catfile=config.redmagicdirnersc+'y1a1_gold_1.0.2b-full_redmapper_v6.4.11_redmagic_highdens_0.5-10.fit',release='y1')

    cols=['mask_frac','mcal_s2n_r','mcal_T_r','nimage_tot','nimage_use','number','psf_flux','psf_flux_s2n','psf_mag','psf1','psf2','psfrec_T','time_last_fit','pos','ra','dec','airmass','exptime','maglimit','skysigma','skybrite','fwhm','psfpos','r','ri']

    sys_split.split.cat_splits_2pt(ng,rm,cols=cols)

elif int(sys.argv[1])==35:

    def chi(z,omegam=0.27,H=100):
      # import cosmology  
      # c0=cosmology.Cosmo(H0=H,omega_m=omegam)
      # return c0.Dc(0.,z)
      from astropy import cosmology
      from astropy.cosmology import FlatLambdaCDM
      cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
      return cosmo.comoving_distance(z).value    

    rmd=catalog.CatalogStore('y1_rm_highdens',cattype='gal',cols=['coadd','ra','dec','zp','e1','e2','c1','c2','m','w'],catfile=config.redmagicdirnersc+'y1a1_gold_1.0.2b-full_redmapper_v6.4.11_redmagic_highdens_0.5-10_e.fit',release='y1',ranfile=config.redmagicdirnersc+'y1a1_gold_1.0.2b-full_redmapper_v6.4.11_redmagic_highdens_0.5-10_randoms.fit')
    rmd.r=chi(rmd.zp)
    rmd.ran_r=np.load(config.redmagicdirnersc+'highdens_ran_r.npy')

    import treecorr

    maskd=rmd.e1!=-9999

    catdd=treecorr.Catalog(ra=rmd.ra, dec=rmd.dec, r=rmd.r, ra_units='deg', dec_units='deg')
    cated=treecorr.Catalog(g1=rmd.e1[maskd]-rmd.c1[maskd], g2=rmd.e2[maskd]-rmd.c2[maskd], w=rmd.w[maskd], ra=rmd.ra[maskd], dec=rmd.dec[maskd], r=rmd.r[maskd], ra_units='deg', dec_units='deg')
    catmd=treecorr.Catalog(k=(1.+rmd.m[maskd]), w=rmd.w[maskd], ra=rmd.ra[maskd], dec=rmd.dec[maskd], r=rmd.r[maskd], ra_units='deg', dec_units='deg')
    catrd=treecorr.Catalog(ra=rmd.ran_ra, dec=rmd.ran_dec, r=rmd.ran_r, ra_units='deg', dec_units='deg')

    de = treecorr.NGCorrelation(nbins=10, min_sep=.5, max_sep=100, min_rpar = -50, max_rpar = 50, bin_slop=0.01, verbose=0)
    dm = treecorr.NKCorrelation(nbins=10, min_sep=.5, max_sep=100, min_rpar = -50, max_rpar = 50, bin_slop=0.01, verbose=0)
    re = treecorr.NGCorrelation(nbins=10, min_sep=.5, max_sep=100, min_rpar = -50, max_rpar = 50, bin_slop=0.01, verbose=0)
    rm = treecorr.NKCorrelation(nbins=10, min_sep=.5, max_sep=100, min_rpar = -50, max_rpar = 50, bin_slop=0.01, verbose=0)

    de.process(catdd,cated,metric='Rperp')
    re.process(catrd,cated,metric='Rperp')
    dm.process(catdd,catmd,metric='Rperp')
    rm.process(catrd,catmd,metric='Rperp')

    xip=de.xi/dm.xi-re.xi/rm.xi
    xim=de.xi_im/dm.xi-re.xi_im/rm.xi
    varxi=np.sqrt(de.varxi)
    print np.exp(de.meanlogr),xip,varxi
    print np.exp(de.meanlogr),xim,varxi

    fig.plot_methods.plot_IA(np.exp(de.meanlogr),[xip,xim],[varxi,varxi],'dd_phot')

elif int(sys.argv[1])==36:

    def chi(z,omegam=0.27,H=100):
      # import cosmology  
      # c0=cosmology.Cosmo(H0=H,omega_m=omegam)
      # return c0.Dc(0.,z)
      from astropy import cosmology
      from astropy.cosmology import FlatLambdaCDM
      cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
      return cosmo.comoving_distance(z).value    

    rml=catalog.CatalogStore('y1_rm_highdens',cattype='gal',cols=['coadd','ra','dec','zp','e1','e2','c1','c2','m','w'],catfile=config.redmagicdirnersc+'y1a1_gold_1.0.2b-full_redmapper_v6.4.11_redmagic_highlum_1.0-04_e.fit',release='y1',ranfile=config.redmagicdirnersc+'y1a1_gold_1.0.2b-full_redmapper_v6.4.11_redmagic_highlum_1.0-04_randoms.fit')
    rml.r=chi(rml.zp)
    rml.ran_r=np.load(config.redmagicdirnersc+'highlum_ran_r.npy')

    import treecorr

    maskl=rml.e1!=-9999

    catdl=treecorr.Catalog(ra=rml.ra, dec=rml.dec, r=rml.r, ra_units='deg', dec_units='deg')
    catel=treecorr.Catalog(g1=rml.e1[maskl]-rml.c1[maskl], g2=rml.e2[maskl]-rml.c2[maskl], w=rml.w[maskl], ra=rml.ra[maskl], dec=rml.dec[maskl], r=rml.r[maskl], ra_units='deg', dec_units='deg')
    catml=treecorr.Catalog(k=(1.+rml.m[maskl]), w=rml.w[maskl], ra=rml.ra[maskl], dec=rml.dec[maskl], r=rml.r[maskl], ra_units='deg', dec_units='deg')
    catrl=treecorr.Catalog(ra=rml.ran_ra, dec=rml.ran_dec, r=rml.ran_r, ra_units='deg', dec_units='deg')

    de = treecorr.NGCorrelation(nbins=10, min_sep=.5, max_sep=100, min_rpar = -50, max_rpar = 50, bin_slop=0.01, verbose=0)
    dm = treecorr.NKCorrelation(nbins=10, min_sep=.5, max_sep=100, min_rpar = -50, max_rpar = 50, bin_slop=0.01, verbose=0)
    re = treecorr.NGCorrelation(nbins=10, min_sep=.5, max_sep=100, min_rpar = -50, max_rpar = 50, bin_slop=0.01, verbose=0)
    rm = treecorr.NKCorrelation(nbins=10, min_sep=.5, max_sep=100, min_rpar = -50, max_rpar = 50, bin_slop=0.01, verbose=0)

    de.process(catdl,catel,metric='Rperp')
    re.process(catrl,catel,metric='Rperp')
    dm.process(catdl,catml,metric='Rperp')
    rm.process(catrl,catml,metric='Rperp')

    xip=de.xi/dm.xi-re.xi/rm.xi
    xim=de.xi_im/dm.xi-re.xi_im/rm.xi
    varxi=np.sqrt(de.varxi)
    print np.exp(de.meanlogr),xip,varxi
    print np.exp(de.meanlogr),xim,varxi

    fig.plot_methods.plot_IA(np.exp(de.meanlogr),[xip,xim],[varxi,varxi],'ll_phot')

elif int(sys.argv[1])==37:

    import treecorr as treecorr

    rml=catalog.CatalogStore('y1_rm_highdens',cattype='gal',cols=['coadd','ra','dec','zp'],catfile=config.redmagicdirnersc+'y1a1_gold_1.0.2b-full_redmapper_v6.4.11_redmagic_highlum_1.0-04.fit',release='y1',ranfile=config.redmagicdirnersc+'y1a1_gold_1.0.2b-full_redmapper_v6.4.11_redmagic_highlum_1.0-04_randoms.fit')

    def chi(z,omegam=0.27,H=100):
      from astropy import cosmology
      from astropy.cosmology import FlatLambdaCDM
      cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
      return cosmo.comoving_distance(z).value

    rml.r=chi(rml.zp)
    rml.ran_r=np.load(config.redmagicdirnersc+'highlum_ran_r.npy')

    i3=catalog.CatalogStore('y1_i3_sv_v1',cattype='i3',cols=['coadd','ra','dec','psf1','psf2','psffwhm','rgp','snr','info'],catdir='/share/c12/y1/im3shape/single_band/r/y1v1/complete/main/',release='y1',tiles=np.load('s82_tiles_b.npy'),cutfunc=None)
    mask=(i3.info==0)&(i3.rgp>1.13)&(i3.snr>12)&(i3.snr<200)&(i3.rgp<3)&(~(np.isnan(i3.psf1)|np.isnan(i3.psf2)|np.isnan(i3.snr)|np.isnan(i3.psffwhm)))
    catalog.CatalogMethods.match_cat(i3,mask)
    i3ran=fio.FITS('b_random.fits')[-1].read()

    catxa=treecorr.Catalog(ra=i3.ra, dec=i3.dec, ra_units='deg', dec_units='deg')
    catra=treecorr.Catalog(ra=i3ran['alphawin_j2000_i'], dec=i3ran['deltawin_j2000_i'], ra_units='deg', dec_units='deg')
    nn = treecorr.NNCorrelation(nbins=20, min_sep=.1, max_sep=60., sep_units='arcmin',bin_slop=.01,verbose=0)
    nr = treecorr.NNCorrelation(nbins=20, min_sep=.1, max_sep=60., sep_units='arcmin',bin_slop=.01,verbose=0)
    rr = treecorr.NNCorrelation(nbins=20, min_sep=.1, max_sep=60., sep_units='arcmin',bin_slop=.01,verbose=0)
    nn.process(catxa)
    nr.process(catra,catxa)
    rr.process(catra)
    xip,xiperr=nn.calculateXi(rr,dr=nr)
    xiperr=np.sqrt(xiperr)
    theta=np.exp(nn.meanlogr)
    np.save('i3_wt.npy',np.vstack((theta,xip,xiperr)))

    zp=np.linspace(.1,.95,9)
    for i in range(len(zp)-1):
      mask=(rml.zp>zp[i])&(rml.zp<=zp[i+1])
      maskr=(rml.ran_zp>zp[i])&(rml.ran_zp<=zp[i+1])
      catxa=treecorr.Catalog(ra=rml.ra[mask], dec=rml.dec[mask], r=rml.r[mask], ra_units='deg', dec_units='deg')
      catra=treecorr.Catalog(ra=rml.ran_ra[maskr], dec=rml.ran_dec[maskr], r=rml.ran_r[maskr], ra_units='deg', dec_units='deg')
      nn = treecorr.NNCorrelation(nbins=20, min_sep=.1, max_sep=10.,min_rpar=-.5,max_rpar=.5,bin_slop=.01,verbose=0)
      nr = treecorr.NNCorrelation(nbins=20, min_sep=.1, max_sep=10.,min_rpar=-.5,max_rpar=.5,bin_slop=.01,verbose=0)
      rr = treecorr.NNCorrelation(nbins=20, min_sep=.1, max_sep=10.,min_rpar=-.5,max_rpar=.5,bin_slop=.01,verbose=0)
      nn.process(catxa,metric='Rperp')
      nr.process(catra,catxa,metric='Rperp')
      rr.process(catra,metric='Rperp')
      xip,xiperr=nn.calculateXi(rr,dr=nr)
      xiperr=np.sqrt(xiperr)
      theta=np.exp(nn.meanlogr)
      np.save('rml_wp_'+str(i)+'.npy',np.vstack((theta,xip,xiperr)))

      catxap=treecorr.Catalog(ra=i3.ra, dec=i3.dec, ra_units='deg', dec_units='deg')
      catrap=treecorr.Catalog(ra=i3ran['alphawin_j2000_i'], dec=i3ran['deltawin_j2000_i'], ra_units='deg', dec_units='deg')
      nn = treecorr.NNCorrelation(nbins=20, min_sep=.01, max_sep=100., sep_units='arcmin',bin_slop=.01,verbose=0)
      nr = treecorr.NNCorrelation(nbins=20, min_sep=.01, max_sep=100., sep_units='arcmin',bin_slop=.01,verbose=0)
      rn = treecorr.NNCorrelation(nbins=20, min_sep=.01, max_sep=100., sep_units='arcmin',bin_slop=.01,verbose=0)
      rr = treecorr.NNCorrelation(nbins=20, min_sep=.01, max_sep=100., sep_units='arcmin',bin_slop=.01,verbose=0)
      nn.process(catxa,catxap)
      nr.process(catra,catxap)
      rn.process(catxa,catrap)
      rr.process(catra,catrap)
      xip,xiperr=nn.calculateXi(rr,dr=nr,rd=rn)
      xiperr=np.sqrt(xiperr)
      theta=np.exp(nn.meanlogr)
      np.save('rml_i3_wt_'+str(i)+'.npy',np.vstack((theta,xip,xiperr)))

  sys.exit()

# cd /global/cscratch1/sd/troxel/destest
# source /project/projectdirs/cmb/modules/hpcports_NERSC.sh
# source /scratch2/scratchdirs/zuntz/stack/setup

tmp=fio.FITS(config.redmagicdirnersc+'y1a1_gold_1.0.2b-full_redmapper_v6.4.11_redmagic_highlum_1.0-04.fit')[-1].read()
store=np.ones(tmp.shape, dtype=tmp.dtype.descr + [('e1','f8')]+[('e2','f8')]+[('m','f8')]+[('c1','f8')]+[('c2','f8')]+[('weight','f8')])
for name in tmp.dtype.names:
  store[name]=tmp[name]

store['e1']=-9999*store['e1']
store['e2']=-9999*store['e2']
store['c1']=-9999*store['c1']
store['c2']=-9999*store['c2']
store['m']=-9999*store['m']
store['weight']=-9999*store['weight']

spec=fio.FITS('/global/cscratch1/sd/troxel/spec_cat_0.fits.gz')[-1].read()
x,y=catalog.CatalogMethods.sort2(store['COADD_OBJECTS_ID'],spec['coadd_objects_id'])
store['ZSPEC'][x]=spec['z_spec'][y]

for ifile,file in enumerate(glob.glob('/project/projectdirs/des/wl/desdata/wlpipe/im3shape_y1a1_v1/nbc/main/*')):
  print ifile,file
  tmp2=fio.FITS(file)[-1].read(columns=['coadd_objects_id','e1','e2','mean_psf_e1_sky','mean_psf_e2_sky','mean_psf_fwhm','mean_rgpp_rp','snr','m','c1','c2','weight','info_flag'])
  mask=(tmp2['info_flag']==0)&(tmp2['mean_rgpp_rp']>1.13)&(tmp2['snr']>12)&(tmp2['snr']<200)&(tmp2['mean_rgpp_rp']<3)&(~(np.isnan(tmp2['mean_psf_e1_sky'])|np.isnan(tmp2['mean_psf_e2_sky'])|np.isnan(tmp2['snr'])|np.isnan(tmp2['mean_psf_fwhm'])))
  tmp2=tmp2[mask]
  x,y=catalog.CatalogMethods.sort2(store['COADD_OBJECTS_ID'],tmp2['coadd_objects_id'])
  store['e1'][x]=tmp2['e1'][y]
  store['e2'][x]=tmp2['e2'][y]
  store['c1'][x]=tmp2['c1'][y]
  store['c2'][x]=tmp2['c2'][y]
  store['m'][x]=tmp2['m'][y]
  store['weight'][x]=tmp2['weight'][y]

fio.write(config.redmagicdirnersc+'y1a1_gold_1.0.2b-full_redmapper_v6.4.11_redmagic_highlum_1.0-04_e.fit',store,clobber=True)

spec=fio.FITS('/home/troxel/spec_cat_0.fits.gz')[-1].read()
spec2=fio.FITS('/home/troxel/Y1SPEC_MATCH2as_Y1Goldv103.fits')[-1].read()
tmp=fio.FITS(config.redmagicdir+'y1a1_gold_1.0.2b-full_redmapper_v6.4.11_redmagic_highdens_0.5-10.fit')[-1].read()
tmp=tmp[tmp['ZSPEC']>0]
x,y=catalog.CatalogMethods.sort2(tmp['COADD_OBJECTS_ID'],spec['coadd_objects_id'])
x1,y1=catalog.CatalogMethods.sort2(tmp['COADD_OBJECTS_ID'],spec2['COADD_OBJECTS_ID'])
x2,y2=catalog.CatalogMethods.sort2(tmp['COADD_OBJECTS_ID'],spec2['COADD_OBJECTS_ID'][spec2['SOURCE']!='PRIMUS'])
x3,y3=catalog.CatalogMethods.sort2(spec['coadd_objects_id'],spec2['COADD_OBJECTS_ID'][spec2['SOURCE']!='PRIMUS'])

tmp0=np.copy(tmp['ZSPEC'])
tmp0[x]=spec['z_spec'][y]
tmp1=np.copy(tmp['ZSPEC'])
tmp1[x1]=spec2['SPEC_Z'][y1]
tmp2=np.copy(tmp['ZSPEC'])
tmp2[x2]=spec2['SPEC_Z'][y2]

np.sum(tmp['ZSPEC']>0)
len(x)
len(x1)
len(x2)
np.sum(tmp0>0)
np.sum(tmp1>0)
np.sum(tmp2>0)

np.mean(np.abs(tmp['ZSPEC'][x2][tmp['ZSPEC'][x2]>0]-spec2['SPEC_Z'][y2][tmp['ZSPEC'][x2]>0]))
np.mean(np.abs(spec['z_spec'][x3]-spec2['SPEC_Z'][y3]))

i3=catalog.CatalogStore('y1_i3_sv_v1',cattype='i3',cols=['coadd','e1','e2','ra','dec','psf1','psf2','psffwhm','rgp','snr','info','ra_off','dec_off'],catdir='/project/projectdirs/des/wl/desdata/wlpipe/im3shape_y1a1_v1/nbc/main/',release='y1',tiles=np.load('s82_tiles_i3.npy'),cutfunc=None)
mask=(i3.info==0)&(i3.rgp>1.13)&(i3.snr>12)&(i3.snr<200)&(i3.rgp<3)&(~(np.isnan(i3.psf1)|np.isnan(i3.psf2)|np.isnan(i3.snr)|np.isnan(i3.psffwhm)))


import glob
import healpy as hp
gal=np.array([])
wl=np.array([])
ra=np.array([])
dec=np.array([])
for ifile,file in enumerate(glob.glob('/project/projectdirs/des/wl/desdata/wlpipe/im3shape_y1a1_v1/nbc/main/*')):
  print ifile,file
  tmp2=fio.FITS(file)[-1].read(columns=['coadd_objects_id','ra','dec','mean_psf_e1_sky','mean_psf_e2_sky','mean_psf_fwhm','mean_rgpp_rp','snr','info_flag'])
  if tmp2['dec'][0]<-10:
    continue
  mask=(tmp2['info_flag']==0)&(tmp2['mean_rgpp_rp']>1.13)&(tmp2['snr']>12)&(tmp2['snr']<200)&(tmp2['mean_rgpp_rp']<3)&(~(np.isnan(tmp2['mean_psf_e1_sky'])|np.isnan(tmp2['mean_psf_e2_sky'])|np.isnan(tmp2['snr'])|np.isnan(tmp2['mean_psf_fwhm'])))
  pix=hp.ang2pix(4096*2, np.pi/2.-np.radians(tmp2['dec']),np.radians(tmp2['ra']), nest=True)
  gal=np.append(gal,pix)
  wl=np.append(wl,pix[mask])

b=fio.FITS('b1.fits')[-1].read()
tmp=fio.FITS('b2.fits')[-1].read()
b=np.append(b,tmp)
tmp=fio.FITS('b3.fits')[-1].read()
b=np.append(b,tmp)
tmp=fio.FITS('b4.fits')[-1].read()
b=np.append(b,tmp)
tmp=fio.FITS('b5.fits')[-1].read()
b=np.append(b,tmp)
tmp=fio.FITS('b6.fits')[-1].read()
b=np.append(b,tmp)
tmp=fio.FITS('b7.fits')[-1].read()
b=np.append(b,tmp)

gdmask=hp.read_map('y1a1_gold_1.0.2_wide_footprint_4096.fit.gz')
badmask=hp.read_map('y1a1_gold_1.0.2_wide_badmask_4096.fit.gz')
pix=hp.ang2pix(4096, np.pi/2.-np.radians(b['deltawin_j2000_i']),np.radians(b['alphawin_j2000_i']), nest=False)
gold_mask=(gdmask[pix] >=1)
gold_flag=badmask[pix]
mask=(gold_mask==True)&(gold_flag==0)
b=b[mask]

gal=np.load('gal.npy')
wl=np.load('wl.npy')

pixwl,cntwl=np.unique(wl,return_counts=True)
pixgal,cntgal=np.unique(gal,return_counts=True)

bpix=hp.ang2pix(4096, np.pi/2.-np.radians(b['deltawin_j2000_i']),np.radians(b['alphawin_j2000_i']), nest=True)
pixb,cntb=np.unique(bpix,return_counts=True)

x,y=catalog.CatalogMethods.sort2(pixb,pixwl)
b=b[np.in1d(bpix,pixb[x],assume_unique=False)]
pixwl=pixwl[y]
cntwl=cntwl[y]

bpix=hp.ang2pix(4096, np.pi/2.-np.radians(b['deltawin_j2000_i']),np.radians(b['alphawin_j2000_i']), nest=True)
sort=np.argsort(bpix)
b=b[sort]

bpix=hp.ang2pix(4096, np.pi/2.-np.radians(b['deltawin_j2000_i']),np.radians(b['alphawin_j2000_i']), nest=True)

galmask=np.in1d(pixgal,pixwl,assume_unique=True)
pixgal=pixgal[galmask]
cntgal=cntgal[galmask]

x,y=catalog.CatalogMethods.sort2(pixgal,pixwl)
pixgal=pixgal[x]
cntgal=cntgal[x]
pixwl=pixwl[y]
cntwl=cntwl[y]

ratio=1.*cntwl/cntgal

ratio/=.8
ratio[ratio>1]=1

import numpy.random as rand
diff=np.diff(bpix)
diffmask=np.append(np.array([0]),np.where(diff!=0)[0])
mask=np.zeros(len(bpix))
for i in range(len(diffmask)-1):
  num=int(round((diffmask[i+1]-diffmask[i])*ratio[i]))
  ind=rand.random_integers(diffmask[i],diffmask[i+1],num)
  mask[ind]=1.

b['alphawin_j2000_i'][b['alphawin_j2000_i']<-180]=b['alphawin_j2000_i'][b['alphawin_j2000_i']<-180]+360
mask=mask.astype(bool)
plt.hist2d(b['alphawin_j2000_i'][mask],b['deltawin_j2000_i'][mask],bins=500)
plt.savefig('tmp.png')
plt.close()

b=b[mask]
fio.write('b_random.fits',b)


import treecorr as treecorr

rml=catalog.CatalogStore('y1_rm_highdens',cattype='gal',cols=['coadd','ra','dec','zp'],catfile='/share/c12/y1/redmagicv6.4.11/'+'y1a1_gold_1.0.2b-full_redmapper_v6.4.11_redmagic_highlum_1.0-04.fit',release='y1',ranfile='/share/c12/y1/redmagicv6.4.11/'+'y1a1_gold_1.0.2b-full_redmapper_v6.4.11_redmagic_highlum_1.0-04_randoms.fit')

def chi(z,omegam=0.27,H=100):
  from astropy import cosmology
  from astropy.cosmology import FlatLambdaCDM
  cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
  return cosmo.comoving_distance(z).value

rml.r=chi(rml.zp)
rml.ran_r=np.load('/home/troxel/highlum_ran_r.npy')

i3=catalog.CatalogStore('y1_i3_sv_v1',cattype='i3',cols=['coadd','ra','dec','psf1','psf2','psffwhm','rgp','snr','info'],catdir='/share/c12/y1/im3shape/single_band/r/y1v1/complete/main/',release='y1',tiles=np.load('/home/troxel/s82_tiles_b.npy'),cutfunc=None)
mask=(i3.info==0)&(i3.rgp>1.13)&(i3.snr>12)&(i3.snr<200)&(i3.rgp<3)&(~(np.isnan(i3.psf1)|np.isnan(i3.psf2)|np.isnan(i3.snr)|np.isnan(i3.psffwhm)))
catalog.CatalogMethods.match_cat(i3,mask)
i3ran=fio.FITS('/home/troxel/b_random.fits')[-1].read()

catxa=treecorr.Catalog(ra=i3.ra, dec=i3.dec, ra_units='deg', dec_units='deg')
catra=treecorr.Catalog(ra=i3ran['alphawin_j2000_i'], dec=i3ran['deltawin_j2000_i'], ra_units='deg', dec_units='deg')
nn = treecorr.NNCorrelation(nbins=20, min_sep=.1, max_sep=60., sep_units='arcmin',bin_slop=.01,verbose=0)
nr = treecorr.NNCorrelation(nbins=20, min_sep=.1, max_sep=60., sep_units='arcmin',bin_slop=.01,verbose=0)
rr = treecorr.NNCorrelation(nbins=20, min_sep=.1, max_sep=60., sep_units='arcmin',bin_slop=.01,verbose=0)
nn.process(catxa)
nr.process(catra,catxa)
rr.process(catra)
xip,xiperr=nn.calculateXi(rr,dr=nr)
xiperr=np.sqrt(xiperr)
theta=np.exp(nn.meanlogr)
np.save('i3_wt.npy',np.vstack((theta,xip,xiperr)))

zp=np.linspace(.1,.95,9)
for i in range(len(zp)-1):
  mask=(rml.zp>zp[i])&(rml.zp<=zp[i+1])
  maskr=(rml.ran_zp>zp[i])&(rml.ran_zp<=zp[i+1])
  catxa=treecorr.Catalog(ra=rml.ra[mask], dec=rml.dec[mask], r=rml.r[mask], ra_units='deg', dec_units='deg')
  catra=treecorr.Catalog(ra=rml.ran_ra[maskr], dec=rml.ran_dec[maskr], r=rml.ran_r[maskr], ra_units='deg', dec_units='deg')
  nn = treecorr.NNCorrelation(nbins=20, min_sep=.1, max_sep=10.,bin_slop=.01,verbose=0)
  nr = treecorr.NNCorrelation(nbins=20, min_sep=.1, max_sep=10.,bin_slop=.01,verbose=0)
  rr = treecorr.NNCorrelation(nbins=20, min_sep=.1, max_sep=10.,bin_slop=.01,verbose=0)
  nn.process(catxa,metric='Rperp',min_rpar=-.5,max_rpar=.5)
  nr.process(catra,catxa,metric='Rperp',min_rpar=-.5,max_rpar=.5)
  rr.process(catra,metric='Rperp',min_rpar=-.5,max_rpar=.5)
  xip,xiperr=nn.calculateXi(rr,dr=nr,metric='Rperp',min_rpar=-.5,max_rpar=.5)
  xiperr=np.sqrt(xiperr)
  theta=np.exp(nn.meanlogr)
  np.save('rml_wp_'+str(i)+'.npy',np.vstack((theta,xip,xiperr)))

  catxap=treecorr.Catalog(ra=i3.ra, dec=i3.dec, ra_units='deg', dec_units='deg')
  catrap=treecorr.Catalog(ra=i3ran['alphawin_j2000_i'], dec=i3ran['deltawin_j2000_i'], ra_units='deg', dec_units='deg')
  nn = treecorr.NNCorrelation(nbins=20, min_sep=.01, max_sep=100., sep_units='arcmin',bin_slop=.01,verbose=0)
  nr = treecorr.NNCorrelation(nbins=20, min_sep=.01, max_sep=100., sep_units='arcmin',bin_slop=.01,verbose=0)
  rn = treecorr.NNCorrelation(nbins=20, min_sep=.01, max_sep=100., sep_units='arcmin',bin_slop=.01,verbose=0)
  rr = treecorr.NNCorrelation(nbins=20, min_sep=.01, max_sep=100., sep_units='arcmin',bin_slop=.01,verbose=0)
  nn.process(catxa,catxap)
  nr.process(catra,catxap)
  rn.process(catxa,catrap)
  rr.process(catra,catrap)
  xip,xiperr=nn.calculateXi(rr,dr=nr,rd=rn)
  xiperr=np.sqrt(xiperr)
  theta=np.exp(nn.meanlogr)
  np.save('rml_i3_wt_'+str(i)+'.npy',np.vstack((theta,xip,xiperr)))



tmp0=np.array(glob.glob('/project/projectdirs/des/wl/desdata/wlpipe/im3shape_y1a1_v1/nbc/main/*'))
tmp=np.array(glob.glob('/project/projectdirs/des/wl/desdata/wlpipe/im3shape_y1a1_v1/nbc/main/*'))
for i in range(len(tmp)):
  tmp0[i]=tmp[i][-17:-5]
  tmp[i]=tmp[i][-10:-5]

tmp=tmp.astype(int)
np.save('s82_tiles.npy',tmp0[tmp>-10])


# r=np.load('r.npy')
# gp=np.load('gp.npy')
# gx=np.load('gx.npy')
# gperr=np.sqrt(np.diagonal(np.load('covgp.npy')))
# gxerr=np.sqrt(np.diagonal(np.load('covgx.npy')))

# chi2=0.
# for i in xrange(10):
#   for j in xrange(10):
#     chi2+ in files and build catalog class. See catalog.py for further details.
i3=catalog.CatalogStore('y1_i3_sv_v1',cattype='i3',cols=cols,catfile='/share/des/disc2/y1/im3shape/single_band/r/y1a1-im3shape-1-1-1/combined_r.fits',release='y1')
i3=catalog.CatalogStore('y1_i3_sv_v1',cattype='i3',cols=cols,catdir='/share/des/
nbc=nbc[mask]
x,y=catalog.CatalogMethods.sort2(nbc[:,0],i3.coadd)
catalog.CatalogMethods.match_cat(i3,y)
i3.m=nbc[x,1]
i3.c1=nbc[x,2]
i3.c2=nbc[x,3]
i3.w=nbc[x,4]
rm=catalog.CatalogStore('y1_rm_highdens',cattype='gal',cols=['coadd','ra','dec','zp'],catfile=config.redmagicdir+'y1a1_gold_1.0.2b-full_redmapper_v6.4.11_redmagic_highdens_0.5-10.fit',release='y1',ranfile=config.redmagicdir+'y1a1_gold_1.0.2b-full_redmapper_v6.4.11_redmagic_highdens_0.5-10_randoms.fit')

a=np.argsort(i3.coadd)
mask=np.diff(i3.coadd[a])
mask=mask==0
mask=~mask
mask=a[mask]
catalog.CatalogMethods.match_cat(i3,mask)

sim=catalog.CatalogStore('y1_sim_v1',cattype='i3',cols=cols,catdir='/share/des/disc2/y1/nbc/v0/bord/main/',release='y1')
sim=catalog.CatalogStore('y1_sim_v1',cattype='i3',cols=cols,catdir='/share/des/disc2/y1/nbc/v1/results/disc/main',release='y1')
sim2=catalog.CatalogStore('y1_sim_v1',cattype='i3',cols=cols,catdir='/share/des/disc2/y1/nbc/v0/disc/main/',release='y1')
truth=catalog.CatalogStore('y1-sim-truth-v1',cattype='i3',cols=['coadd','e1','e2'],catdir='/share/des/disc2/y1/nbc/v0/truth/',release='y1')

x,y=catalog.CatalogMethods.sort2(sim.coadd,i3.coadd)
catalog.CatalogMethods.match_cat(i3,y)
catalog.CatalogMethods.match_cat(sim,x)

cols=['chi2pix','psffwhm','nlike','flux','rgp','dec','evals','rad','dec_off','ra_off','dflux','fluxfrac','psf1','psf2','modmax','modmin','ra','resmax','maskfrac','snr','resmin','e1','e2','iter','bflux','dflux','flux']
lin.hist.hist_comp_tests(cols,sim,i3,mask=sim.info<=4,mask2=i3.info<=4)
lin.hist.hist_2D_tests(cols,cols,i3,mask=i3.info<=4)
lin.hist.hist_2D_tests(cols,cols,sim,mask=sim.info<=4)


cols=['chi2pix','psffwhm','nlike','rgp','dec','evals','rad','fluxfrac','psf1','psf2','modmax','modmin','ra','resmax','snr','resmin','flux']
sys_split.split.cat_splits_lin(cols,sim,mask=sim.info<=4)
sys_split.split.cat_splits_lin(cols,i3,mask=i3.info<=4)

i3.logdflux=np.log(i3.dflux*i3.flux)
i3.logbflux=np.log(i3.bflux*i3.flux)
sim.logdflux=np.log(sim.dflux*sim.flux)
sim.logbflux=np.log(sim.bflux*sim.flux)

catalog.CatalogMethods.match_cat(i3,~np.isinf(i3.snr))
catalog.CatalogMethods.match_cat(sim,~np.isinf(sim.snr))

x,y=catalog.CatalogMethods.sort2(sim.coadd[sim.info<=4],i3.coadd[i3.info<=4])
lin.summary_stats.e1_psf_stats(i3,mask=np.in1d(np.arange(len(i3.coadd)),y,assume_unique=True))
lin.summary_stats.e1_psf_stats(sim,mask=np.in1d(np.arange(len(sim.coadd)),x,assume_unique=True))

x,y=catalog.CatalogMethods.sort2(sim.coadd,truth.coadd)
catalog.CatalogMethods.match_cat(truth,y)
catalog.CatalogMethods.match_cat(sim,x)
def func(x,m0,a0):
  return ((1.+m0/x[2]/x[2])*x[0]+a0/x[2]/x[2]*x[1])
bins=5
bins2=5
st=np.zeros((bins,bins2))
sterr=np.zeros((bins,bins2))
st2=np.zeros((bins,bins2))
st2err=np.zeros((bins,bins2))
st4=np.zeros((bins,bins2))
sta=np.zeros((bins,bins2))
sterra=np.zeros((bins,bins2))
st2a=np.zeros((bins,bins2))
st2erra=np.zeros((bins,bins2))
st4a=np.zeros((bins,bins2))
snrbins=lin.linear_methods.find_bin_edges(sim.snr[sim.info==0],bins)
rgpbins=lin.linear_methods.find_bin_edges(sim.rgp[sim.info==0],bins2)
for i in range(bins):
  for j in range(bins2):
    mask=(sim.snr>snrbins[i])&(sim.snr<=snrbins[i+1])&(sim.rgp>rgpbins[j])&(sim.rgp<=rgpbins[j+1])
    arr1,arr1err,e1,e1err,e2,e2err=lin.linear_methods.bin_means(truth.e1[mask],sim,mask=sim.info[mask]==0,noe=True,y=sim.e1[mask]-truth.e1[mask])
    m1,b1,m1err,b1err=lin.fitting.lin_fit(arr1,e1,e1err)
    st[i,j]=m1
    sterr[i,j]=m1err
    st2[i,j]=b1
    st2err[i,j]=b1err
    st4[i,j]=np.mean(sim.psf1[(sim.info==0)&(mask)])
    arr1,arr1err,e1,e1err,e2,e2err=lin.linear_methods.bin_means(truth.e2[mask],sim,mask=sim.info[mask]==0,noe=True,y=sim.e2[mask]-truth.e2[mask])
    m1,b1,m1err,b1err=lin.fitting.lin_fit(arr1,e1,e1err)
    sta[i,j]=m1
    sterra[i,j]=m1err
    st2a[i,j]=b1
    st2erra[i,j]=b1err
    st4a[i,j]=np.mean(sim.psf2[(sim.info==0)&(mask)])

col=['r','g','b','c','m']
for i in range(bins2):
  mask=(sim.info<=4)&(sim.rgp>rgpbins[i])&(sim.rgp<=rgpbins[i+1])
  plt.errorbar((snrbins[1:]+snrbins[:-1])/2.*(1.+.1*i),st[:,i],yerr=sterr[:,i],ls='',color=col[i],label=str(np.around(((rgpbins[1:]+rgpbins[:-1])/2.)[i],2)))
  a,b=curve_fit(func,[truth.e1[mask],sim.psf1[mask],sim.snr[mask]/100.],sim.e1[mask],p0=(0.,0.))
  print i,a
  plt.plot((snrbins[1:]+snrbins[:-1])/2.,a[0]/((snrbins[1:]+snrbins[:-1])/2./100.)**2,marker='',color=col[i])

plt.xlim((10,2000))
plt.xlabel('snr')
plt.xscale('log')
plt.legend(loc='upper left',ncol=1, frameon=False,prop={'size':12},framealpha=0.2)
plt.savefig('tmp.png')
plt.close()

for i in range(bins2):
  mask=(sim.info<=4)&(sim.rgp>rgpbins[i])&(sim.rgp<=rgpbins[i+1])
  plt.errorbar((snrbins[1:]+snrbins[:-1])/2.*(1.+.1*i),st2[:,i],yerr=st2err[:,i],ls='',label=str(np.around(((rgpbins[1:]+rgpbins[:-1])/2.)[i],2)))
  a,b=curve_fit(func,[truth.e1[mask],sim.psf1[mask],sim.snr[mask]/100.],sim.e1[mask],p0=(0.,0.))
  print i,a
  plt.plot((snrbins[1:]+snrbins[:-1])/2.,a[1]/((snrbins[1:]+snrbins[:-1])/2./100.)**2*st4[:,i],marker='',color=col[i])

plt.xlim((10,2000))
plt.xlabel('snr')
plt.xscale('log')
plt.legend(loc='upper left',ncol=1, frameon=False,prop={'size':12},framealpha=0.2)
plt.savefig('tmp2.png')
plt.close()

for i in range(bins2):
  mask=(sim.info<=4)&(sim.rgp>rgpbins[i])&(sim.rgp<=rgpbins[i+1])
  plt.errorbar((snrbins[1:]+snrbins[:-1])/2.*(1.+.1*i),sta[:,i],yerr=sterra[:,i],ls='',color=col[i],label=str(np.around(((rgpbins[1:]+rgpbins[:-1])/2.)[i],2)))
  a,b=curve_fit(func,[truth.e2[mask],sim.psf2[mask],sim.snr[mask]/100.],sim.e2[mask],p0=(0.,0.))
  print i,a
  plt.plot((snrbins[1:]+snrbins[:-1])/2.,a[0]/((snrbins[1:]+snrbins[:-1])/2./100.)**2,marker='',color=col[i])

plt.xlim((10,2000))
plt.xlabel('snr')
plt.xscale('log')
plt.legend(loc='upper left',ncol=1, frameon=False,prop={'size':12},framealpha=0.2)
plt.savefig('tmpa.png')
plt.close()

for i in range(bins2):
  mask=(sim.info<=4)&(sim.rgp>rgpbins[i])&(sim.rgp<=rgpbins[i+1])
  plt.errorbar((snrbins[1:]+snrbins[:-1])/2.*(1.+.1*i),st2a[:,i],yerr=st2erra[:,i],ls='',label=str(np.around(((rgpbins[1:]+rgpbins[:-1])/2.)[i],2)))
  a,b=curve_fit(func,[truth.e2[mask],sim.psf2[mask],sim.snr[mask]/100.],sim.e2[mask],p0=(0.,0.))
  print i,a
  plt.plot((snrbins[1:]+snrbins[:-1])/2.,a[1]/((snrbins[1:]+snrbins[:-1])/2./100.)**2*st4a[:,i],marker='',color=col[i])

plt.xlim((10,2000))
plt.xlabel('snr')
plt.xscale('log')
plt.legend(loc='upper left',ncol=1, frameon=False,prop={'size':12},framealpha=0.2)
plt.savefig('tmp2a.png')
plt.close()

plt.imshow(st.T,interpolation='bicubic')
plt.colorbar(orientation='horizontal')
plt.savefig('tmpa.png')
plt.close()

plt.imshow(st2.T,interpolation='bicubic')
plt.colorbar(orientation='horizontal')
plt.savefig('tmp2a.png')
plt.close()


def func(x,m0,a0):
  return ((1.+m0/x[2]/x[2])*x[0]+a0/x[2]/x[2]*x[1])

def func2(x,m0):
  return ((1.+m0/x[2]/x[2])*x[0])

curve_fit(func,[truth.e1[sim.info<=4],sim.psf1[sim.info<=4],sim.snr[sim.info<=4]/100.],sim.e1[sim.info<=4],p0=(-.01,0.01))





from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import RidgeCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
catalog.CatalogMethods.match_cat(truth,sim.info==0)
catalog.CatalogMethods.match_cat(sim,sim.info==0)

mask=np.zeros(len(sim.coadd)).astype(bool)
mask[np.random.choice(np.arange(len(sim.coadd)),len(sim.coadd)*9/10)]=True

tree=DecisionTreeRegressor()




tree=ExtraTreesRegressor()
tree=LinearRegression()
tree=RidgeCV()
tree=KNeighborsRegressor(n_neighbors=500,weights='distance',algorithm='kd_tree')

tree=DecisionTreeRegressor()

rf = RandomForestRegressor(n_estimators=1000)
rf.fit(np.vstack((sim.e1[mask],sim.psf1[mask],sim.e2[mask],sim.psf2[mask],sim.snr[mask],sim.rgp[mask],sim.psffwhm[mask])).T,np.vstack((truth.e1[mask],truth.e2[mask])).T)

a=rf.predict(np.vstack((sim.e1[~mask],sim.psf1[~mask],sim.e2[~mask],sim.psf2[~mask],sim.snr[~mask],sim.rgp[~mask],sim.psffwhm[~mask])).T)
plt.hist2d(truth.e1[~mask],a[:,0],bins=100)
plt.savefig('tmp.png')
plt.close()


cols=['chi2pix','psffwhm','nlike','fluxfrac','psf1','psf2','snr','flux']
bins=10
for col in cols:
  st=np.zeros((bins))
  sterr=np.zeros((bins))
  st2=np.zeros((bins))
  st2err=np.zeros((bins))
  snrbins=lin.linear_methods.find_bin_edges(getattr(sim,col)[sim.info==0],bins)
  for i in range(bins):
    print col,i
    mask=(getattr(sim,col)>snrbins[i])&(getattr(sim,col)<=snrbins[i+1])&(sim.info==0)&(sim.rgp>2)
    arr1,arr1err,e1,e1err,e2,e2err=lin.linear_methods.bin_means(truth.e1[mask],sim,mask=sim.info[mask]==0,noe=True,y=sim.e1[mask]-truth.e1[mask])
    m1,b1,m1err,b1err=lin.fitting.lin_fit(arr1,e1,e1err)
    print m1,b1,m1err,b1err
    st[i]=m1
    sterr[i]=m1err
    st2[i]=b1
    st2err[i]=b1err
  plt.errorbar((snrbins[1:]+snrbins[:-1])/2.*(1.+.1*i),st,yerr=sterr)
  plt.xlabel(col)
  if config.log_val.get(col,None)==True:
    plt.xscale('log')
  plt.savefig('m_'+col+'.png')
  plt.close()
  plt.errorbar((snrbins[1:]+snrbins[:-1])/2.*(1.+.1*i),st2,yerr=st2err)
  plt.xlabel(col)
  if config.log_val.get(col,None)==True:
    plt.xscale('log')
  plt.savefig('c_'+col+'.png')
  plt.close()



  #Load in systematics maps and match to galaxy positions.
  sys_split.split_methods.load_maps(i3)

  # Select columns to read in from epoch catalog files. A dictionary in config.py translates these shortnames to the actual column names in the files.
  cols=['coadd','expnum','xoff','yoff','psf1_exp','psf2_exp','ccd','row','col','e1','e2']

  #Read in files and build catalog class. See catalog.py for further details.
  i3epoch=catalog.CatalogStore('y1_i3_sv_epoch_v1',cutfunc=None,cattype='i3',cols=cols,catdir='/share/des/disc2/y1/im3shape/single_band/r/y1v1/spte_sv/epoch_cats/',release='y1',maxiter=500)

  #Match epoch entries to main catalog entries.
  epochmask=np.in1d(i3epoch.coadd,i3.coadd[i3.info==0],assume_unique=False)

  #Read in lens catalog for tangential shear.
  rmdens=catalog.CatalogStore('y1_rm_highdens',cattype='gal',catfile=config.redmagicdir+'y1a1_gold_1.0-pre2_run_redmapper_v6.4.4_redmagic_0.5-10_wshapes.fit',release='y1',ranfile=config.redmagicdir+'y1_rm_05_random.fits.gz')
  rmlum=catalog.CatalogStore('y1_rm_highlum',cattype='gal',catfile=config.redmagicdir+'y1a1_gold_1.0-pre2_run_redmapper_v6.4.4_redmagic_1.0-02_wshapes.fit',release='y1',ranfile=config.redmagicdir+'y1_rm_10_random.fits.gz')

  rm.ra=rm.ra/180.*np.pi
  rm.dec=rm.dec/180.*np.pi
  rm2.ra=rm2.ra/180.*np.pi
  rm2.dec=rm2.dec/180.*np.pi
  rm.ran_ra=rm.ran_ra/180.*np.pi
  rm.ran_dec=rm.ran_dec/180.*np.pi
  rm2.ran_ra=rm2.ran_ra/180.*np.pi
  rm2.ran_dec=rm2.ran_dec/180.*np.pi

  #Run full set of shear tests on shear catalog. Separate tests are available for comparing two catalogs, but untested in this version - commented out above.
  single_tests(i3epoch,epochmask,i3,i3.info==0,rm10)


  ###   Photo-z testing Examples    ###

  # Load photo-z catalog of pdfs - h5 version

  sn=catalog.PZStore('skynet',setup=True,pztype='SKYNET',filetype='h5',file='sv_skynet_final.h5')

  # Build photo-z bins from PZStore object

  pz.pz_methods.build_nofz_bins(sn,pzlow=0.3,pzhigh=1.3,cat=None,bins=3,split='mean',pzmask=None,catmask=None)

  # Load photo-z n(z)'s and bootstrap samples from standard dict file used for spec validation.
  
  sn=catalog.PZStore('skynet',setup=True,pztype='SKYNET',filetype='dict',file='WL_test_3_bins.pickle')

  # Plot comparison of photo-z vs spec.

  fig.plot_methods.plot_nofz(sn,'pz_test')

  # Write cosmosis n(z) files from PZStore object.

  cosmosis.make.nofz(sn,'pz_test')

  # Submit cosmosis runs for spec validation
  # With submit=False, this produces bash scripts in the root directory of destest that must be run separately.

  cosmosis.run.submit_pz_spec_test(sn,'pz_test',boot=False,cosmo=False,submit=False)
  cosmosis.run.submit_pz_spec_test(sn,'pz_test',boot=True,cosmo=False,submit=False)
  # First two jobs must finish writing simulated data before running second two commands
  cosmosis.run.submit_pz_spec_test(sn,'pz_test',boot=False,cosmo=True,submit=False)
  cosmosis.run.submit_pz_spec_test(sn,'pz_test',boot=True,cosmo=True,submit=False)

  # When cosmosis jobs finished, calculate output figures and stats

  fig.plot_methods.plot_pz_sig8('pz_test',boot=True,tomobins=3)


cosmosis.run.submit(label='tomo_all_w3',nodes=1,procs=32,hr=96,pts=1000,mneff=0.5,mntol=0.5,ia=True,pz=True,mbias=True,planck=False,tomobins=3,data='/home/troxel/cosmosis/cosmosis-des-library/photoztests/y1/threebins/out/sim_data_skynet/data.txt',cov='/home/troxel/cosmosis/cosmosis-des-library/photoztests/y1/threebins/out/sim_data_skynet/covmat.txt',nofz='/home/troxel/cosmosis/cosmosis-des-library/photoztests/y1/threebins/nofz/skynet.txt',cldir='/home/troxel/cosmosis/cosmosis-des-library/photoztests/y1/threebins/out/sim_data_skynet',resume=False)
  
cosmosis.run.submit(label='tomo_all_planck_w3',nodes=1,procs=32,hr=96,pts=1000,mneff=0.5,mntol=0.5,ia=True,pz=True,mbias=True,planck=True,tomobins=3,data='/home/troxel/cosmosis/cosmosis-des-library/photoztests/y1/threebins/out/sim_data_skynet/data.txt',cov='/home/troxel/cosmosis/cosmosis-des-library/photoztests/y1/threebins/out/sim_data_skynet/covmat.txt',nofz='/home/troxel/cosmosis/cosmosis-des-library/photoztests/y1/threebins/nofz/skynet.txt',cldir='/home/troxel/cosmosis/cosmosis-des-library/photoztests/y1/threebins/out/sim_data_skynet',resume=False)

cosmosis.run.submit(label='tomo_all_planck_w4',nodes=1,procs=32,hr=96,pts=1000,mneff=0.5,mntol=0.5,ia=True,pz=True,mbias=True,planck=True,tomobins=4,data='/home/troxel/cosmosis/cosmosis-des-library/photoztests/y1/fourbins/out/sim_data_skynet/data.txt',cov='/home/troxel/cosmosis/cosmosis-des-library/photoztests/y1/fourbins/out/sim_data_skynet/covmat.txt',nofz='/home/troxel/cosmosis/cosmosis-des-library/photoztests/y1/fourbins/nofz/skynet.txt',cldir='/home/troxel/cosmosis/cosmosis-des-library/photoztests/y1/fourbins/out/sim_data_skynet',resume=True)

cosmosis.run.submit(label='tomo_all_planck_w5',nodes=1,procs=32,hr=96,pts=1000,mneff=0.5,mntol=0.5,ia=True,pz=True,mbias=True,planck=True,tomobins=5,data='/home/troxel/cosmosis/cosmosis-des-library/photoztests/y1/fivebins/out/sim_data_skynet/data.txt',cov='/home/troxel/cosmosis/cosmosis-des-library/photoztests/y1/fivebins/out/sim_data_skynet/covmat.txt',nofz='/home/troxel/cosmosis/cosmosis-des-library/photoztests/y1/fivebins/nofz/skynet.txt',cldir='/home/troxel/cosmosis/cosmosis-des-library/photoztests/y1/fivebins/out/sim_data_skynet',resume=True)

cosmosis.run.submit(label='tomo_all_planck_w6',nodes=1,procs=32,hr=96,pts=1000,mneff=0.5,mntol=0.5,ia=True,pz=True,mbias=True,planck=True,tomobins=6,data='/home/troxel/cosmosis/cosmosis-des-library/photoztests/y1/sixbins/out/sim_data_skynet/data.txt',cov='/home/troxel/cosmosis/cosmosis-des-library/photoztests/y1/sixbins/out/sim_data_skynet/covmat.txt',nofz='/home/troxel/cosmosis/cosmosis-des-library/photoztests/y1/sixbins/nofz/skynet.txt',cldir='/home/troxel/cosmosis/cosmosis-des-library/photoztests/y1/sixbins/out/sim_data_skynet',resume=True)


cosmosis.run.submit(label='tomo_all_planck_w3b',nodes=1,procs=32,hr=96,pts=200,mneff=0.5,mntol=0.5,ia=True,pz=True,mbias=True,planck=True,tomobins=3,data='/home/troxel/cosmosis/cosmosis-des-library/photoztests/y1/threebins/out/sim_data_skynet/data.txt',cov='/home/troxel/cosmosis/cosmosis-des-library/photoztests/y1/threebins/out/sim_data_skynet/covmat.txt',nofz='/home/troxel/cosmosis/cosmosis-des-library/photoztests/y1/threebins/nofz/skynet.txt',cldir='/home/troxel/cosmosis/cosmosis-des-library/photoztests/y1/threebins/out/sim_data_skynet',resume=False,submit=False)


query="""SELECT expnum, ccdnum, tilename, cunit1, cunit2, ctype1, ctype2, crval1, crval2, crpix1, crpix2, cd1_1, cd1_2, cd2_1, cd2_2, pv1_0, pv1_1, pv1_2, pv1_3, pv1_4, pv1_5, pv1_6, pv1_7, pv1_8, pv1_9, pv1_10, pv2_0, pv2_1, pv2_2, pv2_3, pv2_4, pv2_5, pv2_6, pv2_7, pv2_8, pv2_9, pv2_10 FROM erykoff.y1a1_coadd_astrometry where rownum<10"""
QQ=cursor.execute(query)
header = [item[0] for item in cursor.description]
rows = cursor.fetchall()  ## Bring the data

query="""SELECT expnum FROM erykoff.y1a1_coadd_astrometry where rownum<2"""
QQ=cursor.execute(query)
header = [item[0] for item in cursor.description]
rows = cursor.fetchall()  ## Bring the data

query="""SELECT * FROM des_admin.y1a1_image where rownum<2"""
QQ=cursor.execute(query)
header = [item[0] for item in cursor.description]
rows = cursor.fetchall()  ## Bring the data



import meds

# create a MEDS object for the given MEDS file
m=meds.MEDS(filename)

# get coadd_objects_id for object index
coadd_objects_id=m['id'][index]

# get exposure list for object at row index, ignoring coadd
exp=m.get_cutout_list(index, type='weight')[1:]

# get weight value for each exposure of object index
wt=[]
for i in range(len(exp)):
  wt=np.append(wt,np.max(exp[i]))

# build string list of imgage id and ccd number for each exposure for object index
imgid=[]
ccdnum=[]
for i in range(len(exp)):
  source_path=m.get_source_path(index,i+1)
  source = os.path.split(source_path)[1].split('.')[0]
  imgid=np.append(imgid,source[6:14])
  ccdnum=np.append(ccdnum,source[15:17])

cols=['psfpos','dpsf','chi2pix','psffwhm','nlike','flux','rgp','evals','rad','dflux','invfluxfrac','psf1','psf2','modmax','modmin','resmax','maskfrac','snr','resmin','iter','pz','r','g','i','z']
sys_split.split.cat_splits_lin(cols,i3,mask=mask)

for n in [1,2,3,4]:
  for i in [2048,4096]:
    for j in [1,2,3,4,5,6,7,8,9,10]:
      print n,i,j
      catalog.CatalogMethods.footprint_area(i3,nexp=n,ngal=j,mask=mask,nside=i,nest=True)

cols=['stamp','coadd','fluxfrac','e1','e2']



pix=hp.ang2pix(4096, np.pi/2.-np.radians(i3.dec),np.radians(i3.ra), nest=False)
gdmask=hp.read_map(config.golddir+'y1a1_gold_1.0.1_wide_footprint_4096.fit')
badmask=hp.read_map(config.golddir+'y1a1_gold_1.0.1_wide_badmask_4096.fit')
mask=(gdmask>=1)&(badmask==0)

hpmap=np.ones((12*4096**2))*hp.UNSEEN
hpmap[(gdmask>=1)]=0
cnt=np.zeros(12*4096**2)
cnt[:np.max(pix)+1]=np.bincount(pix)
hpmap[mask]=cnt[mask] 
hp.cartview(hpmap,latra=config.dec_lim.get('sptc'),lonra=config.ra_lim.get('sptc'),xsize=10000)
plt.savefig('test.png', dpi=1000,bbox_inches='tight')
plt.close()


cosmosis.run.submit(label='tomo_all_planck_w3',nodes=1,procs=32,hr=96,pts=1000,mneff=0.5,mntol=0.5,ia=True,pz=True,mbias=True,planck=False,tomobins=3,data='/home/troxel/cosmosis/cosmosis-des-library/photoztests/y1/threebins/out/sim_data_skynet/data.txt',cov='/home/troxel/cosmosis/cosmosis-des-library/photoztests/y1/threebins/out/sim_data_skynet/covmat.txt',nofz='/home/troxel/cosmosis/cosmosis-des-library/photoztests/y1/threebins/nofz/skynet.txt',cldir='/home/troxel/cosmosis/cosmosis-des-library/photoztests/y1/threebins/out/sim_data_skynet',resume=True)

'test_zbins_3bin_251','test_zbins_3bin_200','test_zbins_3bin_150','test_zbins_3bin_100','test_zbins_3bin_50'
'test_zbins_6bin_251','test_zbins_6bin_200','test_zbins_6bin_150','test_zbins_6bin_100','test_zbins_6bin_50'
'test_zmax_3bin_50','test_zmax_3bin_30','test_zmax_3bin_25','test_zmax_3bin_20','test_zmax_3bin_175','test_zmax_3bin_15'
'test_zmax_6bin_50','test_zmax_6bin_30','test_zmax_6bin_25','test_zmax_6bin_20','test_zmax_6bin_175','test_zmax_6bin_15'

for test in ['test_zmax_3bin_50','test_zmax_3bin_30','test_zmax_3bin_25','test_zmax_3bin_20','test_zmax_3bin_175','test_zmax_3bin_15','test_zmax_6bin_50','test_zmax_6bin_30','test_zmax_6bin_25','test_zmax_6bin_20','test_zmax_6bin_175','test_zmax_6bin_15']:
  cosmosis.run.submit_pz_spec_test(bpz,test,boot=False,cosmo=False,submit=True,procs=8)

cosmosis.run.submit(label='simon_sig8_pz2',nodes=1,procs=32,hr=96,pts=100,mneff=0.5,mntol=0.5,ia=False,pz=True,mbias=False,planck=False,tomobins=3,data='/home/troxel/cosmosis/cosmosis-des-library/wl/y1prep/data/data.txt',cov='/home/troxel/cosmosis/cosmosis-des-library/wl/y1prep/data/covmat.txt',nofz='/home/troxel/cosmosis/cosmosis-des-library/wl/y1prep/data/n_of_zs.hist',cldir='/home/troxel/cosmosis/cosmosis-des-library/wl/y1prep/data',resume=False)




# python bpz.py /home/troxel/bpz/data/spec_bord.txt -NEW_AB no -DZ 0.01 -ODDS 0.68 -PRIOR flat -INTERP 8 -MIN_RMS 0.067 -ZMAX 5. -PHOTO_ERRORS yes -MIN_MAGERR 0.05 -PROBS_LITE yes -VERBOSE no -COLUMNS /home/troxel/bpz/data/spec_bord.columns -SPECTRA CWWSB4.list -SED_DIR /home/troxel/bpz/bpz/SED/ -FILTER_DIR /home/troxel/bpz/data/FILTERS/ -OUTPUT /home/troxel/bpz/data/output_bord_false.bpz



tmp=fio.FITS('/share/des/disc2/y1/im3shape/likelihoods/full/v1.fits')[-1].read()
bord=0.2*np.exp((tmp['bulge_like']-tmp['disc_like']))
bord=bord/(1.+bord)
bord[np.isnan(bord)]=1.
plt.hist(bord,bins=100)
plt.savefig('tmp.png')
plt.close()


spec=fio.FITS('/home/troxel/spec_cat_0.fits.gz')[-1].read()
mask=np.in1d(spec['coadd_objects_id'],tmp['coadd_objects_id'],assume_unique=True)
m1,s1,m2,s2=catalog.CatalogMethods.sort(spec['coadd_objects_id'],tmp['coadd_objects_id'])
spec=spec[m1]
spec=spec[s1]

out=np.vstack((spec['mag_auto_y'],spec['mag_auto_g'],spec['mag_auto_r'],spec['mag_auto_i'],spec['mag_auto_z'],spec['magerr_auto_y'],spec['magerr_auto_g'],spec['magerr_auto_r'],spec['magerr_auto_i'],spec['magerr_auto_z'],spec['coadd_objects_id'],bord)).T
np.savetxt('/home/troxel/bpz/data/bord2.txt',out)

# python bpz.py /home/troxel/bpz/data/bord2.txt -NEW_AB no -DZ 0.01 -ODDS 0.68 -PRIOR flat -INTERP 8 -MIN_RMS 0.067 -ZMAX 5. -PHOTO_ERRORS yes -MIN_MAGERR 0.05 -PROBS_LITE yes -VERBOSE no -COLUMNS /home/troxel/bpz/data/spec_bord_false.columns -SPECTRA CWWSB4.list -SED_DIR /home/troxel/bpz/bpz/SED/ -FILTER_DIR /home/troxel/bpz/data/FILTERS/ -OUTPUT /home/troxel/bpz/data/bord_false.bpz

# mv /home/troxel/bpz/data/bord2.probs /home/troxel/bpz/data/bord_false.probs

# python bpz.py /home/troxel/bpz/data/bord2.txt -NEW_AB no -DZ 0.01 -ODDS 0.68 -PRIOR flat -INTERP 8 -MIN_RMS 0.067 -ZMAX 5. -PHOTO_ERRORS yes -MIN_MAGERR 0.05 -PROBS_LITE yes -VERBOSE no -COLUMNS /home/troxel/bpz/data/spec_bord_true.columns -SPECTRA CWWSB4.list -SED_DIR /home/troxel/bpz/bpz/SED/ -FILTER_DIR /home/troxel/bpz/data/FILTERS/ -OUTPUT /home/troxel/bpz/data/bord_true.bpz

# mv /home/troxel/bpz/data/bord2.probs /home/troxel/bpz/data/bord_true.probs

# python bpz.py /home/troxel/bpz/data/bord2.txt -NEW_AB no -DZ 0.01 -ODDS 0.68 -INTERP 8 -MIN_RMS 0.067 -ZMAX 5. -PHOTO_ERRORS yes -MIN_MAGERR 0.05 -PROBS_LITE yes -VERBOSE no -COLUMNS /home/troxel/bpz/data/spec_bord_false.columns -SPECTRA CWWSB4.list -SED_DIR /home/troxel/bpz/bpz/SED/ -FILTER_DIR /home/troxel/bpz/data/FILTERS/ -OUTPUT /home/troxel/bpz/data/bord_false_prior.bpz

# mv /home/troxel/bpz/data/bord2.probs /home/troxel/bpz/data/bord_false_prior.probs

# python bpz.py /home/troxel/bpz/data/bord2.txt -NEW_AB no -DZ 0.01 -ODDS 0.68 -INTERP 8 -MIN_RMS 0.067 -ZMAX 5. -PHOTO_ERRORS yes -MIN_MAGERR 0.05 -PROBS_LITE yes -VERBOSE no -COLUMNS /home/troxel/bpz/data/spec_bord_true.columns -SPECTRA CWWSB4.list -SED_DIR /home/troxel/bpz/bpz/SED/ -FILTER_DIR /home/troxel/bpz/data/FILTERS/ -OUTPUT /home/troxel/bpz/data/bord_true_prior.bpz

# mv /home/troxel/bpz/data/bord2.probs /home/troxel/bpz/data/bord_true_prior.probs

bfb=np.loadtxt('/home/troxel/bpz/data/bord_false.bpz')
bfp=np.loadtxt('/home/troxel/bpz/data/bord_false.probs')
btb=np.loadtxt('/home/troxel/bpz/data/bord_true.bpz')
btp=np.loadtxt('/home/troxel/bpz/data/bord_true.probs')

bfba=np.loadtxt('/home/troxel/bpz/data/bord_false_prior.bpz')
bfpa=np.loadtxt('/home/troxel/bpz/data/bord_false_prior.probs')
btba=np.loadtxt('/home/troxel/bpz/data/bord_true_prior.bpz')
btpa=np.loadtxt('/home/troxel/bpz/data/bord_true_prior.probs')

spec=fio.FITS('/home/troxel/spec_cat_0.fits.gz')[-1].read()
m1,s1,m2,s2=catalog.CatalogMethods.sort(spec['coadd_objects_id'],bfp[:,0])




a=np.argsort(i3.coadd)
mask=np.diff(i3.coadd[a])
mask=mask==0
mask=~mask
mask=a[mask]
catalog.CatalogMethods.match_cat(i3,mask)




spec=spec[m1]
spec=spec[s1]

bpzt=catalog.PZStore('bpz_bord_true',setup=False,pztype='BPZ',filetype='h5',file='sv_skynet_final.h5')
bpzt.bin=np.arange(0.0100,5.0100,0.0100)
bpzt.bins=len(bpzt.bin)
bpzt.pdftype='full'
bpzt.pz_full=btp[:,1:]
bpzt.w=spec['weights']
bpzt.z_mean_full=np.zeros(len(bpzt.pz_full))
bpzt.coadd=btp[:,1:]
bpzt.spec_full=spec['z_spec']
for i in range(len(bpzt.pz_full)):
  bpzt.z_mean_full[i]=np.average(bpzt.bin,weights=bpzt.pz_full[i])

bpzf=catalog.PZStore('bpz_bord_false',setup=False,pztype='BPZ',filetype='h5',file='sv_skynet_final.h5')
bpzf.bin=np.arange(0.0100,5.0100,0.0100)
bpzf.bins=len(bpzf.bin)
bpzf.pdftype='full'
bpzf.pz_full=bfp[:,1:]
bpzf.w=spec['weights']
bpzf.z_mean_full=np.zeros(len(bpzf.pz_full))
bpzf.coadd=bfp[:,1:]
bpzf.spec_full=spec['z_spec']
for i in range(len(bpzf.pz_full)):
  bpzf.z_mean_full[i]=np.average(bpzf.bin,weights=bpzf.pz_full[i])

bpzta=catalog.PZStore('bpz_bord_true_prior',setup=False,pztype='BPZ',filetype='h5',file='sv_skynet_final.h5')
bpzta.bin=np.arange(0.0100,5.0100,0.0100)
bpzta.bins=len(bpzta.bin)
bpzta.pdftype='full'
bpzta.pz_full=btpa[:,1:]
bpzta.w=spec['weights']
bpzta.z_mean_full=np.zeros(len(bpzta.pz_full))
bpzta.coadd=btpa[:,1:]
bpzta.spec_full=spec['z_spec']
for i in range(len(bpzta.pz_full)):
  bpzta.z_mean_full[i]=np.average(bpzta.bin,weights=bpzta.pz_full[i])

bpzfa=catalog.PZStore('bpz_bord_false_prior',setup=False,pztype='BPZ',filetype='h5',file='sv_skynet_final.h5')
bpzfa.bin=np.arange(0.0100,5.0100,0.0100)
bpzfa.bins=len(bpzfa.bin)
bpzfa.pdftype='full'
bpzfa.pz_full=bfpa[:,1:]
bpzfa.w=spec['weights']
bpzfa.z_mean_full=np.zeros(len(bpzfa.pz_full))
bpzfa.coadd=bfpa[:,1:]
bpzfa.spec_full=spec['z_spec']
for i in range(len(bpzfa.pz_full)):
  bpzfa.z_mean_full[i]=np.average(bpzfa.bin,weights=bpzfa.pz_full[i])

bpzwill=np.loadtxt('/home/troxel/bpz_will.txt')
spec=fio.FITS('/home/troxel/spec_cat_0.fits.gz')[-1].read()
m1,s1,m2,s2=catalog.CatalogMethods.sort(spec['coadd_objects_id'],bpzwill[:,0])
spec=spec[m1]
spec=spec[s1]
bpzwill=bpzwill[m2]

bpzw=catalog.PZStore('bpz_will',setup=False,pztype='BPZ',filetype='h5',file='sv_skynet_final.h5')
bpzw.bin=np.arange(0.0100,10.0100,0.0100)
bpzw.bins=len(bpzw.bin)
bpzw.pdftype='full'
bpzw.pz_full=bpzwill[:,1:]
bpzw.w=spec['weights']
bpzw.z_mean_full=np.zeros(len(bpzw.pz_full))
bpzw.coadd=bpzwill[:,0]
bpzw.spec_full=spec['z_spec']
for i in range(len(bpzw.pz_full)):
  bpzw.z_mean_full[i]=np.average(bpzw.bin,weights=bpzw.pz_full[i])

pz.pz_methods.build_nofz_bins(bpzt,pzlow=0.0,pzhigh=5.,cat=None,bins=3,split='mean',pzmask=None,catmask=None,spec=False,point=False)
pz.pz_methods.build_nofz_bins(bpzf,pzlow=0.0,pzhigh=5.,cat=None,bins=3,split='mean',pzmask=None,catmask=None,spec=False,point=False)
pz.pz_methods.build_nofz_bins(bpzw,pzlow=0.0,pzhigh=5.,cat=None,bins=3,split='mean',pzmask=None,catmask=None,spec=True,point=False)
pz.pz_methods.build_nofz_bins(bpzta,pzlow=0.0,pzhigh=5.,cat=None,bins=3,split='mean',pzmask=None,catmask=None,spec=False,point=False)
pz.pz_methods.build_nofz_bins(bpzfa,pzlow=0.0,pzhigh=5.,cat=None,bins=3,split='mean',pzmask=None,catmask=None,spec=False,point=False)

fig.plot_methods.plot_nofz_comp([bpzw,bpzt,bpzf,bpzta,bpzfa],'bpz_comp_3',spec=True)

pz.pz_methods.build_nofz_bins(bpzt,pzlow=0.0,pzhigh=5.,cat=None,bins=6,split='mean',pzmask=None,catmask=None,spec=False,point=False)
pz.pz_methods.build_nofz_bins(bpzf,pzlow=0.0,pzhigh=5.,cat=None,bins=6,split='mean',pzmask=None,catmask=None,spec=False,point=False)
pz.pz_methods.build_nofz_bins(bpzw,pzlow=0.0,pzhigh=5.,cat=None,bins=6,split='mean',pzmask=None,catmask=None,spec=True,point=False)
pz.pz_methods.build_nofz_bins(bpzta,pzlow=0.0,pzhigh=5.,cat=None,bins=6,split='mean',pzmask=None,catmask=None,spec=False,point=False)
pz.pz_methods.build_nofz_bins(bpzfa,pzlow=0.0,pzhigh=5.,cat=None,bins=6,split='mean',pzmask=None,catmask=None,spec=False,point=False)

fig.plot_methods.plot_nofz_comp([bpzw,bpzt,bpzf,bpzta,bpzfa],'bpz_comp_6',spec=True)

plt.hist2d(bpzt.z_mean_full,bpzt.spec_full,bins=500,range=((0,2.5),(0,2.5)),norm=LogNorm())
plt.savefig('bpz_spec_comp_bord_true.png')
plt.close()
plt.hist2d(bpzf.z_mean_full,bpzf.spec_full,bins=500,range=((0,2.5),(0,2.5)),norm=LogNorm())
plt.savefig('bpz_spec_comp_bord_false.png')
plt.close()
plt.hist2d(bpzta.z_mean_full,bpzta.spec_full,bins=500,range=((0,2.5),(0,2.5)),norm=LogNorm())
plt.savefig('bpz_spec_comp_bord_true_prior.png')
plt.close()
plt.hist2d(bpzfa.z_mean_full,bpzfa.spec_full,bins=500,range=((0,2.5),(0,2.5)),norm=LogNorm())
plt.savefig('bpz_spec_comp_bord_false_prior.png')
plt.close()
plt.hist2d(bpzw.z_mean_full,bpzw.spec_full,bins=500,range=((0,2.5),(0,2.5)),norm=LogNorm())
plt.savefig('bpz_spec_comp_will.png')
plt.close()


ax=plt.subplot()
plt.plot(bpzta.bin,bpzta.pz[0],color='r')
plt.plot(bpzfa.bin,bpzfa.pz[0],color='g')
plt.plot(bpzw.bin,bpzw.pz[0],color='b')
plt.plot(bpzw.bin,bpzw.spec[0],color='k')
# ax.axvline(x=np.average(bpzta.bin,weights=bpzta.pz[0]),color='r',ls='solid') 
# ax.axvline(x=np.average(bpzfa.bin,weights=bpzfa.pz[0]),color='g',ls='solid') 
# ax.axvline(x=np.average(bpzw.bin,weights=bpzw.pz[0]),color='b',ls='solid') 
# ax.axvline(x=np.average(bpzw.bin,weights=bpzw.spec[0]),color='k',ls='solid') 
plt.xlim((0,2.5))
plt.savefig('tmp.png')
plt.close()


from src.weighted_kde import gaussian_kde
tmp=gaussian_kde(bpzta.spec_full,weights=bpzta.w,bw_method='scott')
nofz=tmp(bpzta.bin)
nofz/=np.sum(nofz)




r=np.load('r.npy')
gp=np.load('gp.npy')
gx=np.load('gx.npy')
ee=np.load('ee.npy')
xx=np.load('xx.npy')
covgp=np.load('covgp.npy')
covgx=np.load('covgx.npy')
covee=np.load('covee.npy')
covxx=np.load('covxx.npy')


import fitsio as fio

bpz=catalog.PZStore('bpz',setup=False,pztype='BPZ')
bpz.pdftype='sample'
fits=fio.FITS('/share/des/disc2/y1/photo_z/PHOTOZ_BPZ_Y1_v0.1.fits')[-1].read()
bpz.coadd=fits['coadd_objects_id']
bpz.z_mean_full=fits['MEAN_Z']
bpz.z_peak_full=fits['MODE_Z']
bpz.pz_full=fits['Z_MC']
bpz.w=np.ones(len(bpz.coadd))
bpz.bins=200
bpz.bin=(np.linspace(0.005, 2.0, 201)[1:] + np.linspace(0.005, 2.0, 201)[0:-1])/2.0
bpz.binlow=np.linspace(0.005, 2.0, 201)[0:-1]
bpz.binhigh=np.linspace(0.005, 2.0, 201)[1:]

pzmask=np.in1d(bpz.coadd,i3.coadd,assume_unique=True)
catalog.CatalogMethods.match_cat(bpz,pzmask)

hwe=catalog.PZStore('hwe',setup=False,pztype='HWE')
hwe.pdftype='sample'
fits=fio.FITS('/share/des/disc2/y1/photo_z/PHOTOZ_HWE_Y1_v0.1.fits')[-1].read()
hwe.coadd=fits['COADD_OBJECTS_ID']
hwe.z_mean_full=fits['Z_MEAN']
hwe.pz_full=fits['Z_HWE']
hwe.w=np.ones(len(hwe.coadd))
hwe.bins=200
hwe.bin=(np.linspace(0.005, 2.0, 201)[1:] + np.linspace(0.005, 2.0, 201)[0:-1])/2.0
hwe.binlow=np.linspace(0.005, 2.0, 201)[0:-1]
hwe.binhigh=np.linspace(0.005, 2.0, 201)[1:]

pzmask=np.in1d(hwe.coadd,i3.coadd,assume_unique=True)
catalog.CatalogMethods.match_cat(hwe,pzmask)

ada=catalog.PZStore('ada',setup=False,pztype='ADA')
ada.pdftype='sample'
fits=fio.FITS('/share/des/disc2/y1/photo_z/PHOTOZ_ADA_Z_Y1_v0.3.fits')[-1].read()
ada.coadd=fits['COADD_OBJECTS_ID']
ada.z_mean_full=fits['MEAN_Z']
ada.z_peak_full=fits['MODE_Z']
ada.pz_full=fits['Z_MC']
ada.w=np.ones(len(ada.coadd))
ada.bins=200
ada.bin=(np.linspace(0.005, 2.0, 201)[1:] + np.linspace(0.005, 2.0, 201)[0:-1])/2.0
ada.binlow=np.linspace(0.005, 2.0, 201)[0:-1]
ada.binhigh=np.linspace(0.005, 2.0, 201)[1:]

pzmask=np.in1d(ada.coadd,i3.coadd,assume_unique=True)
catalog.CatalogMethods.match_cat(ada,pzmask)

rf=catalog.PZStore('rf',setup=False,pztype='RF')
rf.pdftype='sample'
fits=fio.FITS('/share/des/disc2/y1/photo_z/photo_z_Y1_gold_v0.1_cbonnett.fits')[-5].read()
rf.coadd=fits['coadd_objects_id']
fits=fio.FITS('/share/des/disc2/y1/photo_z/photo_z_Y1_gold_v0.1_cbonnett.fits')[-3].read()
rf.z_mean_full=fits['mean_z']
fits=fio.FITS('/share/des/disc2/y1/photo_z/photo_z_Y1_gold_v0.1_cbonnett.fits')[-2].read()
rf.pz_full=fits['sample_z']
rf.w=np.ones(len(rf.coadd))
rf.bins=200
rf.bin=(np.linspace(0.005, 2.0, 201)[1:] + np.linspace(0.005, 2.0, 201)[0:-1])/2.0
rf.binlow=np.linspace(0.005, 2.0, 201)[0:-1]
rf.binhigh=np.linspace(0.005, 2.0, 201)[1:]

pzmask=np.in1d(rf.coadd,i3.coadd,assume_unique=True)
catalog.CatalogMethods.match_cat(rf,pzmask)

dnf=catalog.PZStore('dnf',setup=False,pztype='DNF')
dnf.pdftype='sample'
fits=np.genfromtxt('/share/des/disc2/y1/photo_z/dnfv0.1_des_y1a1_gold_v1.0.1.csv',skip_header=1,usecols=['coadd_objects_id','z_phot','z_nz'],names=['coadd_objects_id','ra','dec','z_phot','z_phot_err','z_nz'])
dnf.coadd=fits['coadd_objects_id']
dnf.z_mean_full=fits['z_phot']
dnf.pz_full=fits['z_nz']
dnf.w=np.ones(len(dnf.coadd))
dnf.bins=200
dnf.bin=(np.linspace(0.005, 2.0, 201)[1:] + np.linspace(0.005, 2.0, 201)[0:-1])/2.0
dnf.binlow=np.linspace(0.005, 2.0, 201)[0:-1]
dnf.binhigh=np.linspace(0.005, 2.0, 201)[1:]

pzmask=np.in1d(dnf.coadd,i3.coadd,assume_unique=True)
catalog.CatalogMethods.match_cat(dnf,pzmask)

numbins=6
pz.pz_methods.build_nofz_bins(bpz,pzlow=0.0,pzhigh=2.5,cat=None,bins=numbins,split='mean',pzmask=None,catmask=None,spec=False,point=True)
pz.pz_methods.build_nofz_bins(hwe,pzlow=0.0,pzhigh=2.5,cat=None,bins=numbins,split='mean',pzmask=None,catmask=None,spec=False,point=True)
pz.pz_methods.build_nofz_bins(ada,pzlow=0.0,pzhigh=2.5,cat=None,bins=numbins,split='mean',pzmask=None,catmask=None,spec=False,point=True)
pz.pz_methods.build_nofz_bins(rf,pzlow=0.0,pzhigh=2.5,cat=None,bins=numbins,split='mean',pzmask=None,catmask=None,spec=False,point=True)
pz.pz_methods.build_nofz_bins(dnf,pzlow=0.0,pzhigh=2.5,cat=None,bins=3,split='mean',pzmask=None,catmask=None,spec=False,point=True)

fig.plot_methods.plot_nofz_comp_pz([bpz,hwe,ada,rf,dnf],spec=True)

cosmosis.make.nofz(sn,'pz_test')
cosmosis.make.nofz(sn,'pz_test')
cosmosis.make.nofz(sn,'pz_test')
cosmosis.make.nofz(sn,'pz_test')
cosmosis.make.nofz(sn,'pz_test')



self.pdftype='sample'
fits=fio.FITS(config.pzdir+file)
self.bin=fits[1].read()['redshift']
self.coadd=fits[2].read()['coadd_objects_id'].astype(int)
self.z_peak_full=fits[3].read()['mode_z']
self.z_mean_full=fits[4].read()['mean_z']
self.pz_full=fits[5].read()['sample_z']
self.binlow=self.bin-(self.bin[1]-self.bin[0])/2.
self.binhigh=self.bin+(self.bin[1]-self.bin[0])/2.
self.bins=len(self.bin)
self.w=np.ones(len(self.z_mean_full))


# cosmosis /home/troxel/cosmosis/cosmosis-des-library/photoztests/y1/tcp.ini -p runtime.sampler=test output.filename=/home/troxel/cosmosis/cosmosis-des-library/photoztests/y1/y1_v1_spec_validation_mean_pdf_3_weighted/out/spec_ada.txt test.save_dir=/home/troxel/cosmosis/cosmosis-des-library/photoztests/y1/y1_v1_spec_validation_mean_pdf_3_weighted/out/spec_ada multinest.live_points=50 multinest.tolerance=0.5 multinest.efficiency=0.8 grid.nsample_dimension=50 pipeline.modules='consistency camb sigma8_rescale halofit growth extrapolate fits_nz unbiased_galaxies pk_to_cl ggl_bias save_c_ell_fits' pipeline.values=/home/troxel/cosmosis/cosmosis-des-library/photoztests/y1/y1_v1_spec_validation_mean_pdf_3_weighted/ini/spec_ada_values.ini pipeline.priors='' pipeline.likelihoods='' pk_to_cl.shear-shear=shear-shear pk_to_cl.shear-position=shear-position pk_to_cl.position-position=position-position pk_to_cl.intrinsic-intrinsic='' pk_to_cl.shear-intrinsic='' pk_to_cl.position-intrinsic='' save_c_ell_fits.ell_min=200.0 save_c_ell_fits.ell_max=2000.0 save_c_ell_fits.n_ell=10 save_c_ell_fits.shear_nz_name=nz_shear save_c_ell_fits.position_nz_name=nz_position save_c_ell_fits.filename=/home/troxel/cosmosis/cosmosis-des-library/photoztests/y1/y1_v1_spec_validation_mean_pdf_3_weighted/out/spec_ada.fits.gz save_c_ell_fits.survey_area=1000.0 save_c_ell_fits.number_density_shear_bin='1.33333333333 1.33333333333 1.33333333333' save_c_ell_fits.number_density_lss_bin='1.16666666667 1.16666666667 1.16666666667' save_c_ell_fits.sigma_e_bin='0.25 0.25 0.25' ggl_bias.perbin=T fits_nz.nz_file=/home/troxel/cosmosis/cosmosis-des-library/photoztests/y1/y1_v1_spec_validation_mean_pdf_3_weighted/nofz/spec_ada.fits.gz fits_nz.data_sets='shear position' 2pt_like.data_file=/home/troxel/cosmosis/cosmosis-des-library/photoztests/y1/y1_v1_spec_validation_mean_pdf_3_weighted/out/spec_ada.fits.gz 2pt_like.data_sets='shear_cl shear_galaxy_cl galaxy_cl' 2pt_like.gaussian_covariance=T 2pt_like.survey_area=1000.0 2pt_like.number_density_shear_bin='1.33333333333 1.33333333333 1.33333333333' 2pt_like.number_density_lss_bin='1.16666666667 1.16666666667 1.16666666667' 2pt_like.sigma_e_bin='0.25 0.25 0.25'

# cosmosis /home/troxel/cosmosis/cosmosis-des-library/photoztests/y1/tcp.ini -p runtime.sampler=test output.filename=/home/troxel/cosmosis/cosmosis-des-library/photoztests/y1/y1_v1_spec_validation_mean_pdf_3_weighted/out/spec_ada.txt test.save_dir=/home/troxel/cosmosis/cosmosis-des-library/photoztests/y1/y1_v1_spec_validation_mean_pdf_3_weighted/out/spec_ada multinest.live_points=50 multinest.tolerance=0.5 multinest.efficiency=0.8 grid.nsample_dimension=50 pipeline.modules='consistency camb sigma8_rescale halofit growth extrapolate fits_nz unbiased_galaxies pk_to_cl ggl_bias 2pt_like save_c_ell_fits' pipeline.values=/home/troxel/cosmosis/cosmosis-des-library/photoztests/y1/y1_v1_spec_validation_mean_pdf_3_weighted/ini/spec_ada_values.ini pipeline.priors='' pipeline.likelihoods='' pk_to_cl.shear-shear=shear-shear pk_to_cl.shear-position=shear-position pk_to_cl.position-position=position-position pk_to_cl.intrinsic-intrinsic='' pk_to_cl.shear-intrinsic='' pk_to_cl.position-intrinsic='' save_c_ell_fits.ell_min=200.0 save_c_ell_fits.ell_max=2000.0 save_c_ell_fits.n_ell=10 save_c_ell_fits.shear_nz_name=nz_shear save_c_ell_fits.position_nz_name=nz_position save_c_ell_fits.filename=/home/troxel/cosmosis/cosmosis-des-library/photoztests/y1/y1_v1_spec_validation_mean_pdf_3_weighted/out/spec_ada.fits.gz save_c_ell_fits.survey_area=1000.0 save_c_ell_fits.number_density_shear_bin='1.33333333333 1.33333333333 1.33333333333' save_c_ell_fits.number_density_lss_bin='1.16666666667 1.16666666667 1.16666666667' save_c_ell_fits.sigma_e_bin='0.25 0.25 0.25' ggl_bias.perbin=T fits_nz.nz_file=/home/troxel/cosmosis/cosmosis-des-library/photoztests/y1/y1_v1_spec_validation_mean_pdf_3_weighted/nofz/spec_ada.fits.gz fits_nz.data_sets='shear position' 2pt_like.data_file=/home/troxel/cosmosis/cosmosis-des-library/photoztests/y1/y1_v1_spec_validation_mean_pdf_3_weighted/out/spec_ada.fits.gz 2pt_like.data_sets='shear_cl shear_galaxy_cl galaxy_cl' 2pt_like.gaussian_covariance=T 2pt_like.survey_area=1000.0 2pt_like.number_density_shear_bin='1.33333333333 1.33333333333 1.33333333333' 2pt_like.number_density_lss_bin='1.16666666667 1.16666666667 1.16666666667' 2pt_like.sigma_e_bin='0.25 0.25 0.25'

# mpirun -n 5 cosmosis --mpi /home/troxel/cosmosis/cosmosis-des-library/photoztests/y1/tcp.ini -p runtime.sampler=grid output.filename=/home/troxel/cosmosis/cosmosis-des-library/photoztests/y1/y1_v1_spec_validation_mean_pdf_3_weighted/out/wl_spec_ada_ada.txt test.save_dir='' multinest.live_points=50 multinest.tolerance=0.5 multinest.efficiency=0.8 grid.nsample_dimension=50 pipeline.modules='consistency camb sigma8_rescale halofit growth extrapolate fits_nz unbiased_galaxies pk_to_cl 2pt_like' pipeline.values=/home/troxel/cosmosis/cosmosis-des-library/photoztests/y1/y1_v1_spec_validation_mean_pdf_3_weighted/ini/spec_ada_ada_values.ini pipeline.priors='' pipeline.likelihoods=2pt pk_to_cl.shear-shear=shear-shear pk_to_cl.shear-position='' pk_to_cl.position-position='' pk_to_cl.intrinsic-intrinsic='' pk_to_cl.shear-intrinsic='' pk_to_cl.position-intrinsic='' save_c_ell_fits.ell_min=200.0 save_c_ell_fits.ell_max=2000.0 save_c_ell_fits.n_ell=10 save_c_ell_fits.shear_nz_name=nz_shear save_c_ell_fits.position_nz_name=nz_position save_c_ell_fits.filename=/home/troxel/cosmosis/cosmosis-des-library/photoztests/y1/y1_v1_spec_validation_mean_pdf_3_weighted/out/spec_ada_ada.fits.gz save_c_ell_fits.survey_area=1000.0 save_c_ell_fits.number_density_shear_bin='1.33333333333 1.33333333333 1.33333333333' save_c_ell_fits.number_density_lss_bin='1.16666666667 1.16666666667 1.16666666667' save_c_ell_fits.sigma_e_bin='0.25 0.25 0.25' ggl_bias.perbin=T fits_nz.nz_file=/home/troxel/cosmosis/cosmosis-des-library/photoztests/y1/y1_v1_spec_validation_mean_pdf_3_weighted/nofz/spec_ada.fits.gz fits_nz.data_sets='shear' 2pt_like.data_file=/home/troxel/cosmosis/cosmosis-des-library/photoztests/y1/y1_v1_spec_validation_mean_pdf_3_weighted/out/spec_ada.fits.gz 2pt_like.data_sets='shear_cl' 2pt_like.gaussian_covariance=F 2pt_like.survey_area=1000.0 2pt_like.number_density_shear_bin='1.33333333333 1.33333333333 1.33333333333' 2pt_like.number_density_lss_bin='1.16666666667 1.16666666667 1.16666666667' 2pt_like.sigma_e_bin='0.25 0.25 0.25'

# mpirun -n 1 postprocess /home/troxel/cosmosis/cosmosis-des-library/photoztests/y1/y1_v1_spec_validation_mean_pdf_3_weighted/out/lss_spec_ada_ada.txt -o /home/troxel/cosmosis/cosmosis-des-library/photoztests/y1/y1_v1_spec_validation_mean_pdf_3_weighted/out -p lss_spec_ada_ada 


# cosmosis /home/troxel/cosmosis/cosmosis-des-library/photoztests/y1/tcp.ini -p runtime.sampler=test output.filename=/home/troxel/cosmosis/cosmosis-des-library/photoztests/y1/y1_v1_spec_validation_mean_pdf_3_weighted/out/ada.txt test.save_dir=/home/troxel/cosmosis/cosmosis-des-library/photoztests/y1/y1_v1_spec_validation_mean_pdf_3_weighted/out/ada multinest.live_points=200 multinest.tolerance=0.5 multinest.efficiency=0.8 grid.nsample_dimension=50 pipeline.modules='consistency camb sigma8_rescale halofit growth extrapolate fits_nz unbiased_galaxies pk_to_cl ggl_bias 2pt_like save_c_ell_fits' pipeline.values=/home/troxel/cosmosis/cosmosis-des-library/photoztests/y1/y1_v1_spec_validation_mean_pdf_3_weighted/ini/ada_values.ini pipeline.priors='' pipeline.likelihoods='' pk_to_cl.shear-shear=shear-shear pk_to_cl.shear-position=shear-position pk_to_cl.position-position=position-position pk_to_cl.intrinsic-intrinsic='' pk_to_cl.shear-intrinsic='' pk_to_cl.position-intrinsic='' save_c_ell_fits.ell_min=200.0 save_c_ell_fits.ell_max=2000.0 save_c_ell_fits.n_ell=10 save_c_ell_fits.shear_nz_name=nz_shear save_c_ell_fits.position_nz_name=nz_position save_c_ell_fits.filename=/home/troxel/cosmosis/cosmosis-des-library/photoztests/y1/y1_v1_spec_validation_mean_pdf_3_weighted/out/ada.fits.gz save_c_ell_fits.survey_area=1000.0 save_c_ell_fits.number_density_shear_bin='1.33333333333 1.33333333333 1.33333333333' save_c_ell_fits.number_density_lss_bin='1.16666666667 1.16666666667 1.16666666667' save_c_ell_fits.sigma_e_bin='0.25 0.25 0.25' ggl_bias.perbin=T fits_nz.nz_file=/home/troxel/cosmosis/cosmosis-des-library/photoztests/y1/y1_v1_spec_validation_mean_pdf_3_weighted/nofz/ada.fits.gz fits_nz.data_sets='shear position' 2pt_like.data_file=/home/troxel/cosmosis/cosmosis-des-library/photoztests/y1/y1_v1_spec_validation_mean_pdf_3_weighted/out/ada.fits.gz 2pt_like.data_sets='shear_cl shear_galaxy_cl galaxy_cl' 2pt_like.gaussian_covariance=T 2pt_like.survey_area=1000.0 2pt_like.number_density_shear_bin='1.33333333333 1.33333333333 1.33333333333' 2pt_like.number_density_lss_bin='1.16666666667 1.16666666667 1.16666666667' 2pt_like.sigma_e_bin='0.25 0.25 0.25'

# cosmosis /home/troxel/cosmosis/cosmosis-des-library/photoztests/y1/tcp.ini -p runtime.sampler=test output.filename=/home/troxel/cosmosis/cosmosis-des-library/photoztests/y1/y1_v1_spec_validation_mean_pdf_3_weighted/out/ada.txt test.save_dir=/home/troxel/cosmosis/cosmosis-des-library/photoztests/y1/y1_v1_spec_validation_mean_pdf_3_weighted/out/ada multinest.live_points=200 multinest.tolerance=0.5 multinest.efficiency=0.8 grid.nsample_dimension=50 pipeline.modules='consistency camb sigma8_rescale halofit growth extrapolate fits_nz unbiased_galaxies pk_to_cl ggl_bias 2pt_like save_c_ell_fits' pipeline.values=/home/troxel/cosmosis/cosmosis-des-library/photoztests/y1/y1_v1_spec_validation_mean_pdf_3_weighted/ini/ada_values.ini pipeline.priors='' pipeline.likelihoods='' pk_to_cl.shear-shear=shear-shear pk_to_cl.shear-position=shear-position pk_to_cl.position-position=position-position pk_to_cl.intrinsic-intrinsic='' pk_to_cl.shear-intrinsic='' pk_to_cl.position-intrinsic='' save_c_ell_fits.ell_min=200.0 save_c_ell_fits.ell_max=2000.0 save_c_ell_fits.n_ell=10 save_c_ell_fits.shear_nz_name=nz_shear save_c_ell_fits.position_nz_name=nz_position save_c_ell_fits.filename=/home/troxel/cosmosis/cosmosis-des-library/photoztests/y1/y1_v1_spec_validation_mean_pdf_3_weighted/out/ada.fits.gz save_c_ell_fits.survey_area=1000.0 save_c_ell_fits.number_density_shear_bin='1.33333333333 1.33333333333 1.33333333333' save_c_ell_fits.number_density_lss_bin='1.16666666667 1.16666666667 1.16666666667' save_c_ell_fits.sigma_e_bin='0.25 0.25 0.25' ggl_bias.perbin=T fits_nz.nz_file=/home/troxel/cosmosis/cosmosis-des-library/photoztests/y1/y1_v1_spec_validation_mean_pdf_3_weighted/nofz/ada.fits.gz fits_nz.data_sets='shear position' 2pt_like.data_file=/home/troxel/cosmosis/cosmosis-des-library/photoztests/y1/y1_v1_spec_validation_mean_pdf_3_weighted/out/ada.fits.gz 2pt_like.data_sets='shear_cl shear_galaxy_cl galaxy_cl' 2pt_like.gaussian_covariance=T 2pt_like.survey_area=1000.0 2pt_like.number_density_shear_bin='1.33333333333 1.33333333333 1.33333333333' 2pt_like.number_density_lss_bin='1.16666666667 1.16666666667 1.16666666667' 2pt_like.sigma_e_bin='0.25 0.25 0.25'

# cosmosis /home/troxel/cosmosis/cosmosis-des-library/photoztests/y1/tcp.ini -p runtime.sampler=test output.filename=/home/troxel/cosmosis/cosmosis-des-library/photoztests/y1/y1_v1_spec_validation_mean_pdf_3_weighted/out/spec_ada.txt test.save_dir=/home/troxel/cosmosis/cosmosis-des-library/photoztests/y1/y1_v1_spec_validation_mean_pdf_3_weighted/out/spec_ada multinest.live_points=200 multinest.tolerance=0.5 multinest.efficiency=0.8 grid.nsample_dimension=50 pipeline.modules='consistency camb sigma8_rescale halofit growth extrapolate fits_nz unbiased_galaxies pk_to_cl ggl_bias 2pt_like save_c_ell_fits' pipeline.values=/home/troxel/cosmosis/cosmosis-des-library/photoztests/y1/y1_v1_spec_validation_mean_pdf_3_weighted/ini/spec_ada_values.ini pipeline.priors='' pipeline.likelihoods='' pk_to_cl.shear-shear=shear-shear pk_to_cl.shear-position=shear-position pk_to_cl.position-position=position-position pk_to_cl.intrinsic-intrinsic='' pk_to_cl.shear-intrinsic='' pk_to_cl.position-intrinsic='' save_c_ell_fits.ell_min=200.0 save_c_ell_fits.ell_max=2000.0 save_c_ell_fits.n_ell=10 save_c_ell_fits.shear_nz_name=nz_shear save_c_ell_fits.position_nz_name=nz_position save_c_ell_fits.filename=/home/troxel/cosmosis/cosmosis-des-library/photoztests/y1/y1_v1_spec_validation_mean_pdf_3_weighted/out/spec_ada.fits.gz save_c_ell_fits.survey_area=1000.0 save_c_ell_fits.number_density_shear_bin='1.33333333333 1.33333333333 1.33333333333' save_c_ell_fits.number_density_lss_bin='1.16666666667 1.16666666667 1.16666666667' save_c_ell_fits.sigma_e_bin='0.25 0.25 0.25' ggl_bias.perbin=T fits_nz.nz_file=/home/troxel/cosmosis/cosmosis-des-library/photoztests/y1/y1_v1_spec_validation_mean_pdf_3_weighted/nofz/spec_ada.fits.gz fits_nz.data_sets='shear position' 2pt_like.data_file=/home/troxel/cosmosis/cosmosis-des-library/photoztests/y1/y1_v1_spec_validation_mean_pdf_3_weighted/out/spec_ada.fits.gz 2pt_like.data_sets='shear_cl shear_galaxy_cl galaxy_cl' 2pt_like.gaussian_covariance=T 2pt_like.survey_area=1000.0 2pt_like.number_density_shear_bin='1.33333333333 1.33333333333 1.33333333333' 2pt_like.number_density_lss_bin='1.16666666667 1.16666666667 1.16666666667' 2pt_like.sigma_e_bin='0.25 0.25 0.25'

# mpirun -n 4 cosmosis --mpi /home/troxel/cosmosis/cosmosis-des-library/photoztests/y1/tcp.ini -p runtime.sampler=multinest output.filename=/home/troxel/cosmosis/cosmosis-des-library/photoztests/y1/y1_v1_spec_validation_mean_pdf_3_weighted/out/lss_spec_ada_ada.txt test.save_dir='' multinest.live_points=200 multinest.tolerance=0.5 multinest.efficiency=0.8 grid.nsample_dimension=50 pipeline.modules='consistency camb sigma8_rescale halofit growth extrapolate fits_nz unbiased_galaxies pk_to_cl ggl_bias 2pt_like' pipeline.values=/home/troxel/cosmosis/cosmosis-des-library/photoztests/y1/y1_v1_spec_validation_mean_pdf_3_weighted/ini/spec_ada_ada_values.ini pipeline.priors='' pipeline.likelihoods=2pt pk_to_cl.shear-shear='' pk_to_cl.shear-position='' pk_to_cl.position-position=position-position pk_to_cl.intrinsic-intrinsic='' pk_to_cl.shear-intrinsic='' pk_to_cl.position-intrinsic='' save_c_ell_fits.ell_min=200.0 save_c_ell_fits.ell_max=2000.0 save_c_ell_fits.n_ell=10 save_c_ell_fits.shear_nz_name=nz_shear save_c_ell_fits.position_nz_name=nz_position save_c_ell_fits.filename=/home/troxel/cosmosis/cosmosis-des-library/photoztests/y1/y1_v1_spec_validation_mean_pdf_3_weighted/out/spec_ada_ada.fits.gz save_c_ell_fits.survey_area=1000.0 save_c_ell_fits.number_density_shear_bin='1.33333333333 1.33333333333 1.33333333333' save_c_ell_fits.number_density_lss_bin='1.16666666667 1.16666666667 1.16666666667' save_c_ell_fits.sigma_e_bin='0.25 0.25 0.25' ggl_bias.perbin=T fits_nz.nz_file=/home/troxel/cosmosis/cosmosis-des-library/photoztests/y1/y1_v1_spec_validation_mean_pdf_3_weighted/nofz/spec_ada.fits.gz fits_nz.data_sets='position' 2pt_like.data_file=/home/troxel/cosmosis/cosmosis-des-library/photoztests/y1/y1_v1_spec_validation_mean_pdf_3_weighted/out/spec_ada.fits.gz 2pt_like.data_sets='galaxy_cl' 2pt_like.gaussian_covariance=F 2pt_like.survey_area=1000.0 2pt_like.number_density_shear_bin='1.33333333333 1.33333333333 1.33333333333' 2pt_like.number_density_lss_bin='1.16666666667 1.16666666667 1.16666666667' 2pt_like.sigma_e_bin='0.25 0.25 0.25'
# mpirun -n 1 postprocess /home/troxel/cosmosis/cosmosis-des-library/photoztests/y1/y1_v1_spec_validation_mean_pdf_3_weighted/out/lss_spec_ada_ada.txt -o /home/troxel/cosmosis/cosmosis-des-library/photoztests/y1/y1_v1_spec_validation_mean_pdf_3_weighted/out -p lss_spec_ada_ada --no-plots


ada=catalog.PZStore('ada',setup=True,pztype='ada',filetype='h5',file=config.pzdir+'Y1A1_GOLD101_Y1A1trainValid_14.12.2015.validsY1A1.25215.out.DES.pdf.hdf5')
hwe=catalog.PZStore('hwe',setup=True,pztype='hwe',filetype='h5',file=config.pzdir+'PDF_MMRAU_standard_output.hdf5')
bpz=catalog.PZStore('bpz',setup=True,pztype='bpz',filetype='h5',file=config.pzdir+'BPZ_v1_probs_DEC15_trainvalid.hdf5')
dnf=catalog.PZStore('dnf',setup=True,pztype='dnf',filetype='h5',file=config.pzdir+'dnf_validation.hdf5')
annz=catalog.PZStore('dnf',setup=True,pztype='dnf',filetype='h5',file=config.pzdir+'dnf_validation.hdf5')
names=['COADD_OBJECTS_ID','Z','KNN_w','inTrainFlag','ANNZ_best','ANNZ_best_wgt','ANNZ_best_err','ANNZ_MLM_avg_0','ANNZ_MLM_avg_0_err','ANNZ_MLM_avg_0_wgt','ANNZ_PDF_avg_0','ANNZ_PDF_avg_0_err','ANNZ_PDF_avg_0_wgt','ANNZ_PDF_0_0','ANNZ_PDF_0_1','ANNZ_PDF_0_2','ANNZ_PDF_0_3','ANNZ_PDF_0_4','ANNZ_PDF_0_5','ANNZ_PDF_0_6','ANNZ_PDF_0_7','ANNZ_PDF_0_8','ANNZ_PDF_0_9','ANNZ_PDF_0_10','ANNZ_PDF_0_11','ANNZ_PDF_0_12','ANNZ_PDF_0_13','ANNZ_PDF_0_14','ANNZ_PDF_0_15','ANNZ_PDF_0_16','ANNZ_PDF_0_17','ANNZ_PDF_0_18','ANNZ_PDF_0_19','ANNZ_PDF_0_20','ANNZ_PDF_0_21','ANNZ_PDF_0_22','ANNZ_PDF_0_23','ANNZ_PDF_0_24','ANNZ_PDF_0_25','ANNZ_PDF_0_26','ANNZ_PDF_0_27','ANNZ_PDF_0_28','ANNZ_PDF_0_29','ANNZ_PDF_0_30','ANNZ_PDF_0_31','ANNZ_PDF_0_32','ANNZ_PDF_0_33','ANNZ_PDF_0_34','ANNZ_PDF_0_35','ANNZ_PDF_0_36','ANNZ_PDF_0_37','ANNZ_PDF_0_38','ANNZ_PDF_0_39','ANNZ_PDF_0_40','ANNZ_PDF_0_41','ANNZ_PDF_0_42','ANNZ_PDF_0_43','ANNZ_PDF_0_44','ANNZ_PDF_0_45','ANNZ_PDF_0_46','ANNZ_PDF_0_47','ANNZ_PDF_0_48','ANNZ_PDF_0_49','ANNZ_PDF_0_50','ANNZ_PDF_0_51','ANNZ_PDF_0_52','ANNZ_PDF_0_53','ANNZ_PDF_0_54','ANNZ_PDF_0_55','ANNZ_PDF_0_56','ANNZ_PDF_0_57','ANNZ_PDF_0_58','ANNZ_PDF_0_59','ANNZ_PDF_0_60','ANNZ_PDF_0_61','ANNZ_PDF_0_62','ANNZ_PDF_0_63','ANNZ_PDF_0_64','ANNZ_PDF_0_65','ANNZ_PDF_0_66','ANNZ_PDF_0_67','ANNZ_PDF_0_68','ANNZ_PDF_0_69','ANNZ_PDF_0_70','ANNZ_PDF_0_71','ANNZ_PDF_0_72','ANNZ_PDF_0_73','ANNZ_PDF_0_74','ANNZ_PDF_0_75','ANNZ_PDF_0_76','ANNZ_PDF_0_77','ANNZ_PDF_0_78','ANNZ_PDF_0_79','ANNZ_PDF_0_80','ANNZ_PDF_0_81','ANNZ_PDF_0_82','ANNZ_PDF_0_83','ANNZ_PDF_0_84','ANNZ_PDF_0_85','ANNZ_PDF_0_86','ANNZ_PDF_0_87','ANNZ_PDF_0_88','ANNZ_PDF_0_89','ANNZ_PDF_0_90','ANNZ_PDF_0_91','ANNZ_PDF_0_92','ANNZ_PDF_0_93','ANNZ_PDF_0_94','ANNZ_PDF_0_95','ANNZ_PDF_0_96','ANNZ_PDF_0_97','ANNZ_PDF_0_98','ANNZ_PDF_0_99','ANNZ_PDF_0_100','ANNZ_PDF_0_101','ANNZ_PDF_0_102','ANNZ_PDF_0_103','ANNZ_PDF_0_104','ANNZ_PDF_0_105','ANNZ_PDF_0_106','ANNZ_PDF_0_107','ANNZ_PDF_0_108','ANNZ_PDF_0_109','ANNZ_PDF_0_110','ANNZ_PDF_0_111','ANNZ_PDF_0_112','ANNZ_PDF_0_113','ANNZ_PDF_0_114','ANNZ_PDF_0_115','ANNZ_PDF_0_116','ANNZ_PDF_0_117','ANNZ_PDF_0_118','ANNZ_PDF_0_119','ANNZ_PDF_0_120','ANNZ_PDF_0_121','ANNZ_PDF_0_122','ANNZ_PDF_0_123','ANNZ_PDF_0_124','ANNZ_PDF_0_125','ANNZ_PDF_0_126','ANNZ_PDF_0_127','ANNZ_PDF_0_128','ANNZ_PDF_0_129','ANNZ_PDF_0_130','ANNZ_PDF_0_131','ANNZ_PDF_0_132','ANNZ_PDF_0_133','ANNZ_PDF_0_134','ANNZ_PDF_0_135','ANNZ_PDF_0_136','ANNZ_PDF_0_137','ANNZ_PDF_0_138','ANNZ_PDF_0_139','ANNZ_PDF_0_140','ANNZ_PDF_0_141','ANNZ_PDF_0_142','ANNZ_PDF_0_143','ANNZ_PDF_0_144','ANNZ_PDF_0_145','ANNZ_PDF_0_146','ANNZ_PDF_0_147','ANNZ_PDF_0_148','ANNZ_PDF_0_149']
tmp=np.genfromtxt('/share/des/disc2/y1/photo_z/annz2_y1_25_01_16_valid_wl.csv',delimiter=',')
annz.name='annz'
annz.pztype='annz'
annz.z_mean_full=tmp[:,10]
annz.coadd=tmp[:,0]
annz.pz_full=tmp[:,-150:]
annz.spec_full=tmp[:,1]
annz.bins=150
annz.bin=(np.linspace(0,1.50,151)[:-1]+np.linspace(0,1.50,151)[1:])/2.
annz.binlow=np.linspace(0,1.50,151)[:-1]
annz.binhigh=np.linspace(0,1.50,151)[1:]
annz.w=tmp[:,2]


wts=np.load(config.pzdir+'gold_weights_v2.npy')
for bin in [3,6]:
  for x,pz0 in enumerate([ada,hwe,bpz,dnf]):
    pz0.wt=True
    x,y=catalog.CatalogMethods.sort2(pz0.coadd,wts[:,0])
    pz0.w=np.zeros(len(pz0.coadd))
    pz0.w[x]=wts[y,1]
    pz.pz_methods.build_nofz_bins(pz0,label='source',pzlow=0.,pzhigh=3.,cat=None,bins=bin,split='mean',nztype='pdf',pzmask=None,catmask=None,spec=True)
    pz0.w[x]=wts[y,2]
    pz.pz_methods.build_nofz_bins(pz0,label='lens',pzlow=0.,pzhigh=3.,cat=None,bins=bin,split='mean',nztype='pdf',pzmask=None,catmask=None,spec=True)
    cosmo.make.nofz(pz0,'y1_v1_spec_validation_mean_pdf')
  for x,pz0 in enumerate([annz]):
    pz0.wt=True 
    tmp=np.genfromtxt('/share/des/disc2/y1/photo_z/annz2_y1_25_01_16_valid_wl.csv',delimiter=',')
    annz.name='annz'
    annz.pztype='annz'
    annz.z_mean_full=tmp[:,10]
    annz.coadd=tmp[:,0]
    annz.pz_full=tmp[:,-150:]
    annz.spec_full=tmp[:,1]
    annz.bins=150
    annz.bin=(np.linspace(0,1.50,151)[:-1]+np.linspace(0,1.50,151)[1:])/2.
    annz.binlow=np.linspace(0,1.50,151)[:-1]
    annz.binhigh=np.linspace(0,1.50,151)[1:]
    annz.w=tmp[:,2]
    x,y=catalog.CatalogMethods.sort2(pz0.coadd,wts[:,0])
    pz0.w=np.zeros(len(pz0.coadd))
    pz0.w[x]=wts[y,1]
    pz.pz_methods.build_nofz_bins(pz0,label='source',pzlow=0.,pzhigh=3.,cat=None,bins=bin,split='mean',nztype='pdf',pzmask=None,catmask=None,spec=True)
    tmp=np.genfromtxt('/share/des/disc2/y1/photo_z/annz2_y1_25_01_16_valid_lss.csv',delimiter=',')
    annz.name='annz'
    annz.pztype='annz'
    annz.z_mean_full=tmp[:,10]
    annz.coadd=tmp[:,0]
    annz.pz_full=tmp[:,-150:]
    annz.spec_full=tmp[:,1]
    annz.bins=150
    annz.bin=(np.linspace(0,1.50,151)[:-1]+np.linspace(0,1.50,151)[1:])/2.
    annz.binlow=np.linspace(0,1.50,151)[:-1]
    annz.binhigh=np.linspace(0,1.50,151)[1:]
    annz.w=tmp[:,2]
    pz0.w[x]=wts[y,2]
    pz.pz_methods.build_nofz_bins(pz0,label='lens',pzlow=0.,pzhigh=3.,cat=None,bins=bin,split='mean',nztype='pdf',pzmask=None,catmask=None,spec=True)
    cosmo.make.nofz(pz0,'y1_v1_spec_validation_mean_pdf')
  fig.plot_methods.plot_nofz_comp_pz([ada,annz,bpz,dnf,hwe],pztypes=['source','lens'],label='',spec=True,notomo=False)


for x,pz0 in enumerate([ada,annz,hwe,bpz,dnf]):
  for bin in [3,6]:
    for wtd in ['','_weighted']:
      cosmo.run.submit_pz_spec_test(pz0,'y1_v1_spec_validation_mean_pdf_'+str(bin)+wtd,'tcp',bins=bin,boot=False,cosmo=False,submit=True,fillin=False)

cosmosis.run.submit_pz_spec_test(dnf,'y1_v1_spec_validation_mean_pdf_'+str(3)+'_weighted','wl',bins=3,boot=False,cosmo=True,submit=True)



fig.plot_methods.spec_loop_hist2d([ada,hwe,bpz])
pz.pz_spec_validation.spec_comp([ada,hwe,bpz])

pz.pz_spec_validation.sig_crit_spec([ada,annz,hwe,bpz,dnf],bins=5,pzlow=0.2,pzhigh=1.3,load=False,point='mean',lensbins=[.1,1.,20])

fig.plot_methods.plot_pz_corr('y1_v1_spec_validation_mean_pdf_3',[ada,annz,bpz,dnf,hwe],label='',boot=False,ylim=0.5)
fig.plot_methods.plot_pz_corr('y1_v1_spec_validation_mean_pdf_3_weighted',[ada,annz,bpz,dnf,hwe],label='',boot=False,ylim=0.5)
fig.plot_methods.plot_pz_corr('y1_v1_spec_validation_mean_pdf_6',[ada,annz,bpz,dnf,hwe],label='',boot=False,ylim=0.5)
fig.plot_methods.plot_pz_corr('y1_v1_spec_validation_mean_pdf_6_weighted',[ada,annz,bpz,dnf,hwe],label='',boot=False,ylim=0.5)

fig.plot_methods.plot_pz_sig8('y1_v1_spec_validation_6bins',[ada2,dnf2],label='',boot=True,ylim=0.3)


ada=catalog.PZStore('ADA',setup=True,pztype='ADA',filetype='dict',file='ADA_Z_3_bins.pickle')
dnf=catalog.PZStore('DNF',setup=True,pztype='DNF',filetype='dict',file='DNF_Z_3_bins.pickle')
ada2=catalog.PZStore('ADA',setup=True,pztype='ADA',filetype='dict',file='ADA_Z_6_bins.pickle')
dnf2=catalog.PZStore('DNF',setup=True,pztype='DNF',filetype='dict',file='DNF_Z_6_bins.pickle')


for i in range(3):
  for j in range(3):
    np.unique(np.loadtxt('/home/troxel/cosmosis/cosmosis-des-library/photoztests/y1/y1_v1_spec_validation_mean_pdf_3_weighted/out/spec_ada/galaxy_cl/bin_'+str(i+1)+'_'+str(j+1)+'.txt')/np.loadtxt('/home/troxel/cosmosis/cosmosis-des-library/photoztests/y1/y1_v1_spec_validation_mean_pdf_3_weighted/out/spec_ada2/galaxy_cl/bin_'+str(i+1)+'_'+str(j+1)+'.txt'))

for ifile,file in enumerate(glob.glob('/home/troxel/cosmosis/cosmosis-des-library/photoztests/y1/y1_v1_spec_validation_mc_3/nofz/*')):
  tmp=np.loadtxt(file)
  if (tmp[0,0]==0)&(tmp[1,0]==0):
    tmp=tmp[1:,:]
  np.savetxt(file,tmp)

for i,pz0 in enumerate([ada,bpz,dnf,hwe]):
  pz0.wt=False
  pz.pz_methods.build_nofz_bins(pz0,pzlow=0.2,pzhigh=1.3,cat=None,bins=3,split='spec',nztype='pdf',pzmask=None,catmask=None,spec=True)


for i,pz0 in enumerate([ada,bpz,dnf,hwe]):
  for binning in [3,6]:
    for wt in [True,False]:
      for ztype in ['mean','peak']:
        for stack in ['pdf','mc']:
          pz0.wt=wt
          pz.pz_methods.build_nofz_bins(pz0,pzlow=0.2,pzhigh=1.3,cat=None,bins=binning,split=ztype,nztype=stack,pzmask=None,catmask=None,spec=True,boot=200)
          cosmosis.make.nofz(pz0,'y1_v1_spec_validation_'+ztype+'_'+stack)
        
for i,pz0 in enumerate([ada,bpz,dnf,hwe]):
  for binning in [3,6]:
    for wt in ['','_weighted']:
      for ztype in ['mean','peak']:
        for stack in ['pdf','mc']:
          cosmosis.run.submit_pz_spec_test(pz0,'y1_v1_spec_validation_'+ztype+'_'+stack+'_'+str(binning)+wt,boot=False,cosmo=True,submit=True)

for binning in [3,6]:
  for wt in ['','_weighted']:
    for ztype in ['mean','peak']:
      for stack in ['pdf','mc']:
        fig.plot_methods.plot_pz_sig8('y1_v1_spec_validation_'+ztype+'_'+stack+'_'+str(binning)+wt,[ada,bpz,dnf,hwe],label='',boot=False,ylim=0.6,noparam=False)




from popen2 import popen2
import subprocess as sp
import time

cnt=0
for i,pz0 in enumerate([bpz,dnf]):
  for binning in [3,6]:
    for wt in [True,False]:
      for ztype in ['mean','peak']:
        for stack in ['pdf','mc']:
          p = sp.Popen('qsub', shell=True, bufsize=1, stdin=sp.PIPE, stdout=sp.PIPE, close_fds=True, cwd='/home/troxel/destest/')
          output,input = p.stdout, p.stdin
          job_string = """#!/bin/bash
          #PBS -l nodes=1:ppn=1
          #PBS -l walltime=24:00:00
          #PBS -N %s
          #PBS -o %s.log
          #PBS -j oe
          #PBS -m abe 
          #PBS -M michael.troxel@manchester.ac.uk
          module use /home/zuntz/modules/module-files
          module load python
          module use /etc/modulefiles/
          cd /home/troxel/destest/
          cd $PBS_O_WORKDIR
          python testsuite.py 9 %s
          """ % (ztype+stack+str(i)+str(binning)+str(wt),ztype+stack+str(i)+str(binning)+str(wt),cnt)    
          output,outputerr=p.communicate(input=job_string)
          cnt+=1


pz.pz_spec_validation.spec_comp([ada,hwe,bpz,dnf])

for i,pz0 in enumerate([ada,bpz,dnf,hwe]):
  for binning in [3,6]:
    for wt in [True,False]:
      for ztype in ['mean','peak']:
        for stack in ['pdf','mc']:
          pz0.wt=wt
          pz.pz_methods.build_nofz_bins(pz0,pzlow=0.2,pzhigh=1.3,cat=None,bins=binning,split=ztype,nztype=stack,pzmask=None,catmask=None,spec=True,boot=200)
          cosmosis.make.nofz(pz0,'y1_v1_spec_validation_'+ztype+'_'+stack)

hwe=catalog.PZStore('hwe',setup=True,pztype='hwe',filetype='h5',file=config.pzdir+'PDF_MMRAU_standard_output.hdf5')
hwe1=catalog.PZStore('hwe1',setup=True,pztype='hwe',filetype='h5',file=config.pzdir+'PDF_MMRAU_standard_output.hdf5')
hwe2=catalog.PZStore('hwe2',setup=True,pztype='hwe',filetype='h5',file=config.pzdir+'PDF_MMRAU_standard_output.hdf5')
hwe3=catalog.PZStore('hwe3',setup=True,pztype='hwe',filetype='h5',file=config.pzdir+'PDF_MMRAU_standard_output.hdf5')
hwe4=catalog.PZStore('hwe4',setup=True,pztype='hwe',filetype='h5',file=config.pzdir+'PDF_MMRAU_standard_output.hdf5')
for i,pz0 in enumerate([hwe,hwe1,hwe2,hwe3,hwe4]):
  for binning in [3]:
      for ztype in ['peak']:
        for stack in ['pdf']:
          pz.pz_methods.build_nofz_bins(pz0,pzlow=0.2,pzhigh=1.3,cat=None,bins=binning,split=ztype,nztype=stack,pzmask=None,catmask=None,spec=True)

pz.pz_methods.pdf_mvsk(hwe.bin,hwe.bin[0,:],dm,dv,ds,dk)

def p_mvsk(x,m,v,s,k,pdf=None,p6=False):
  if pdf is None:
    pdf=stats.norm.pdf(x,loc=m,scale=v)
  if p6:
    return hermite.hermval(x,[1.,0.,0.,s/6.,k/24.,0.,s**2./72.])*pdf
  else:
    return hermite.hermval(x,[1.,0.,0.,s/6.,k/24.])*pdf

params=curve_fit(p_mvsk,hwe.bin,hwe.pz[0,:],p0=(m,v,0.,0.))

plt.plot(hwe.bin,hwe.pz[0,:]/np.sum(hwe.pz[0,:]),label='original')
tmp=pz.pz_methods.pdf_mvsk(hwe.bin,hwe.pz[0,:],-.1,1,0,0)
plt.plot(hwe.bin,tmp/np.sum(tmp),label='m=-0.1')
tmp=pz.pz_methods.pdf_mvsk(hwe.bin,hwe.pz[0,:],.1,1,0,0)
plt.plot(hwe.bin,tmp/np.sum(tmp),label='m=+0.1')
plt.xlim((0,2.))
plt.xlabel('Redshift')
plt.legend(loc='upper right',ncol=1, frameon=True,prop={'size':12})
plt.savefig('pz_bias_test_m.png')
plt.close()

plt.plot(hwe.bin,hwe.pz[0,:]/np.sum(hwe.pz[0,:]),label='original')
tmp=pz.pz_methods.pdf_mvsk(hwe.bin,hwe.pz[0,:],0,.9,0,0)
plt.plot(hwe.bin,tmp/np.sum(tmp),label='v=0.9')
tmp=pz.pz_methods.pdf_mvsk(hwe.bin,hwe.pz[0,:],0,1.1,0,0)
plt.plot(hwe.bin,tmp/np.sum(tmp),label='v=1.1')
plt.xlim((0,2.))
plt.xlabel('Redshift')
plt.legend(loc='upper right',ncol=1, frameon=True,prop={'size':12})
plt.savefig('pz_bias_test_v.png')
plt.close()

plt.plot(hwe.bin,hwe.pz[0,:]/np.sum(hwe.pz[0,:]),label='original')
tmp=pz.pz_methods.pdf_mvsk(hwe.bin,hwe.pz[0,:],0,1,.2,0)
plt.plot(hwe.bin,tmp/np.sum(tmp),label='s=+0.2')
tmp=pz.pz_methods.pdf_mvsk(hwe.bin,hwe.pz[0,:],0,1,-.2,0)
plt.plot(hwe.bin,tmp/np.sum(tmp),label='s=-0.2')
plt.xlim((0,2.))
plt.xlabel('Redshift')
plt.legend(loc='upper right',ncol=1, frameon=True,prop={'size':12})
plt.savefig('pz_bias_test_s.png')
plt.close()

plt.plot(hwe.bin,hwe.pz[0,:]/np.sum(hwe.pz[0,:]),label='original')
tmp=pz.pz_methods.pdf_mvsk(hwe.bin,hwe.pz[0,:],0,1,0,.5)
plt.plot(hwe.bin,tmp/np.sum(tmp),label='k=+0.5')
tmp=pz.pz_methods.pdf_mvsk(hwe.bin,hwe.pz[0,:],0,1,0,-.5)
plt.plot(hwe.bin,tmp/np.sum(tmp),label='k=-0.5')
plt.xlim((0,2.))
plt.xlabel('Redshift')
plt.legend(loc='upper right',ncol=1, frameon=True,prop={'size':12})
plt.savefig('pz_bias_test_k.png')
plt.close()




i=0

import scipy.interpolate as interp
for i in range(5):
  tmp=np.genfromtxt('/home/troxel/bpz/data/spec_cat_0.txt',names=True)
  mask=(tmp['magerr_auto_g']>=.9)|(tmp['magerr_auto_r']>=.9)|(tmp['magerr_auto_i']>=.9)|(tmp['magerr_auto_z']>=.9)
  tmp=tmp[~mask]
  h=plt.hist(tmp['magerr_auto_r'],bins=100)
  plt.close()
  magerr=np.vstack((h[1][1:],h[0])).T
  f=interp.interp1d(magerr[:,0],magerr[:,1],kind='cubic',bounds_error=False,fill_value=0.)
  tmp1=np.abs(f(magerr[:,0]*(1.+.5*i**2)))/np.sum(np.abs(f(magerr[:,0]*(1.+.5*i**2))))
  tmp['magerr_auto_g']=np.random.choice(magerr[:,0],len(tmp),p=tmp1)
  tmp['magerr_auto_r']=np.random.choice(magerr[:,0],len(tmp),p=tmp1)
  tmp['magerr_auto_i']=np.random.choice(magerr[:,0],len(tmp),p=tmp1)
  tmp['magerr_auto_z']=np.random.choice(magerr[:,0],len(tmp),p=tmp1)
  np.savetxt('spec_cat_'+str(i+1)+'.txt',tmp)


# python bpz.py /home/troxel/bpz/data/spec_cat_1/spec_cat_1.txt -NEW_AB no -DZ 0.001 -ODDS 0.68 -INTERP 8 -MIN_RMS 0.067 -ZMAX 5. -PHOTO_ERRORS yes -MIN_MAGERR 0.05 -PROBS_LITE yes -VERBOSE no -COLUMNS /home/troxel/bpz/data/spec_cat_0.columns -SPECTRA CWWSB4.list -SED_DIR /home/troxel/bpz/bpz/SED/ -FILTER_DIR /home/troxel/bpz/data/FILTERS/ -OUTPUT /home/troxel/bpz/data/spec_cat_1/spec_cat_1.bpz
# python bpz.py /home/troxel/bpz/data/spec_cat_2/spec_cat_2.txt -NEW_AB no -DZ 0.001 -ODDS 0.68 -INTERP 8 -MIN_RMS 0.067 -ZMAX 5. -PHOTO_ERRORS yes -MIN_MAGERR 0.05 -PROBS_LITE yes -VERBOSE no -COLUMNS /home/troxel/bpz/data/spec_cat_0.columns -SPECTRA CWWSB4.list -SED_DIR /home/troxel/bpz/bpz/SED/ -FILTER_DIR /home/troxel/bpz/data/FILTERS/ -OUTPUT /home/troxel/bpz/data/spec_cat_2/spec_cat_2.bpz
# python bpz.py /home/troxel/bpz/data/spec_cat_3/spec_cat_3.txt -NEW_AB no -DZ 0.001 -ODDS 0.68 -INTERP 8 -MIN_RMS 0.067 -ZMAX 5. -PHOTO_ERRORS yes -MIN_MAGERR 0.05 -PROBS_LITE yes -VERBOSE no -COLUMNS /home/troxel/bpz/data/spec_cat_0.columns -SPECTRA CWWSB4.list -SED_DIR /home/troxel/bpz/bpz/SED/ -FILTER_DIR /home/troxel/bpz/data/FILTERS/ -OUTPUT /home/troxel/bpz/data/spec_cat_3/spec_cat_3.bpz
# python bpz.py /home/troxel/bpz/data/spec_cat_4/spec_cat_4.txt -NEW_AB no -DZ 0.001 -ODDS 0.68 -INTERP 8 -MIN_RMS 0.067 -ZMAX 5. -PHOTO_ERRORS yes -MIN_MAGERR 0.05 -PROBS_LITE yes -VERBOSE no -COLUMNS /home/troxel/bpz/data/spec_cat_0.columns -SPECTRA CWWSB4.list -SED_DIR /home/troxel/bpz/bpz/SED/ -FILTER_DIR /home/troxel/bpz/data/FILTERS/ -OUTPUT /home/troxel/bpz/data/spec_cat_4/spec_cat_4.bpz
# python bpz.py /home/troxel/bpz/data/spec_cat_5/spec_cat_5.txt -NEW_AB no -DZ 0.001 -ODDS 0.68 -INTERP 8 -MIN_RMS 0.067 -ZMAX 5. -PHOTO_ERRORS yes -MIN_MAGERR 0.05 -PROBS_LITE yes -VERBOSE no -COLUMNS /home/troxel/bpz/data/spec_cat_0.columns -SPECTRA CWWSB4.list -SED_DIR /home/troxel/bpz/bpz/SED/ -FILTER_DIR /home/troxel/bpz/data/FILTERS/ -OUTPUT /home/troxel/bpz/data/spec_cat_5/spec_cat_5.bpz



plt.plot(hg[1][1:],hg[0])
plt.savefig('tmpg.png')
plt.close()
plt.plot(hr[1][1:],hr[0])
plt.savefig('tmpr.png')
plt.close()
plt.plot(hi[1][1:],hi[0])
plt.savefig('tmpi.png')
plt.close()
plt.plot(hz[1][1:],hz[0])
plt.savefig('tmpz.png')
plt.close()

np.save('y1_r_magerr.npy',np.vstack((hr[1][1:],hr[0])).T)


# rm *fits
# rm results.*
# python end-to-end.py -c config_files/end-to-end-jaz-fornax.yaml -v 2
# python -m py3shape.analyze_meds2 output/meds/tb-y1a1-spt/final/DES0419-4914-r-sim-nfexp-meds-tb-y1a1-spt.fits e2e.ini all results 0 4





tmp=np.genfromtxt('/home/troxel/bpz/data/spec_cat_1/spec_cat_1.txt')
np.mean(tmp[:,7]),np.median(tmp[:,7])
tmp=np.genfromtxt('/home/troxel/bpz/data/spec_cat_2/spec_cat_2.txt')
np.mean(tmp[:,7]),np.median(tmp[:,7])
tmp=np.genfromtxt('/home/troxel/bpz/data/spec_cat_3/spec_cat_3.txt')
np.mean(tmp[:,7]),np.median(tmp[:,7])
tmp=np.genfromtxt('/home/troxel/bpz/data/spec_cat_4/spec_cat_4.txt')
np.mean(tmp[:,7]),np.median(tmp[:,7])
tmp=np.genfromtxt('/home/troxel/bpz/data/spec_cat_5/spec_cat_5.txt')
np.mean(tmp[:,7]),np.median(tmp[:,7])

# >>> tmp=np.genfromtxt('/home/troxel/bpz/data/spec_cat_1/spec_cat_1.txt')

# >>> np.mean(tmp[:,7]),np.median(tmp[:,7])5
# (0.059471371579627862, 0.036156000000000001)
# >>> tmp=np.genfromtxt('/home/troxel/bpz/data/spec_cat_2/spec_cat_2.txt')
# >>> np.mean(tmp[:,7]),np.median(tmp[:,7])
# (0.041716150429166696, 0.027167)
# >>> tmp=np.genfromtxt('/home/troxel/bpz/data/spec_cat_3/spec_cat_3.txt')
# >>> np.mean(tmp[:,7]),np.median(tmp[:,7])
# (0.023657830368744116, 0.018178)
# >>> tmp=np.genfromtxt('/home/troxel/bpz/data/spec_cat_4/spec_cat_4.txt')
# >>> np.mean(tmp[:,7]),np.median(tmp[:,7])
# (0.015789283704467048, 0.009189000000000001)
# >>> tmp=np.genfromtxt('/home/troxel/bpz/data/spec_cat_5/spec_cat_5.txt')
# >>> np.mean(tmp[:,7]),np.median(tmp[:,7])
# (0.01247826757471213, 0.009189000000000001)

spec=fio.FITS('~/spec_cat_0.fits.gz')[-1].read()


btp=pd.read_csv(config.pzdir+'spec_cat_1.probs',skiprows=1,sep=' ',engine='c',header=None,names=['pdf'+str(i) for i in range(4992) ])
btb=np.genfromtxt(config.pzdir+'spec_cat_1.bpz')
bpz1=catalog.PZStore('bpz1',setup=False,pztype='BPZ')
bpz1.bin=np.arange(0.0100,5.0010,0.0010)[:-1]
bpz1.bins=len(bpz1.bin)
bpz1.pdftype='full'
bpz1.pz_full=btp.values[:,1:-1]
bpz1.z_mean_full=np.zeros(len(bpz1.pz_full))
bpz1.coadd=btb[:,0]
m1,s1,m2,s2=catalog.CatalogMethods.sort(bpz1.coadd,spec['coadd_objects_id'])
bpz1.z_peak_full=btb[:,1]
bpz1.spec_full=spec['z_spec'][m2]
bpz1.w=spec['weights'][m2]
bpz1.wt=False

btp=pd.read_csv(config.pzdir+'spec_cat_2.probs',skiprows=1,sep=' ',engine='c',header=None,names=['pdf'+str(i) for i in range(4992) ])
btb=np.genfromtxt(config.pzdir+'spec_cat_2.bpz')
bpz2=catalog.PZStore('bpz2',setup=False,pztype='BPZ')
bpz2.bin=np.arange(0.0100,5.0010,0.0010)[:-1]
bpz2.bins=len(bpz2.bin)
bpz2.pdftype='full'
bpz2.pz_full=btp.values[:,1:-1]
bpz2.z_mean_full=np.zeros(len(bpz2.pz_full))
bpz2.coadd=btb[:,0]
m1,s1,m2,s2=catalog.CatalogMethods.sort(bpz2.coadd,spec['coadd_objects_id'])
bpz2.z_peak_full=btb[:,1]
bpz2.spec_full=spec['z_spec'][m2]
bpz2.w=spec['weights'][m2]
bpz2.wt=False

btp=pd.read_csv(config.pzdir+'spec_cat_3.probs',skiprows=1,sep=' ',engine='c',header=None,names=['pdf'+str(i) for i in range(4992) ])
btb=np.genfromtxt(config.pzdir+'spec_cat_3.bpz')
bpz3=catalog.PZStore('bpz3',setup=False,pztype='BPZ')
bpz3.bin=np.arange(0.0100,5.0010,0.0010)[:-1]
bpz3.bins=len(bpz3.bin)
bpz3.pdftype='full'
bpz3.pz_full=btp.values[:,1:-1]
bpz3.z_mean_full=np.zeros(len(bpz3.pz_full))
bpz3.coadd=btb[:,0]
m1,s1,m2,s2=catalog.CatalogMethods.sort(bpz3.coadd,spec['coadd_objects_id'])
bpz3.z_peak_full=btb[:,1]
bpz3.spec_full=spec['z_spec'][m2]
bpz3.w=spec['weights'][m2]
bpz3.wt=False


btp=pd.read_csv(config.pzdir+'spec_cat_4.probs',skiprows=1,sep=' ',engine='c',header=None,names=['pdf'+str(i) for i in range(4992) ])
btb=np.genfromtxt(config.pzdir+'spec_cat_4.bpz')
bpz4=catalog.PZStore('bpz4',setup=False,pztype='BPZ')
bpz4.bin=np.arange(0.0100,5.0010,0.0010)[:-1]
bpz4.bins=len(bpz4.bin)
bpz4.pdftype='full'
bpz4.pz_full=btp.values[:,1:-1]
bpz4.z_mean_full=np.zeros(len(bpz4.pz_full))
bpz4.coadd=btb[:,0]
m1,s1,m2,s2=catalog.CatalogMethods.sort(bpz4.coadd,spec['coadd_objects_id'])
bpz4.z_peak_full=btb[:,1]
bpz4.spec_full=spec['z_spec'][m2]
bpz4.w=spec['weights'][m2]
bpz4.wt=False


btp=pd.read_csv(config.pzdir+'spec_cat_5.probs',skiprows=1,sep=' ',engine='c',header=None,names=['pdf'+str(i) for i in range(4992) ])
btb=np.genfromtxt(config.pzdir+'spec_cat_5.bpz')
bpz5=catalog.PZStore('bpz5',setup=False,pztype='BPZ')
bpz5.bin=np.arange(0.0100,5.0010,0.0010)[:-1]
bpz5.bins=len(bpz5.bin)
bpz5.pdftype='full'
bpz5.pz_full=btp.values[:,1:-1]
bpz5.z_mean_full=np.zeros(len(bpz5.pz_full))
bpz5.coadd=btb[:,0]
m1,s1,m2,s2=catalog.CatalogMethods.sort(bpz5.coadd,spec['coadd_objects_id'])
bpz5.z_peak_full=btb[:,1]
bpz5.spec_full=spec['z_spec'][m2]
bpz5.w=spec['weights'][m2]
bpz5.wt=False



pz.pz_methods.build_nofz_bins(bpz1,pzlow=0.2,pzhigh=1.3,cat=None,bins=6,split='peak',nztype='pdf',pzmask=None,catmask=None,spec=True)
pz.pz_methods.build_nofz_bins(bpz2,pzlow=0.2,pzhigh=1.3,cat=None,bins=6,split='peak',nztype='pdf',pzmask=None,catmask=None,spec=True)
pz.pz_methods.build_nofz_bins(bpz3,pzlow=0.2,pzhigh=1.3,cat=None,bins=6,split='peak',nztype='pdf',pzmask=None,catmask=None,spec=True)
pz.pz_methods.build_nofz_bins(bpz4,pzlow=0.2,pzhigh=1.3,cat=None,bins=6,split='peak',nztype='pdf',pzmask=None,catmask=None,spec=True)
pz.pz_methods.build_nofz_bins(bpz5,pzlow=0.2,pzhigh=1.3,cat=None,bins=6,split='peak',nztype='pdf',pzmask=None,catmask=None,spec=True)
bpz1.pz0=bpz1.pz
bpz2.pz0=bpz2.pz
bpz3.pz0=bpz3.pz
bpz4.pz0=bpz4.pz
bpz5.pz0=bpz5.pz

for i in range(len(bpz1.pz)):
  bpz1.pz[i,:]=pz.pz_methods.pdf_mvsk(bpz1.bin,bpz1.pz0[i,:],-bpz1.bin[bpz1.pz0[i,:]==np.max(bpz1.pz0[i,:])]+bpz1.bin[bpz1.spec[i,:]==np.max(bpz1.spec[i,:])],.999,0,0,p6=False)
  bpz2.pz[i,:]=pz.pz_methods.pdf_mvsk(bpz2.bin,bpz2.pz0[i,:],-bpz2.bin[bpz2.pz0[i,:]==np.max(bpz2.pz0[i,:])]+bpz2.bin[bpz2.spec[i,:]==np.max(bpz2.spec[i,:])],.999,0,0,p6=False)
for i in range(len(bpz1.pz)):
  bpz3.pz[i,:]=pz.pz_methods.pdf_mvsk(bpz3.bin,bpz3.pz0[i,:],-bpz3.bin[bpz3.pz0[i,:]==np.max(bpz3.pz0[i,:])]+bpz3.bin[bpz3.spec[i,:]==np.max(bpz3.spec[i,:])],.999,0,0,p6=False)
  bpz4.pz[i,:]=pz.pz_methods.pdf_mvsk(bpz4.bin,bpz4.pz0[i,:],-bpz4.bin[bpz4.pz0[i,:]==np.max(bpz4.pz0[i,:])]+bpz4.bin[bpz4.spec[i,:]==np.max(bpz4.spec[i,:])],.999,0,0,p6=False)
  bpz5.pz[i,:]=pz.pz_methods.pdf_mvsk(bpz5.bin,bpz5.pz0[i,:],-bpz5.bin[bpz5.pz0[i,:]==np.max(bpz5.pz0[i,:])]+bpz5.bin[bpz5.spec[i,:]==np.max(bpz5.spec[i,:])],.999,0,0,p6=False)

fig.plot_methods.plot_nofz_comp_pz([bpz1,bpz2,bpz3,bpz4,bpz5],label='',spec=True)



w1=np.genfromtxt('match_W1.txt',names=True)
w4=np.genfromtxt('match_W4.txt',names=True)
mask1=(w1['fitclass']==0)&(w1['star_flag']==0)&(w1['MASK']<=1)&(w1['weight']>0)
mask4=(w4['fitclass']==0)&(w4['star_flag']==0)&(w4['MASK']<=1)&(w4['weight']>0)
cfht=catalog.CatalogStore('cfhtvipers',cattype='i3',release='y1',setup=False)
cfht.ra=np.hstack((w1['ALPHA_J2000'][mask1],w4['ALPHA_J2000'][mask4]))
cfht.dec=np.hstack((w1['DELTA_J2000'][mask1],w4['DELTA_J2000'][mask4]))
cfht.coadd=np.arange(len(cfht.ra))
cfht.e1=np.hstack((w1['e1'][mask1],w4['e1'][mask4]))
cfht.e2=np.hstack((w1['e2'][mask1],w4['e2'][mask4]))
cfht.c1=np.zeros(len(cfht.e1))
cfht.c2=np.hstack((w1['c2'][mask1],w4['c2'][mask4]))
cfht.m=np.hstack((w1['m'][mask1],w4['m'][mask4]))
cfht.w=np.hstack((w1['weight'][mask1],w4['weight'][mask4]))
cfht.zp=np.hstack((w1['W1_SPECTRO_PDR1zspec'][mask1],w4['W4_SPECTRO_PDR1zspec'][mask4]))
cfht.bs=True
cfht.wt=True
tmp=fio.FITS('cfhtvipers_random.fits.gz')[-1].read()
cfht.ran_ra=tmp['ra']
cfht.ran_dec=tmp['dec']

cfht.ra=cfht.ra/180.*np.pi
cfht.dec=cfht.dec/180.*np.pi
cfht.ran_ra=cfht.ran_ra/180.*np.pi
cfht.ran_dec=cfht.ran_dec/180.*np.pi
cfht.regs=np.zeros(len(cfht.coadd))
cfht.ran_regs=np.zeros(len(cfht.coadd))
cfht.num_regs=1
cfht.lum=10.*np.ones(len(cfht.coadd))

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

r,gp,gx,ee,xx,gperr,gxerr,eeerr,xxerr,tmp=corr.xi_2pt.ia_estimatorb(cfht,cfht,dlos=60.,rbins=10,rmin=.1,rmax=100.,logr=True,lum=.0,comm=comm,rank=rank,size=size,output=True)
if rank==0:

  print r
  print gp
  print gperr
  print gx
  print gxerr
  print ee
  print eeerr
  print xx
  print xxerr

  fig.plot_methods.plot_IA((r[1:]+r[:-1])/2.,[gp,gx],[gperr,gxerr],'ge')
  fig.plot_methods.plot_IA((r[1:]+r[:-1])/2.,[ee,xx],[eeerr,xxerr],'ee')



def in_hull(p, hull):
  from scipy.spatial import Delaunay
  if not isinstance(hull,Delaunay):
    hull = Delaunay(hull)
  return hull.find_simplex(p)>=0

np.max([np.max(tmpspec[i][:,0]) for i in range(len(tmpspec))])
np.min([np.min(tmpspec[i][:,0]) for i in range(len(tmpspec))])
np.max([np.max(tmpspec[i][:,1]) for i in range(len(tmpspec))])
np.min([np.min(tmpspec[i][:,1]) for i in range(len(tmpspec))])

w1
a=[29,-6]
b=[39,-4]
hpmax=hp.ang2pix(4096, np.pi/2.-np.radians(a[1]),np.radians(a[0]), nest=False)
hpmin=hp.ang2pix(4096, np.pi/2.-np.radians(b[1]),np.radians(b[0]), nest=False)

dec,ra=hp.pix2ang(4096,np.arange(hpmax-hpmin)+hpmin,nest=False)
dec=90.-dec*180./np.pi
ra=ra*180./np.pi
mask=(dec>a[1])&(dec<b[1])&(ra>a[0])&(ra<b[0])
pix=np.arange(hpmax-hpmin)[mask]+hpmin
ra=ra[mask]
dec=dec[mask]

tmpphot=[]
with open('vipers_photo_pdr1_W1.reg','r') as f:
  for line in f:
    tmpphot.append(np.fromstring(line, sep=',').reshape(((1+line.count(','))/2,2)))

tmpspec=[]
with open('vipers_pdr1_spectromask_W1.reg','r') as f:
  for line in f:
    tmpspec.append(np.fromstring(line, sep=',').reshape(((1+line.count(','))/2,2)))

goodpix=[]
badpix=[]
for i in range(len(tmpspec)):
  goodpix=np.hstack((goodpix,pix[in_hull(np.vstack((ra,dec)).T,tmpspec[i])]))

for i in range(len(tmpphot)):
  badpix=np.hstack((badpix,pix[in_hull(np.vstack((ra,dec)).T,tmpphot[i])]))

mask2=np.in1d(pix,np.unique(goodpix),assume_unique=True)
mask3=np.in1d(pix,np.unique(badpix),assume_unique=True)
w1pix=pix[mask2&(~mask3)]

w4
a=[329,0]
b=[336,3]
hpmax=hp.ang2pix(4096, np.pi/2.-np.radians(a[1]),np.radians(a[0]), nest=False)
hpmin=hp.ang2pix(4096, np.pi/2.-np.radians(b[1]),np.radians(b[0]), nest=False)

dec,ra=hp.pix2ang(4096,np.arange(hpmax-hpmin)+hpmin,nest=False)
dec=90.-dec*180./np.pi
ra=ra*180./np.pi
mask=(dec>a[1])&(dec<b[1])&(ra>a[0])&(ra<b[0])
pix=np.arange(hpmax-hpmin)[mask]+hpmin
ra=ra[mask]
dec=dec[mask]

tmpphot=[]
with open('vipers_photo_pdr1_W4.reg','r') as f:
  for line in f:
    tmpphot.append(np.fromstring(line, sep=',').reshape(((1+line.count(','))/2,2)))

tmpspec=[]
with open('vipers_pdr1_spectromask_W4.reg','r') as f:
  for line in f:
    tmpspec.append(np.fromstring(line, sep=',').reshape(((1+line.count(','))/2,2)))

goodpix=[]
badpix=[]
for i in range(len(tmpspec)):
  goodpix=np.hstack((goodpix,pix[in_hull(np.vstack((ra,dec)).T,tmpspec[i])]))

for i in range(len(tmpphot)):
  badpix=np.hstack((badpix,pix[in_hull(np.vstack((ra,dec)).T,tmpphot[i])]))

mask2=np.in1d(pix,np.unique(goodpix),assume_unique=True)
mask3=np.in1d(pix,np.unique(badpix),assume_unique=True)
w4pix=pix[mask2&(~mask3)]

pix=np.sort(np.hstack((w1pix,w4pix)))
np.save('wfirstvipersmask.npy',pix)


dec,ra=hp.pix2ang(4096,pix,nest=False)
dec=90.-dec*180./np.pi
ra=ra*180./np.pi
plt.hist2d(ra,dec,bins=100)
plt.savefig('tmp2.png')
plt.close()


plt.plot(hwe.bin,hwe.pz[0,:]/np.sum(hwe.pz[0,:]),label='original')
tmp=pz.pz_methods.pdf_mvsk(hwe.bin,hwe.pz[0,:],0,1.1,0,0)
plt.plot(hwe.bin,tmp/np.sum(tmp),label='m=-0.1')
tmp=pz.pz_methods.pdf_mvsk(hwe.bin,hwe.pz[0,:],0,1.1,0,0)
plt.plot(hwe.bin,tmp/np.sum(tmp),label='m=+0.1')
plt.xlim((0,5.))
plt.xlabel('Redshift')
plt.legend(loc='upper right',ncol=1, frameon=True,prop={'size':12})
plt.savefig('pz_bias_test_m.png')
plt.close()

plt.plot(hwe.bin,hwe.pz[0,:]/np.sum(hwe.pz[0,:]),label='original')
tmp=pz.pz_methods.pdf_mvsk(hwe.bin,hwe.pz[0,:],-.1,.9,0,0)
plt.plot(hwe.bin,tmp/np.sum(tmp),label='m=-0.1')
tmp=pz.pz_methods.pdf_mvsk(hwe.bin,hwe.pz[0,:],.1,.9,0,0)
plt.plot(hwe.bin,tmp/np.sum(tmp),label='m=+0.1')
plt.xlim((0,5.))
plt.xlabel('Redshift')
plt.legend(loc='upper right',ncol=1, frameon=True,prop={'size':12})
plt.savefig('pz_bias_test_v.png')
plt.close()

image=np.empty(rmlum.shape, dtype=rmlum.dtype.descr + [('e1','f8')]+[('e2','f8')]+[('info_flag',int)])
for name in rmlum.dtype.names:
  image[name]=rmlum[name]

image['e1']=np.zeros(len(image))
image['e2']=np.zeros(len(image))
image['info_flag']=np.zeros(len(image)).astype(int)
image['e1'][xlum]=i3.e1[ylum]
image['e2'][xlum]=i3.e2[ylum]
image['info_flag'][xlum]=i3.info[ylum].astype(int)
fio.write(config.redmagicdir+'y1a1_gold_1.0.2b-full_redmapper_v6.4.11_redmagic_highlum_1.0-04_e.fit',image)

image=np.empty(rmdens.shape, dtype=rmdens.dtype.descr + [('e1','f8')]+[('e2','f8')]+[('info_flag',int)])
for name in rmdens.dtype.names:
  image[name]=rmdens[name]

image['e1']=np.zeros(len(image))
image['e2']=np.zeros(len(image))
image['info_flag']=np.zeros(len(image)).astype(int)
image['e1'][xdens]=i3.e1[ydens]
image['e2'][xdens]=i3.e2[ydens]
image['info_flag'][xdens]=i3.info[ydens].astype(int)
fio.write(config.redmagicdir+'y1a1_gold_1.0.2b-full_redmapper_v6.4.11_redmagic_highdens_0.5-10_e.fit',image)

labels=np.zmdens.coadd))
labels[rmdens.dec<-10]=kmspt.labels
labels[rmdens.dec>-10]=kms82.labels+100
np.save(config.redmagicdir+'highdens_regs.npy',labels)

labelss82=kms82.find_nearest(np.vstack((rmlum.ra[rmlum.dec>-10],rmlum.dec[rmlum.dec>-10])).T)
labelsspt=kmspt.find_nearest(np.vstack((rmlum.ra[rmlum.dec<-10],rmlum.dec[rmlum.dec<-10])).T)
labels=np.zeros(len(rmlum.coadd))
labels[rmlum.dec<-10]=labelsspt
labels[rmlum.dec>-10]=labelss82+100
np.save(config.redmagicdir+'highlum_regs.npy',labels)

ran=fio.FITS(config.redmagicdir+'y1a1_gold_1.0.2b-full_redmapper_v6.4.11_redmagic_highdens_0.5-10_randoms.fit')[-1].read()
for i in range(50):
  print i
  tmp=ran[i*1000000:(i+1)*1000000]
  tmplabel=np.zeros((1000000))
  labelss82=kms82.find_nearest(np.vstack((tmp['RA'][tmp['DEC']>-10],tmp['DEC'][tmp['DEC']>-10])).T)
  labelsspt=kmspt.flse:
    labels=np.append(labels,tmplabel)
np.save(config.redmagicdir+'highdens_ran_regs.npy',labels)

ran=fio.FITS(config.redmagicdir+'y1a1_gold_1.0.2b-full_redmapper_v6.4.11_redmagic_highlum_1.0-04_randoms.fit')[-1].read()
for i in range(50):
  print i
  ['COADD_OBJECTS_ID'],i3.coadd[i3.info==0],assume_unique=True)
shapes=np.in1d(spec['COADD_OBJECTS_ID'],i3.coadd,assume_unique=True)

i=6
plt.plot(i3.ra[i3mask],i3.dec[i3mask],marker=',',linestyle='')
mask=(spec['source']==np.unique(spec['source'])[i])
print np.unique(spec['source'])[i],np.sum(mask),np.sum(shapes&mask)
plt.plot(spec['RA'][mask&shapes],spec['DEC'][mask&shapes],marker=',',linestyle='')
plt.savefig('tmp.png')
plt.close()

plt.plot(i3.ra[i3mask],i3.dec[i3mask],marker=',',linestyle='')
for i,x in enumerate(np.unique(spec['source'])):
  mask=(spec['souDR12_LOWZ 
# 18 SDSS_DR7 
# 25 VIPERS fornax
# 27 VVDS http://cesam.lam.fr/vvds/index.php
# 28 WIGGLEZ 

# 0 2DF           5914 3506 952
# min -50.243752 -59.542767
# max 150.996363 2.206212
# 1 3DHST         7011 7004 0
# min 34.216633 -27.947577
# max 150.204789 2.489553
# 2 6DF           10996 536 16
# min -60.99021 -67.406488
# max 99.549189 -0.061681
# 3 ACES          4244 4236 0
# min 52.810661 -28.064835
# max 53.399727 -27.565388
# 4 ATLAS         678 619 127
# min 7.565019 -44.617846
# max 54.314389 -27.112872
78
# 11 FMOS_COSMOS   276 267 0
# min 149.677925 1.734762
# max 150.639658 2.623783
# 12 GAMA          5521 4781 0
# min 33.548728 -7.454585
# max 37.504796 -3.72805
# 13 GLASS         21 19 8
# min -17.845463 -44.550215
# max -17.789938 -44.513285
# 14 LCRS          2770 1204 104
# min -20.002174 -45.502084
# max 69.713184 -40.98432
# 15 MOSFIRE       22 22 0
# min 34.379131 -27.74539
# max 150.156519 2.31946
# 16 NOAO_OZDES    3255 2915 0
# min 51.884631 -29.231167
# max 54.370539 -27.521336
# 17 PANSTARRS     1067 1012 0
# min 34.351537 -29.360229
# max 54.813577 -3.649538
# 18 SDSS_DR7      17576 12332 3864
# min -42.513928 -1.281939
# max 43.204935 1.273746
# 19 SDSS_OZDES    3002 2463 0
# min 33.568816 -7.441368
# max 43.827786 0.961678
# 20 SNLS_AAO      98 9in 35.05348 -5.344455
# max 36.782895 -3.851333
# 24 UDS           1017 1011 0
# min 34.021662 -5.511548
# max 34.894519 -4.737979
# 25 VIPERS        9455 9449 6022
# min -29.881959 0.863121
# max -24.611503 1.801358
# 26 VUDS          56 56 0
# min 53.029416 -27.958304
# max 150.201954 2.555176
# 27 VVDS          11121 11036 2745
# min -151.106534 -27.98707
# max 151.107443 5.572496
# 28 WIGGLEZ       13496 13465 8203
# min -150.057239 -1.767734
# max -0.rt2(i3.coadd,sim.coadd)

plt.figure()
plt.hist((i3.rgp[x])[i3.info[x]==0],bins=100,range=(0,4),alpha=.5)
plt.hist(sim.rgp[y],bins=100,range=(0,4),alpha=.5)
plt.ylabel(r'$n$')
s=config.lbl.get('rgp',None)
if config.log_val.get('rgp',None):
  s='log '+s

plt.xlabel(s)
plt.minorticks_on()
plt.savefig('plots/hist/hist_'+i3.name+'_'+sim.name+'_'+'rgp'+'.png', bbox_inches='tight')
plt.close()

plt.figure()
plt.hist(i3.ra[x],bins=100,alpha=.5)
plt.hist(sim.ra[y],bins=100,alpha=.5)
plt.ylabel(r'$n$')
s=config.lbl.get('ra',None)
if config.log_val.get('ra',None):
  s='log '+s

plt.xlabel(s)
plt.minorticks_on()
plt.savefig('plots/hist/hist_'+i3.
plt.close()

plt.figure()
plt.hist(i3.snr[x],bins=100,range=(0,200),alpha=.5)
plt.hist(sim.snr[y],bins=100,range=(0,200),alpha=.5)
plt.ylabel(r'$n$')
s=config.lbl.get('snr',None)
if config.log_val.get('snr',None):
  s='log '+s

plt.xlabel(s)
plt.minorticks_on()
plt.savefig('plots/hist/hist_'+i3.name+'_'+sim.name+'_'+'snr'+'.png', bbox_inches='tight')
plt.close()

plt.figure()
plt.hist(i3.psf1[x],bins=100,range=(-.1,.1),alpha=.5)
plt.hist(sim.psf1_'+sim.name+'_'+'psf2'+'.png', bbox_inches='tight')
plt.close()

plt.figure()
plt.hist(i3.psffwhm[x],bins=100,range=(2,6),alpha=.5)
plt.hist(sim.psffwhm[y],bins=100,range=(2,6),alpha=.5)
plt.ylabel(r'$n$')
s=config.lbl.get('psffwhm',None)
if config.log_val.get('psffwhm',None):
  s='log '+s

plt.xlabel(s)
plt.minorticks_on()
plt.savefig('plots/hist/hist_'+i3.name+'_'+sim.name+'_'+'psffwhm'+'.png', bbox_inches='tight')
plt.close()

plt.figure()
plt.hist(i3.radius[x],bins=100,range=(0,10),alpha=.5)
plt.hist(sim.radius[y],bins=100,range=(0,10),alpha=.5)
plt.ylabel(r'$n$')
s=config.lbl.get('radius',None)
if config.log_val.get('radius',None):
  s='log '+s

plt.xlabel(s)
plt.minorticks_on()
plt.savefig('plots/hist/hist_'+i3.name+'_'+sim.name+'_'+'radius'+'.png', bbox_inches='tight')
plt.close()

plt.figure()
plt.hist(np.log(i3.dflux[x]*i3.flux[x]),bins=100,range=(2,12),alpha=.5)
plt.hist(np.log(sim.dflux[y]*sim.flux[y]),bins=100,range=(2,12),alpha=.5)
plt.hist(np.log(tmp['flux'][a&(tmp['DES_id']!=-9999)]),bins=100,range=(2,12),alpha=.5)
plt.ylabel(r'$n$')
s=config.lbl.get('flux',None)
if config.log_val.get('flux',None):
  s='log '+s

plt.xlabel(s)
plt.minorticks_on()
plt.savefig('plots/hist/hist_'+i3.name+'_'+sim.name+'_'+'flux'+'.png', bbox_inches='tight')
plt.close()

plt.figure()
plt.hist(i3.nlike[x],bins=100,range=(.01,20000),alpha=.5)
plt.hist(sim.nlike[y],bins=100,range=(.01,20000),alpha=.5)
plt.ylabel(r'$n$')
s=config.lbl.get('like',None)
if config.log_val.get('like',None):
  s='log '+s

plt.xlabel(s)
plt.minorticks_on()
plt.savefig('plots/hist/hist_'+i3.name+'_'+sim.name+'_'+'like'+'.png', bbox_inches='tight')
plt.close()

i3b=fio.FITS('/share/des/disc2/y1/im3shape/single_band/r/y1v1/spte_sv_v1/disc/main_cats/DES0419-4914.fits')[-1].read()
simb=fio.FITS('/share/des/disc2/image-sims/simulation_data/results/DES0419-4914-r-sim-test-meds-tb-y1a1-spt.fits.fz')[-1].read()

np.sum(simb['id'][y]-i3b['coadd_objects_id'][x])

plt.hist2d(simb['snr'][y],simb['disc_flux'][y]*simb['mean_flux'][y],bins=100,norm=LogNorm(),range=((0,100),(0,10000)))
plt.xlabel('snr')
plt.ylabel('flux')
plt.savefig('sim.png')
plt.close()

plt.hist2d(i3b['snr'][x],i3b['disc_flux'][x]*i3b['mean_flux'][x],bins=100,norm=LogNorm(),range=((0,100),(0,10000)))
plt.xlabel('snr')
plt.ylabel('flux')
plt.savefig('i3.png')
plt.close()

plt.hist2d(i3.snr[x],i3.dflux[x],bins=100,norm=LogNorm(),range=((0,100),(0,100)))
plt.xlabel('snr')
plt.ylabel('flux')
plt.savefig('i3.png')
plt.close()
plt.hist2d(sim.snr[y],sim.flux[y],bins=100,norm=LogNorm(),range=((0,100),(0,100)))
plt.xlabel('snr')
plt.ylabel('flux')
plt.savefig('sim.png')
plt.close()
plt.hist2d(sim.snr[y],tmp['flux'][mask],bins=100,norm=LogNorm(),range=((0,100),(0,10000)))
plt.xlabel('snr')
plt.ylabel('flux')
plt.savefig('sim-truth.png')
plt.close()

plt.hist2d(sim.flux[y],tmp['flux'][mask],bins=100,norm=LogNorm(),range=((0,100),(0,10000)))
plt.xlabel('simflux')
plt.ylabel('trueflux')
plt.savefig('sim-truth.png')
plt.close()

plt.hist2d(simb['snr'][y],simb['disc_flux'][y]*i3b['mean_flux'][x],bins=100,norm=LogNorm(),range=((0,100),(0,10000)))
plt.xlabel('snr')
plt.ylabel('flux')
plt.savefig('sim.png')
plt.close()


plt.figure()
plt.hist2d(np.log(i3.dflux[i3.info==0]*i3.flux[i3.info==0]),i3.rad[i3.info==0],bins=100,range=((6,12),(0,4)),norm=LogNorm())
plt.savefig('tmp.png')
plt.close()
plt.figure()
plt.hist2d(np.log(sim.dflux[sim.info==0]*sim.flux[sim.info==0]),sim.rad[sim.info==0],bins=100,range=((6,12),(0,4)),norm=LogNorm())
plt.savefig('tmp2.png')
plt.close()

plt.hist(sim.rad,bins=100,range=(1,4))
plt.savefig('tmp2.png')
plt.close()
plt.hist(np.log(sim.dflux*sim.flux),bins=100,range=(2,12))
plt.savefig('tmp2.png')
plt.close()


for i in range(100):
  imlist = m.get_cutout_list(i)
  plt.imshow(imlist)
  plt.savefig('deepcosmos_'+str(i)+'.png')
  plt.close()


# >>> m[m['id']==3066076294]['orig_row']
# array([[   79.31501007,  3456.15897599,   694.44549111,   202.63799856,
#          1170.03351703, -9999.        ]])
# >>> m[m['id']==3066076294]['orig_col']
# array([[ 7342.95556641,  1943.09145671,  1323.19726025,   319.12497594,
#          1679.7944458 , -9999.        ]])
  
field.field_methods.translate_to_wcs(np.vstack((row[i],col[i])).T,image[(image['expnum']==exp)&(image['ccdnum']==ccd)][j])

spec='spec_'
ada0=fio.FITS(pzdir+'nofz/'+spec+'ada.fits.gz')['nz_shear'].read()
ada1=fio.FITS(pzdir+'out/'+spec+'ada.fits.gz')['nz_shear'].read()
bin=1
plt.plot(ada0['Z_MID'],ada0['BIN'+str(bin)],drawstyle='steps-mid')
plt.plot(ada1['Z_MID'],ada1['BIN'+str(bin)])
plt.plot(np.loadtxt(pzdir+'out/'+spec+'ada/nz_shear/z.txt'),np.loadtxt(pzdir+'out/'+spec+'ada/nz_shear/bin_'+str(bin)+'.txt'))
bin=2
plt.plot(ada0['Z_MID'],ada0['BIN'+str(bin)],drawstyle='steps-mid')
plt.plot(ada1['Z_MID'],ada1['BIN'+str(bin)])
plt.plot(np.loadtxt(pzdir+'out/'+spec+'ada/nz_shear/z.txt'),np.loadtxt(pzdir+'out/'+spec+'ada/nz_shear/bin_'+str(bin)+'.txt'))
bin=3
plt.plot(ada0['Z_MID'],ada0['BIN'+str(bin)],drawstyle='steps-mid')
plt.plot(ada1['Z_MID'],ada1['BIN'+str(bin)])
plt.plot(np.loadtxt(pzdir+'out/'+spec+'ada/nz_shear/z.txt'),np.loadtxt(pzdir+'out/'+spec+'ada/nz_shear/bin_'+str(bin)+'.txt'))
plt.savefig('tmp2.png')
plt.close()

ra=tmp['ra']
tmp['ra'][tmp['ra']>180]=tmp['ra'][tmp['ra']>180]-360



bp=corr.bandpowers()
c=corr._cosmosis()
c.cls(1,1,ell=np.array([bp.lm(x+.5) for x in range(bp.nell)]))
c.xi(1,1)
c.xiobs(bp)
cp,cm,cperr,cmerr=bp.bandpowers(c.xipobs,c.ximobs)
cpeb,cmeb,cpeberr,cmeberr=bp.bandpowersEB(c.xipobs,c.ximobs)

plt.plot(c.ell,c.cl*c.ell*(1.+c.ell)/2./np.pi)
plt.plot(c.ell,cp,linestyle='',marker='o')
plt.plot(c.ell,cpeb,linestyle='',marker='o')
plt.plot(c.ell,cmeb,linestyle='',marker='o')
plt.savefig('bp_test_nonoise_nt-1000.png')
plt.close()

theta=np.load('theta2.npy')
xip=np.load('outxip2.npy')
xim=np.load('outxim2.npy')

theta=np.load('thetadataritomo.npy')
xip=np.load('outxipdataritomo.npy')
xim=np.load('outximdataritomo.npy')
xiperr=np.load('outxiperrdataritomo.npy')
ximerr=np.load('outximerrdataritomo.npy')

theta=np.load('bp_theta_i3_notomo.npy')
xip=np.load('bp_xip_i3_notomo.npy')
xim=np.load('bp_xim_i3_notomo.npy')
xiperr=np.load('bp_xiperr_i3_notomo.npy')
ximerr=np.load('bp_ximerr_i3_notomo.npy')

theta=np.load('bp_theta_i3_notomo_bs.npy')
xip=np.load('bp_xip_i3_notomo_bs.npy')
xim=np.load('bp_xim_i3_notomo_bs.npy')
xiperr=np.load('bp_xiperr_i3_notomo_bs.npy')
ximerr=np.load('bp_ximerr_i3_notomo_bs.npy')

bp=corr.bandpowers(nt=1000,nell=7,lmin=100,lmax=1500,load=False)
bp=corr.bandpowers(nt=500,nell=7,lmin=100,lmax=1500,load=False)
cp,cm=bp.bandpowers(xip,xim)
cpeb,cmeb=bp.bandpowersEB(xip,xim)
ell=np.array([bp.lm(x+.5) for x in range(bp.nell)])

plt.plot(ell,cp,linestyle='',marker='o',label='cp')
plt.plot(ell,cpeb,linestyle='',marker='o',label='cpeb')
plt.plot(ell,cmeb,linestyle='',marker='o',label='cmeb')
plt.legend(loc='lower left')
plt.savefig('tmp.png')
plt.close()

bp=corr.bandpowers(nt=500,nell=7,lmin=100,lmax=1500)
cp,cm,cperr,cmerr=bp.bandpowers(xip,xim,xiperr=xiperr,ximerr=ximerr)
cpeb,cmeb,cpeberr,cmeberr=bp.bandpowersEB(xip,xim,xiperr=xiperr,ximerr=ximerr)
cp2,cm2,cperr2,cmerr2=bp.bandpowers(xip,xim,xiperr=xiperr,ximerr=ximerr)
cpeb2,cmeb2,cpeberr2,cmeberr2=bp.bandpowersEB(xip,xim,xiperr=xiperr,ximerr=ximerr)

plt.errorbar(ell,cmeb,yerr=cmeberr,linestyle='',marker='o',label='No nbc')
plt.errorbar(ell,cmeb2,yerr=cmeberr2,linestyle='',marker='o',label='With nbc')
plt.plot(ell,np.zeros(len(ell)))
plt.ylim((-1e-5,1e-5))
plt.legend(loc='upper left')
plt.savefig('tmp.png')
plt.close()


hdulist = fits.open('/home/troxel/cosmosis/cosmosis-des-library/tcp/2pt_like/des_multiprobe_v1.10.fits')
data=hdulist[-2].data
for x in ['BIN1','BIN2','BIN3']:
  f=scipy.interpolate.interp1d(data['Z_MID'],data[x],kind='cubic',bounds_error=False,fill_value=0.)
  data[x]=np.abs(f(data['Z_MID']+.02))

hdulist.writeto('des_multiprobe_v1.10_p2.fits')
for x in ['BIN1','BIN2','BIN3']:
  f=scipy.interpolate.interp1d(data['Z_MID'],data[x],kind='cubic',bounds_error=False,fill_value=0.)
  data[x]=np.abs(f(data['Z_MID']-.02))

hdulist.writeto('des_multiprobe_v1.10_m2.fits')
for x in ['BIN1','BIN2','BIN3']:
  f=scipy.interpolate.interp1d(data['Z_MID'],data[x],kind='cubic',bounds_error=False,fill_value=0.)
  data[x]=np.abs(f(data['Z_MID']+.04))

hdulist.writeto('des_multiprobe_v1.10_p4.fits')
for x in ['BIN1','BIN2','BIN3']:
  f=scipy.interpolate.interp1d(data['Z_MID'],data[x],kind='cubic',bounds_error=False,fill_value=0.)
  data[x]=np.abs(f(data['Z_MID']-.04))

hdulist.writeto('des_multiprobe_v1.10_m4.fits')
for x in ['BIN1','BIN2','BIN3']:
  f=scipy.interpolate.interp1d(data['Z_MID'],data[x],kind='cubic',bounds_error=False,fill_value=0.)
  data[x]=np.abs(f(data['Z_MID']+.06))

hdulist.writeto('des_multiprobe_v1.10_p6.fits')
for x in ['BIN1','BIN2','BIN3']:
  f=scipy.interpolate.interp1d(data['Z_MID'],data[x],kind='cubic',bounds_error=False,fill_value=0.)
  data[x]=np.abs(f(data['Z_MID']-.06))

hdulist.writeto('des_multiprobe_v1.10_m6.fits')
for x in ['BIN1','BIN2','BIN3']:
  f=scipy.interpolate.interp1d(data['Z_MID'],data[x],kind='cubic',bounds_error=False,fill_value=0.)
  data[x]=np.abs(f(data['Z_MID']+.1))

hdulist.writeto('des_multiprobe_v1.10_p10.fits')
for x in ['BIN1','BIN2','BIN3']:
  f=scipy.interpolate.interp1d(data['Z_MID'],data[x],kind='cubic',bounds_error=False,fill_value=0.)
  data[x]=np.abs(f(data['Z_MID']-.1))

hdulist.writeto('des_multiprobe_v1.10_m10.fits')

tomo0=np.genfromtxt('/home/troxel/cosmosis/cosmosis-des-library/tcp/2pt_like/simonout/p0_means.txt',names=True,dtype=None)
for x in ['m15','m10','m6','m4','m2','p0','p2','p4','p6','p8','p10','p12','p15','p20']:
  tomo=np.genfromtxt('/home/troxel/cosmosis/cosmosis-des-library/tcp/2pt_like/simonout/'+x+'_means.txt',names=True,dtype=None)
  print x,(tomo['mean'][tomo['parameter']=='cosmological_parameters--sigma8_input']-tomo0['mean'][tomo['parameter']=='cosmological_parameters--sigma8_input'])/tomo0['mean'][tomo['parameter']=='cosmological_parameters--sigma8_input']


for x in ['m10','m6','m4','m2','p0']:
  hdulist = fits.open('/home/troxel/destest/des_multiprobe_v1.10_'+x+'.fits')
  data=hdulist[-2].data
  plt.plot(data['Z_MID'],data['BIN1'],drawstyle='steps-mid')

plt.savefig('tmp.png')
plt.close()



for x in ['p0','p2','p4','p6','p8','p10','p12','p15']:
  hdulist = fits.open('/home/troxel/destest/des_multiprobe_v1.10_'+x+'.fits')
  data=hdulist[-2].data
  plt.plot(data['Z_MID'],data['BIN1'],drawstyle='steps-mid')

plt.savefig('tmp2.png')
plt.close()


import time
time0=time.time()
import numpy as np
import src.catalog as catalog
import src.corr as corr
c=np.genfromtxt('partW4.txt',names=['ra', 'dec','e1','e2','weight','fitclass','Z_B','m','c2','star_flag'])
c=np.genfromtxt('/share/des/disc2/cfhtlens/ellipfull_mask0_WLpass_Wall_Zcut.tsv',names=['ra', 'dec','z','e1','e2','weight','m','c2','field'],dtype=None)
cc=catalog.CatalogStore('cfhtlens',setup=False,cattype='i3')
cc.coadd=np.arange(len(c))
cc.ra=c['ra']
cc.dec=c['dec']
cc.e1=c['e1']
cc.e2=c['e2']
cc.c2=c['c2']
cc.w=c['weight']
cc.m1=c['m']
cc.m2=c['m']
cc.c1=np.zeros(len(c))
cc.bs=True
cc.wt=True
cc.sep=[.8,350.]
cc.tbins=21
cc.slop=0.01
theta,out,err,chi2=corr.xi_2pt.xi_2pt(cc,corr='GG',plot=True)
#theta,out,err,chi2=corr.xi_2pt.xi_2pt(cc,catb=cc,corr='NG',plot=True,ran=False)
print time.time()-time0

np.savetxt('100kcfht.txt',np.vstack((theta,out[0],err[0])).T)
np.savetxt('fullcfht.txt',np.vstack((theta,out[0],err[0])).T)


from skymapper import *
maskspt=i3.dec<-20
mask=np.random.choice(np.arange(len(i3.ra[maskspt])),len(i3.ra[maskspt])/10)
bc, ra, dec, vertices = getCountAtLocations(i3.ra[maskspt], i3.dec[maskspt], nside=1024, return_vertices=True)
cmap = cm.YlOrRd
fig = plt.figure(figsize=(8,4))
ax = fig.add_subplot(111, aspect='equal')
proj = createConicMap(ax, ra, dec, proj_class=AlbersEqualAreaProjection)
meridians = np.linspace(-55, -30, 6)
parallels = np.linspace(-60, 100, 17)
setMeridianPatches(ax, proj, meridians, linestyle='-', lw=0.5, alpha=0.3, zorder=2)
setParallelPatches(ax, proj, parallels, linestyle='-', lw=0.5, alpha=0.3, zorder=2)
setMeridianLabels(ax, proj, meridians, loc="left", fmt=pmDegFormatter)
setParallelLabels(ax, proj, parallels, loc="top")
vmin = 1
vmax = 8
poly = plotHealpixPolygons(ax, proj, vertices, color=bc, vmin=vmin, vmax=vmax, cmap=cmap, zorder=2, rasterized=True)
from mpl_toolkits.axes_grid1 import make_axes_locatable
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="2%", pad=0.0)
cb = plt.colorbar(poly, cax=cax)
cb.set_label('$n_g$ [arcmin$^{-2}$]')
ticks = np.linspace(vmin, vmax, 6)
cb.set_ticks(ticks)
cb.solids.set_edgecolor("face")
plt.figtext(.15,.5,'Preliminary',fontsize=70, fontweight='bold',alpha=.5)
plt.tight_layout()
fig.savefig('footprintspt.png')
plt.close()

maskspt=i3.dec>-20
mask=np.random.choice(np.arange(len(i3.ra[maskspt])),len(i3.ra[maskspt])/10)
bc, ra, dec, vertices = getCountAtLocations(i3.ra[maskspt], i3.dec[maskspt], nside=1024, return_vertices=True)
cmap = cm.YlOrRd
fig = plt.figure(figsize=(12,2))
ax = fig.add_subplot(111, aspect='equal')
proj = createConicMap(ax, ra, dec, proj_class=AlbersEqualAreaProjection)
meridians = np.linspace(-2, 3, 6)
parallels = np.linspace(-40, 5, 10)
setMeridianPatches(ax, proj, meridians, linestyle='-', lw=0.5, alpha=0.3, zorder=2)
setParallelPatches(ax, proj, parallels, linestyle='-', lw=0.5, alpha=0.3, zorder=2)
setMeridianLabels(ax, proj, meridians, loc="left", fmt=pmDegFormatter)
setParallelLabels(ax, proj, parallels, loc="top")
vmin = 1
vmax = 8
poly = plotHealpixPolygons(ax, proj, vertices, color=bc, vmin=vmin, vmax=vmax, cmap=cmap, zorder=2, rasterized=True)
from mpl_toolkits.axes_grid1 import make_axes_locatable
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="2%", pad=0.0)
cb = plt.colorbar(poly, cax=cax)
cb.set_label('$n_g$ [arcmin$^{-2}$]')
ticks = np.linspace(vmin, vmax, 5)
cb.set_ticks(ticks)
cb.solids.set_edgecolor("face")
plt.figtext(.3,.35,'Preliminary',fontsize=50, fontweight='bold',alpha=.5)
plt.tight_layout()
fig.savefig('footprints82.png')
plt.close()



from skymapper import *
maskspt=rm.dec<-35

for i in range(100):
  print i
  mask=maskspt&(rm.zp>=edge[i])&((rm.zp<edge[i+1]))
  bc, ra, dec, vertices = getCountAtLocations(rm.ra[mask], rm.dec[mask], nside=1024, return_vertices=True)
  cmap = cm.YlOrRd
  fig = plt.figure(figsize=(8,4))
  ax = fig.add_subplot(111, aspect='equal')
  proj = createConicMap(ax, ra, dec, proj_class=AlbersEqualAreaProjection)
  meridians = np.linspace(-55, -30, 6)
  parallels = np.linspace(-60, 100, 17)
  setMeridianPatches(ax, proj, meridians, linestyle='-', lw=0.5, alpha=0.3, zorder=2)
  setParallelPatches(ax, proj, parallels, linestyle='-', lw=0.5, alpha=0.3, zorder=2)
  setMeridianLabels(ax, proj, meridians, loc="left", fmt=pmDegFormatter)
  setParallelLabels(ax, proj, parallels, loc="top")
  vmin = 0
  vmax = .3
  poly = plotHealpixPolygons(ax, proj, vertices, color=bc, vmin=vmin, vmax=vmax, cmap=cmap, zorder=2, rasterized=True)
  from mpl_toolkits.axes_grid1 import make_axes_locatable
  divider = make_axes_locatable(ax)
  cax = divider.append_axes("right", size="2%", pad=0.0)
  cb = plt.colorbar(poly, cax=cax)
  cb.set_label('$n_g$ [arcmin$^{-2}$]')
  ticks = np.linspace(vmin, vmax, 6)
  cb.set_ticks(ticks)
  cb.solids.set_edgecolor("face")
  # plt.figtext(.15,.5,'Preliminary',fontsize=70, fontweight='bold',alpha=.5)
  plt.tight_layout()
  fig.savefig('rmfootprintzslice_'+str(i)+'.png')
  plt.close()


edge=lin.linear_methods.find_bin_edges(rm.zp,100,w=None)


bc, ra, dec, vertices = getCountAtLocations(rm.ra[maskspt], rm.dec[maskspt], nside=64, return_vertices=True)
cmap = cm.YlOrRd
fig = plt.figure(figsize=(8,4))
ax = fig.add_subplot(111, aspect='equal')
proj = createConicMap(ax, ra, dec, proj_class=AlbersEqualAreaProjection)
meridians = np.linspace(-55, -30, 6)
parallels = np.linspace(-60, 100, 17)
setMeridianPatches(ax, proj, meridians, linestyle='-', lw=0.5, alpha=0.3, zorder=2)
setParallelPatches(ax, proj, parallels, linestyle='-', lw=0.5, alpha=0.3, zorder=2)
setMeridianLabels(ax, proj, meridians, loc="left", fmt=pmDegFormatter)
setParallelLabels(ax, proj, parallels, loc="top")
vmin = 0
vmax = .5
poly = plotHealpixPolygons(ax, proj, vertices, color=bc, vmin=vmin, vmax=vmax, cmap=cmap, zorder=2, rasterized=True)
from mpl_toolkits.axes_grid1 import make_axes_locatable
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="2%", pad=0.0)
cb = plt.colorbar(poly, cax=cax)
cb.set_label('$n_g$ [arcmin$^{-2}$]')
ticks = np.linspace(vmin, vmax, 6)
cb.set_ticks(ticks)
cb.solids.set_edgecolor("face")
# plt.figtext(.15,.5,'Preliminary',fontsize=70, fontweight='bold',alpha=.5)
plt.tight_layout()
fig.savefig('rmfootprintzslice_all.png')
plt.close()

q="""select * from INFORMATION_SCHEMA.COLUMNS where table_name=prod.firstcut_eval"""

from desdb import Connection
conn = Connection()
import numpy as np

q="""select ev.expnum, ev.program from prod.firstcut_eval@desoper ev, prod.exposure@desoper e where e.band = 'u' and e.exptime > 90 and ev.expnum = e.expnum and ev.program = 'supernova' order by ev.expnum"""
uexp=np.unique(conn.quick(q, array=True)['expnum'])
for i,x in enumerate(uexp):
  print i
  q="""select se.ra,se.dec,se.flux_auto,se.fluxerr_auto,se.flux_psf,se.fluxerr_psf,se.flags,se.xwin_image,se.ywin_image from prod.se_object@desoper se where filename like '%s'""" % ('D00'+str(x)+"""_%""")
  data=conn.quick(q, array=True)
  tmpexp=uexp[i]*np.ones(len(data))
  if i==0:
    data0=data
    exp0=tmpexp
  else:
    data0=np.append(data0,data)
    exp0=np.append(exp0,tmpexp)

image=np.ones(data0.shape, dtype=data0.dtype.descr + [('expnum',int)])

for name in data0.dtype.names:
  image[name]=data0[name]

image['expnum']=exp0

import fitsio as fio
fits = fio.FITS('uband_raw.fits','rw')
fits.write(image,clobber=True)
fits.close()

data0=fio.FITS('uband_raw.fits')[-1].read()

q="""select coadd_objects_id,ra,dec,mag_auto_g,magerr_auto_g,mag_auto_r,magerr_auto_r,mag_auto_i,magerr_auto_i,mag_auto_z,magerr_auto_z,mag_auto_y,magerr_auto_y,modest_class,desdm_zp,flags_gold,flags_badregion,hpix from nsevilla.y1a1_gold_1_0_2_d04"""
gold=conn.quick(q, array=True)

fits = fio.FITS('gold_d04.fits','rw')
fits.write(gold,clobber=True)
fits.close()

gold=fio.FITS('gold_d04.fits')[-1].read()

info_cuts =[
    """data0['flux_auto']<=0""",
    """data0['flux_psf']<=0""",
    """data0['flags']>=4""",
    """1.0857362*(data0['fluxerr_psf']/data0['flux_psf']) >= 0.5""",
    """data0['xwin_image']<15""",
    """data0['xwin_image']>2018""",
    """data0['ywin_image']<15""",
    """data0['ywin_image']>4066"""
]

info = np.zeros(len(data0), dtype=np.int64)
for i,cut in enumerate(info_cuts):
  mask=eval(cut).astype(int)
  print i,cut,np.sum(mask)
  j=1<<i
  flags=mask*j
  info|=flags

from sklearn.neighbors import KDTree as kdtree
tree=kdtree(np.vstack((data0['ra'],data0['dec'])).T[info<2**4], leaf_size=2)
match=tree.query(np.vstack((gold['ra'],gold['dec'])).T, k=1, return_distance=True, sort_results=True)

mask=np.where(info<2**4)[0]
coadd=-1*np.ones(len(data0)).astype(int)
coadd[mask[match[1].reshape(len(match[1]))[match[0].reshape(len(match[0]))<1./60./60.]]]=gold[np.where(match[0].reshape(len(match[0]))<1./60./60.)[0]]

spec=fio.FITS('/home/troxel/spec_cat_0.fits.gz')[-1].read()
x,y=catalog.CatalogMethods.sort2(gold['coadd_objects_id'],coadd[coadd!=-1])
x2,y2=catalog.CatalogMethods.sort2(spec['coadd_objects_id'],coadd[coadd!=-1])

tree=kdtree(np.vstack((data0['ra'],data0['dec'])).T[info<2**4], leaf_size=2)
match0=tree.query_radius(np.vstack((data0['ra'],data0['dec'])).T[info<2**4], r=2./60./60., return_distance=True, count_only=False,sort_results=True)

num=np.ones(np.max(coadd))
for j in range(50):
  print j
  for i in range(len(match0[0])):
    if i%10000==0:
      print i
    if coadd[i]==-1:
      if len(match0[0][i])>j:
        if data0['expnum'][match0[0][i][j]]!=data0['expnum'][i]:
          if coadd[match0[0][i][j]]!=-1:
            num[coadd[match0[0][i][j]]-1]+=1
            if num[coadd[match0[0][i][j]]-1]<=10:
              coadd[i]=coadd[match0[0][i][j]]

image=np.ones(data0.shape, dtype=data0.dtype.descr + [('info',int)]+[('coadd_objects_id',int)]+[('z_spec','f8')]+[('flux_auto_g','f8')]+[('fluxerr_auto_g','f8')]+[('flux_auto_r','f8')]+[('fluxerr_auto_r','f8')]+[('flux_auto_i','f8')]+[('fluxerr_auto_i','f8')]+[('flux_auto_z','f8')]+[('fluxerr_auto_z','f8')]+[('flux_auto_y','f8')]+[('fluxerr_auto_y','f8')]+[('modest_class',int)]+[('flags_gold',int)]+[('flags_badregion',int)]+[('hpix',int)]+[('desdm_zp','f8')])

image2=np.ones(len(np.unique(coadd))-1, dtype=[('flux_auto_u_mean','f8')]+[('flux_auto_u_median','f8')]+[('fluxerr_auto_u','f8')] + [('info',int)]+[('coadd_objects_id',int)]+[('z_spec','f8')]+[('flux_auto_g','f8')]+[('fluxerr_auto_g','f8')]+[('flux_auto_r','f8')]+[('fluxerr_auto_r','f8')]+[('flux_auto_i','f8')]+[('fluxerr_auto_i','f8')]+[('flux_auto_z','f8')]+[('fluxerr_auto_z','f8')]+[('flux_auto_y','f8')]+[('fluxerr_auto_y','f8')]+[('modest_class',int)]+[('flags_gold',int)]+[('flags_badregion',int)]+[('hpix',int)]+[('desdm_zp','f8')])

for name in image.dtype.names:
  image[name]*=-9999

for name in image2.dtype.names:
  image2[name]*=-9999

for name in data0.dtype.names:
  image[name]=data0[name]

uc=np.unique(coadd)
for i,x in enumerate(uc):
  if x==-1:
    continue
  if i%100==0:
    print i,len(uc)
  mask=gold['coadd_objects_id']==x
  mask2=np.in1d(coadd,x,assume_unique=False)
  mask3=spec['coadd_objects_id']==x
  mask4=info[mask2]==0
  if np.sum(mask4>0):
    image2['flux_auto_u_mean'][i]=np.mean((data0['flux_auto'][mask2])[mask4])
    image2['flux_auto_u_median'][i]=np.median((data0['flux_auto'][mask2])[mask4])
    image2['fluxerr_auto_u'][i]=np.mean((data0['fluxerr_auto'][mask2])[mask4])/np.sqrt(np.sum(mask4))
    image2['info'][i]=0
  else:
    image2['info'][i]=1
  if np.sum(mask3)>0:
    image['z_spec'][mask2]=spec['z_spec'][mask3]
  else:
    image['z_spec'][mask2]=-9999
  for name in gold.dtype.names:
    if name!='coadd_objects_id':
      if name in image.dtype.names:
        image[name][mask2]=gold[name][mask]
      if name in image2.dtype.names:
        image2[name][i]=gold[name][mask]

image['coadd_objects_id']=coadd
image['info']=info

fits = fio.FITS('y3_u_match.fits','rw')
fits.write(image,clobber=True)
fits.close()


data0=fio.FITS('y3_u_match.fits')[-1].read()

tree=kdtree(np.vstack((data0['ra'],data0['dec'])).T[data0['info']<2**4], leaf_size=2)
match0=tree.query_radius(np.vstack((data0['ra'],data0['dec'])).T[data0['info']<2**4], r=1./60./60., return_distance=True, count_only=False,sort_results=True)

coadd=data0['coadd_objects_id'][data0['info']<2**4]

num=np.ones(np.max(coadd))
for j in range(10):
  print j
  for i in range(len(match0[0])):
    if i%10000==0:
      print i
    if coadd[i]==-1:
      if len(match0[0][i])>j:
        if coadd[match0[0][i][j]]!=-1:
          num[coadd[match0[0][i][j]]-1]+=1
          if num[coadd[match0[0][i][j]]-1]<=10:
            coadd[i]=coadd[match0[0][i][j]]
            # if num[match0[0][i][j]]>9:
            #   if np.sum(coadd==coadd[match0[0][i][j]])>10:
            #     print coadd[match0[0][i][j]]

tmp=[]
for i,x in enumerate(np.unique(table['coadd_objects_id'])):
  if i%1000==0:
    print i
  tmp.append(np.sum(table['coadd_objects_id']==x))
  print np.sum(table['coadd_objects_id']==x)

tmp=np.array(tmp)


image=np.ones(data0.shape, dtype=data0.dtype.descr + [('info',int)]+[('coadd_objects_id',int)])

table.join(left,right,keys='coadd_objects_id',join_type='outer')
a=table.Table.read('gold_d04.fits')
b=table.Table.read('/home/troxel/spec_cat_0.fits.gz')
c=table.Table.read('y3_u_match.fits')



tmp0=fio.FITS('y3_u_match_0.fits')[-1].read()
for i in range(19):
  tmp=fio.FITS('y3_u_match_'+str(i+1)+'.fits')[-1].read()
  tmp0[tmp['coadd_objects_id']!=-9999]=tmp[tmp['coadd_objects_id']!=-9999]

fits = fio.FITS('y3_u_match_epoch.fits','rw')
fits.write(tmp0,clobber=True)
fits.close()


tmp0=fio.FITS('y3_u_match2_0.fits')[-1].read()
for i in range(19):
  tmp=fio.FITS('y3_u_match2_'+str(i+1)+'.fits')[-1].read()
  tmp0[tmp['mag_auto_u_mean']!=-9999]=tmp[tmp['mag_auto_u_mean']!=-9999]

fits = fio.FITS('y3_u_match_coadd.fits','rw')
fits.write(tmp0,clobber=True)
fits.close()



import numpy as np
from desdb import Connection
conn = Connection()

tmp=np.genfromtxt('DESSN_stars_ixkael.csv',names=True,dtype=None,delimiter=',')

q="""select ev.expnum, ev.program, ev.cloud_apass from prod.firstcut_eval@desoper ev, prod.exposure@desoper e where e.band = 'u' and e.exptime > 90 and ev.expnum = e.expnum and ev.program = 'supernova' order by ev.expnum"""
uexp=conn.quick(q, array=True)['expnum']
apass=conn.quick(q, array=True)['cloud_apass']
uexp0=np.unique(uexp)

for i,x in enumerate(uexp0):
  print i
  q="""select se.ra,se.dec,se.flux_auto,se.fluxerr_auto,se.flux_psf,se.fluxerr_psf,se.flags,se.xwin_image,se.ywin_image from prod.se_object@desoper se where filename like '%s'""" % ('D00'+str(x)+"""_%""")
  data=conn.quick(q, array=True)
  tmpexp=uexp0[i]*np.ones(len(data))
  tmpapass=np.mean((apass[uexp==uexp0[i]])[~np.isnan(apass[uexp==uexp0[i]])])*np.ones(len(data))
  if i==0:
    data0=data
    exp1=tmpexp
    apass1=tmpapass
  else:
    data0=np.append(data0,data)
    exp1=np.append(exp1,tmpexp)
    apass1=np.append(apass1,tmpapass)

from sklearn.neighbors import KDTree as kdtree
tree=kdtree(np.vstack((tmp['RA'],tmp['DEC'])).T, leaf_size=2)
match=tree.query(np.vstack((data0['ra'],data0['dec'])).T[info<2**4], k=1, return_distance=True, sort_results=True)

mask=match[0].reshape(len(match[0]))<1./60./60.

ugsdss=tmp['MAG_PSF_SFD_U']-tmp['MAG_PSF_SFD_G']

usd=np.zeros(len(tmp))
usd[ugsdss<.5]=tmp['MAG_PSF_SFD_U'][ugsdss<.5] - 0.550*ugsdss[ugsdss<.5] - 0.052
usd[ugsdss>=.5]=tmp['MAG_PSF_SFD_U'][ugsdss>=.5] - 0.204*ugsdss[ugsdss>=.5] - 0.121

desmag=-2.5*np.log10(data0['flux_auto'])
nominal=desmag[mask]-usd[match[1].reshape('magerr_auto_u']=2.5*np.log10(data0['fluxerr_auto']/data0['flux_auto']+1.)


import fitsio as fio
fits line(x=np.average(np.loadtxt('nzs/BCCUFIG_SV_bpz_nouband_zmean_'+bounds[i]+'_'+bounds[i+1]+'_stackedpz.txt')[:,0],weights=np.loadtxt('nzs/BCCUFIG_SV_bpz_nouband_zmean_'+bounds[i]+'_'+bounds[i+1]+'_stackedpz.txt')[:,1]), ymin=0., ymax = 1, linewidth=1, color='r')
  plt.axvline(x=np.average(np.loadtxt('nzs/BCCUFIG_SV_bpz_nouband_zmean_'+bounds[i]+'_'+bounds[i+1]+'_truezhistogram.txt')[:,0],weights=np.loadtxt('nzs/BCCUFIG_SV_bpz_nouam.txt')[:,1],color='k')
  plt.axvline(x=np.average(np.loadtxt('nzs/BCCUFIG_SV_bpz_withuband_zmean_'+bounds[i]+'_'+bounds[i+1]+'_stackedpz.txt')[:,0],weights=np.loadtxt('nzs/BCCUFIG_SV_bpz_withuband_zmean_'+bounds[i]+'_'+bounds[i+1]+'_stackedpz.txt')[:,1]), ymin=0., ymax = 1, linewidth=1, color='b')
  plt.axvline(x=np.average(np.loadtxt('nzs/BCCUFIG_SV_bpz_withuband_zmean_'+bounds[i]+'_'+bounds[i+1]+'_truezhistogram.txt')[:,0],weights=np.loadtxt('nzs/BCCUFIG_SV_bpz_withuband_zmean_'+bounds[i]+'_'+bounds[i+1]+'_truezhistogram.txt')[:,1]), ymin=0., ymax = 1, linewidth=1, color='k')

plt.savefig('withu.png')
plt.close()



bounds1=['0.3','0.5','0.7','0.8','1']
bounds2=['0.48','0.66','0.84','1','1.2']
for i in range(5):
  plt.plot(np.loadtxt('nzs/BCCUFIG_SV_bpz_nouband_zmean_'+bounds1[i]+'_'+bounds2[i]+'_stackedpz.txt')[:,0],np.loadtxt('nzs/BCCUFIG_SV_bpz_nouband_zmean_'+bounds1[i]+'_'+bounds2[i]+'_stackedpz.txt')[:,1],color='r')
  plt.plot(]+'_truezhistogram.txt')[:,0],np.loadtxt('nzs/BCCUFIG_SV_bpz_nouband_zmean_'+bounds1[i]+'_'+bounds2[i]+'_truezhistogram.txt')[:,1],color='k')
  plt.axvline(x=np.average(np.loadtxt('nzs/BCCUFIG_SV_bpz_nouband_zmean_'+bounds1[i]+'_'+bounds2[i]+'_stackedpz.txt')[:,0],weights=np.loadtxt('nzs/BCCUFIG_SV_bpz_nouband_zmean_'+bounds1[i]+'_'+bounds2[i]+'_stackedpz.txt')[:,1]), ymin=0., ymax = 1, linewidth=1, color='r')
  plt.axvline(x=np.average(np.loadtxt('nzs/BCCUFIG_SV_bpz_nouband_zmean_'+bounds1[i]+'_'+bounds2[i]+'_truezhistogram.txt')[:,0],weights=np.loadtxt('nzs/BCCUFIG_SV_bpz_nouband_zmean_'+bounds1[i]+'_'+bounds2[i]+'_truezhistogram.txt')[:,1]), ymin=0., ymax = 1, linewidth=1, color='k')

plt.savefig('nou.png')
plt.close()

for i in range(5):
  plt.plot(np.loadtxt('nzs/BCCUFIG_SV_bpz_withuband_zmean_'+bounds1[i]+'_'+bounds2[i]+'_stackedpz.txt')[:,0],np.loadtxt('nzs/BCCUFIG_SV_bpz_withuband_zmean_'+bounds1[i]+'_'+bounds2[i]+'_stackedpz.txt')[:,1],color='b')
  plt.plot(np.loadtxt('nzs/BCCUFIG_SV_bpz_withuband_zmean_'+bounds1[i]+'_'+bounds2[i]+'_truezhistogram.txt')[:,0],np.loadtxt('nzs/BCCUFIG_SV_bpz_withuband_zmean_'+bounds1[i]+'_'+bounds2[i]+'_truezhistogram.txt')[:,1],color='k')
  plt.axvline(x=np.average(np.loadtxt('nzs/BCCUFIG_SV_bpz_withuband_zmean_'+bounds1[i]+'_'+bounds2[i]+'_stackedpz.txt')[:,0],weights=np.loadtxt('nzs/BCCUFIG_SV_bpz_withuband_zmean_'+bounds1[i]+'_'+bounds2[i]+'_stackedpz.txt')[:,1]), ymin=0., ymax = 1, linewidth=1, color='b')
  plt.axvline(x=np.average(np.loadtxt('nzs/BCCUFIG_SV_bpz_withuband_zmean_'+bounds1[i]+'_'+bounds2[i]+'_truezhistogram.txt')[:,0],weights=np.loadtxt('nzs/BCCUFIG_SV_bpz_withuband_zmean_'+bounds1[i]+'_'+bounds2[i]+'_truezhistogram.txt')[:,1]), ymin=0., ymax = 1, linewidth=1, color='k')

plt.savefig('withu.png')
plt.close()





pz0=catalog.PZStore('bpzu',setup=False,pztype='dnf',filetype='h5',file=config.pzdir+'dnf_validation.hdf5'nl,pz0.pzlens[0,:],pz0.bins,pz0.pzsource[0,:],config.pztestdir+test+'/nofz/notomo_'+pz0.name+'.fits.gz')
cosmo.make.fits(pz0.binl,pz0.pzlens[1:,:],pz0.bins,pz0.pzsource[1:,:],config.pztestdir+test+'/nofz/'+pz0.name+'.fits.gz')
cosmo.make.fits(pz0.binl,pz0.pzlens[0,:],pz0.binspec,pz0.specsource[0,:],config.pztestdir+test+'/nofz/spec_notomo_'+pz0.name+'.fits.gz')
cosmo.make.fits(pz0.binl,pz0.pzlens[1:,:],pz0.binspec,pz0.specsource[1:,:],config.pztestdir+test+'/nofz/spec_'+pz0.name+'.fits.gz')

cosmo.run.submit_pz_spec_test(pz0,'y6test_6','wl',bins=6,boot=False,cosmo=False,submit=True,fillin=False)
# cosmo.run.submit_pz_spec_test(pz0,'y6test_6','wl',bins=6,boot=False,cosmo=True,submit=True,fillin=False)


pz1=catalog.PZStore('bpz',setup=False,pztype='dnf',filetype='h5',file=config.pzdir+'dnf_validation.hdf5')
pz1.name='bpz'
pz1.pztype='bpz'
pz1.pzsource=np.zeros((6,199))
pz1.pzlens=np.zeros((4,100))
pz1:,0]
pz1.binspec=np.loadtxt('nzs/BCCUFIG_SV_bpz_withuband_zmean_0.3_1.2_truezhistogram.txt')[:,0]
pz1.binl=np.loadtxt('/home/troxel/des-mpp/zdistris/zdistribution_redm')[:,0]
pz1.pzsource[0,:]=np.loadtxt('nzs/BCCUFIG_SV_bpz_nouband_zmean_0.3_1.2_stackedpz.txt')[:,1]
pz1.specsource[0,:]=np.loadtxt('nzs/BCCUFIG_SV_bpz_nouband_zmean_0.3_1.2_truezhistogram.txt')[:,1]
bounds1=['0.3','0.5','0.7','0.8','1']
bounds2=['0.48','0.66','0.84','1','1.2']
for i in range(5):
  pz1.pzsource[i+1,:]=np.loadtxt('nzs/BCCUFIG_SV_bpz_nouband_zmean_'+bounds1[i]+'_'+bounds2[i]+'_stackedpz.txt')[:,1]
  pz1.specsource[i+1,:]=np.loadtxt('nzs/BCCUFIG_SV_bpz_nouband_zmean_'+bounds1[i]+'_'+bounds2[i]+'_truezhistogram.txt')[:,1]

pz1.pzlens[0,:]=np.mean(np.loadtxt('/home/troxel/des-mpp/zdistris/zdistribution_redm')[:,1:],axis=1)
for i in range(3):
  pz1.pzlens[i+1,:]=np.loadtxt('/home/troxel/des-mpp/zdistris/zdistribution_redm')[:,i+1]


cosmo.make.fits(pz1.binl,pz1.pzlens[0,:],pz1.bins,pz1.pzsource[0,:],config.pztestdir+test+'/nofz/notomo_'+pz1.name+'.fits.gz')
cosmo.make.fits(pz1.binl,pz1.pzlens[1:,:],pz1.bins,pz1.pzsource[1:,:],config.pztestdir+test+'/nofz/'+pz1.name+'.fits.gz')
cosmo.make.fits(pz1.binl,pz1.pzlens[0,:],pz1.binspec,pz1.specsource[0,:],config.pztestdir+test+'/nofz/spec_notomo_'+pz1.name+'.fits.gz')
cosmo.make.fits(pz1.binl,pz1.pzlens[1:,:],pz1.binspec,pz1.specsource[1:,:],config.pztestdir+test+'/nofz/spec_'+pz1.name+'.fits.gz')

cosmo.run.submit_pz_spec_test(pz1,'y6test_6','wl',bins=6,boot=False,cosmo=False,submit=True,fillin=False)
# cosmo.run.submit_pz_spec_test(pz1,'y6test_6','wl',bins=6,boot=False,cosmo=True,submit=True,fillin=False)





pz2=catalog.PZStore('bpzushift',setup=False,pztype='dnf',filetype='h5',file=config.pzdir+'dnf_validation.hdf5')
pz2.name='bpzushift'
pz2.pztype='bpzushift'
pz2.pzsource=np.zeros((6,199))
pz2.pzlens=np.zeros((4,100))
pz2.specsource=np.zeros((6,30))
pz2.bins=np.loadtxt('nzs/BCCUFIG_SV_bpz_withuband_zmean_0.3_1.2_stackedpz.txt')[:,0]
pz2.binspec=np.loadtxt('nzs/BCCUFIG_SV_bpz_withuband_zmean_0.3_1.2_truezhistogram.txt')[:,0]
pz2.binl=np.loadtxt('/home/troxel/des-mpp/zdistris/zdistribution_redm')[:,0]
pz2.pzsource[0,:]=np.loadtxt('nzs/BCCUFIG_SV_bpz_withuband_zmean_0.3_1.2_stackedpz.txt')[:,1]
pz2.specsource[0,:]=np.loadtxt('nzs/BCCUFIG_SV_bpz_withuband_zmean_0.3_1.2_truezhistogram.txt')[:,1]
shift=np.average(pz2.binspec,weights=pz2.specsource[0,:])-np.average(pz2.bins,weights=pz2.pzsource[0,:])
pz2.pzsource[0,:]=pz.pz_methods.pdf_mvsk(pz2.bins,pz2.pzsource[0,:],dm=shift)
bounds1=['0.3','0.5','0.7','0.8','1']
bounds2=['0.48','0.66','0.84','1','1.2']
for i in range(5):
  pz2.pzsource[i+1,:]=np.loadtxt('nzs/BCCUFIG_SV_bpz_withuband_zmean_'+bounds1[i]+'_'+bounds2[i]+'_stackedpz.txt')[:,1]
  pz2.specsource[i+1,:]=np.loadtxt('nzs/BCCUFIG_SV_bpz_withuband_zmean_'+bounds1[i]+'_'+bounds2[i]+'_truezhistogram.txt')[:,1]
  shift=np.average(pz2.binspec,weights=pz2.specsource[i+1,:])-np.average(pz2.bins,weights=pz2.pzsource[i+1,:])
  pz2.pzsource[i+1,:]=pz.pz_methods.pdf_mvsk(pz2.bins,pz2.pzsource[i+1,:],dm=shift)

pz2.pzlens[0,:]=np.mean(np.loadtxt('/home/troxel/des-mpp/zdistris/zdistribution_redm')[:,1:],axis=1)
for i in range(3):
  pz2.pzlens[i+1,:]=np.loadtxt('/home/troxel/des-mpp/zdistris/zdistribution_redm')[:,i+1]



cosmo.make.fits(pz2.binl,pz2.pzlens[0,:],pz2.bins,pz2.pzsource[0,:],config.pztestdir+test+'/nofz/notomo_'+pz2.name+'.fits.gz')
cosmo.make.fits(pz2.binl,pz2.pzlens[1:,:],pz2.bins,pz2.pzsource[1:,:],config.pztestdir+test+'/nofz/'+pz2.name+'.fits.gz')
cosmo.make.fits(pz2.binl,pz2.pzlens[0,:],pz2.binspec,pz2.specsource[0,:],config.pztestdir+test+'/nofz/spec_notomo_'+pz2.name+'.fits.gz')
cosmo.make.fits(pz2.binl,pz2.pzlens[1:,:],pz2.binspec,pz2.specsource[1:,:],config.pztestdir+test+'/nofz/spec_'+pz2.name+'.fits.gz')

cosmo.run.submit_pz_spec_test(pz2,'y6test_6','wl',bins=6,boot=False,cosmo=False,submit=True,fillin=False)
# cosmo.run.submit_pz_spec_test(pz2,'y6test_6','wl',bins=6,boot=False,cosmo=True,submit=True,fillin=False)


pz3=catalog.PZStore('bpzshift',setup=False,pztype='dnf',filetype='h5',file=config.pzdir+'dnf_validation.hdf5')
pz3.name='bpzshift'
pz3.pztype='bpzshift'
pz3.pzsource=np.zeros((6,199))
pz3.pzlens=np.zeros((4,100))
pz3.specsource=np.zeros((6,30))
pz3.bins=np.loadtxt('nzs/BCCUFIG_SV_bpz_withuband_zmean_0.3_1.2_stackedpz.txt')[:,0]
pz3.binspec=np.loadtxt('nzs/BCCUFIG_SV_bpz_withuband_zmean_0.3_1.2_truezhistogram.txt')[:,0]
pz3.binl=np.loadtxt('/home/troxel/des-mpp/zdistris/zdistribution_redm')[:,0]
pz3.pzsource[0,:]=np.loadtxt('nzs/BCCUFIG_SV_bpz_nouband_zmean_0.3_1.2_stackedpz.txt')[:,1]
pz3.specsource[0,:]=np.loadtxt('nzs/BCCUFIG_SV_bpz_withuband_zmean_0.3_1.2_truezhistogram.txt')[:,1]
shift=np.average(pz3.binspec,weights=pz3.specsource[0,:])-np.average(pz3.bins,weights=pz3.pzsource[0,:])
pz3.pzsource[0,:]=pz.pz_methods.pdf_mvsk(pz3.bins,pz3.pzsource[0,:],dm=shift)
bounds1=['0.3','0.5','0.7','0.8','1']
bounds2=['0.48','0.66','0.84','1','1.2']
for i in range(5):
  pz3.pzsource[i+1,:]=np.loadtxt('nzs/BCCUFIG_SV_bpz_nouband_zmean_'+bounds1[i]+'_'+bounds2[i]+'_stackedpz.txt')[:,1]
  pz3.specsource[i+1,:]=np.loadtxt('nzs/BCCUFIG_SV_bpz_nouband_zmean_'+bounds1[i]+'_'+bounds2[i]+'_truezhistogram.txt')[:,1]
  shift=np.average(pz3.binspec,weights=pz3.specsource[i+1,:])-np.average(pz3.bins,weights=pz3.pzsource[i+1,:])
  pz3.pzsource[i+1,:]=pz.pz_methods.pdf_mvsk(pz3.bins,pz3.pzsource[i+1,:],dm=shift)

pz3.pzlens[0,:]=np.mean(np.loadtxt('/home/troxel/des-mpp/zdistris/zdistribution_redm')[:,1:],axis=1)
for i in range(3):
  pz3.pzlens[i+1,:]=np.loadtxt('/home/troxel/des-mpp/zdistris/zdistribution_redm')[:,i+1]


cosmo.make.fits(pz3.binl,pz3.pzlens[0,:],pz3.bins,pz3.pzsource[0,:],config.pztestdir+test+'/nofz/notomo_'+pz3.name+'.fits.gz')
cosmo.make.fits(pz3.binl,pz3.pzlens[1:,:],pz3.bins,pz3.pzsource[1:,:],config.pztestdir+test+'/nofz/'+pz3.name+'.fits.gz')
cosmo.make.fits(pz3.binl,pz3.pzlens[0,:],pz3.binspec,pz3.specsource[0,:],config.pztestdir+test+'/nofz/spec_notomo_'+pz3.name+'.fits.gz')
cosmo.make.fits(pz3.binl,pz3.pzlens[1:,:],pz3.binspec,pz3.specsource[1:,:],config.pztestdir+test+'/nofz/spec_'+pz3.name+'.fits.gz')

cosmo.run.submit_pz_spec_test(pz3,'y6test_6','wl',bins=6,boot=False,cosmo=False,submit=True,fillin=False)
# cosmo.run.submit_pz_spec_test(pz3,'y6test_6','wl',bins=6,boot=False,cosmo=True,submit=True,fillin=False)


tmp=fio.FITS('/scratch2/scratchdirs/troxel/shape_cat_3z.fits')[-1].read()

i3=np.zeros(len(tmp), dtype=[('coadd_objects_id','i8')]+[('e1','f8')]+[('e2','f8')]+[('m1','f8')]+[('m2','f8')]+[('c1','f8')]+[('c2','f8')]+[('weight','f8')]+[('radius','f8')]+[('snr','f8')]+[('likelihood','f8')]+[('chi2_pixel','f8')]+[('n_exposure','i8')]+[('stampe_size','i8')]+[('info_flag','i8')]+[('ra_as','f8')]+[('dec_as','f8')]+[('covmat_0_0','f8')]+[('covmat_0_1','f8')]+[('covmat_1_1','f8')]+[('rgpp_rp','f8')]+[('psf_e1','f8')]+[('psf_e2','f8')]+[('psf_fwhm','f8')]+[('mask_fraction','f8')]+[('round_snr','f8')]+[('bulge_fraction','f8')])
gold=np.zeros(len(tmp), dtype=[('coadd_objects_id','i8')]+[('ra','f8')]+[('dec','f8')]+[('spread_model_r','f8')]+[('spread_model_i','f8')]+[('spread_model_z','f8')]+[('spread_modelerr_r','f8')]+[('spreaderr_model_i','f8')]+[('spreaderr_model_z','f8')]+[('ebv','f8')]+[('mag_auto_r','f8')]+[('mag_auto_i','f8')]+[('mag_auto_z','f8')]+[('flags','i8')]+[('zp','i8')]+[('zperr','i8')]+[('modest_class','f8')]+[('hpix','f8')]+[('flags_r','f8')]+[('flags_i','f8')]+[('flags_z','f8')])

photoz=np.zeros(len(tmp), dtype=[('coadd_objects_id','i8')]+[('mean_z','f8')]+[('mode_z','f8')]+[('sample_z','f8')]+[('weight','f8')]+[('flags','i8')])

i3['coadd_objects_id']=tmp['ID']
gold['coadd_objects_id']=tmp['ID']
gold['ra']=tmp['RA']
gold['dec']=tmp['DEC']
gold['mag_auto_r']=tmp['MAG_R']
gold['mag_auto_i']=tmp['MAG_I']
gold['mag_auto_z']=tmp['MAG_Z']
gold['zp']=tmp['Z']
i3['psf_fwhm']=tmp['psf_fwhm_r']
i3['e1']=tmp['EPSILON1']
i3['e2']=tmp['EPSILON2']
photoz['mean_z']=tmp['PHOTOZ_GAUSSIAN']
photoz['sample_z']=tmp['Z']
photoz['coadd_objects_id']=tmp['ID']
photoz['weight']=np.ones(len(tmp))

fio.write('buzzard_3_shape_v0.fits',i3,clobber=True)
fio.write('buzzard_3_gold_v0.fits',gold,clobber=True)
fio.write('buzzard_3_photoz_v0.fits',photoz,clobber=True)


files=glob.glob('/global/project/projectdirs/des/jderose/addgals/catalogs/Buzzard/Catalog_v1.1/four-y1-mocks/mock3/y1a1_errormodel_rotated/*')[:-1]
files0=[]
for i in range(len(files)):
  m=re.search('Buzzard_v1.1_Y1A1.(\d*).fit',files[i])
  if m is not None:
    files0.append(m.group(1))

cid=[]
pz=[]
z=[]
for i,file in enumerate(glob.glob('/global/project/projectdirs/des/jderose/addgals/catalogs/Buzzard/Catalog_v1.1/four-y1-mocks/mock3/truth_rotated/*')):
  print i,file
  m=re.search('Buzzard_v1.1_truth.(\d*).fit',file)
  if m is None:
    continue
  else:
    if m.group(1) not in files0:
      continue
  tmp=fio.FITS(file)[-1].read(columns=['ID','PHOTOZ_GAUSSIAN','Z'])
  cid=np.append(cid,tmp['ID'])
  pz=np.append(pz,tmp['PHOTOZ_GAUSSIAN'])
  z=np.append(z,tmp['Z'])

tmp=fio.FITS('/scratch2/scratchdirs/troxel/buzzard_3_gold_v0.fits')[-1].read(columns='coadd_objects_id')


