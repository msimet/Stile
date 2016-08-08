import numpy as np
try:
  import treecorr
except:
  print "No treecorr"
  treecorr=None

import os
if "NERSC_HOST" not in os.environ:
  from mpi4py import MPI
else:
  print 'No mpi4py'

import scipy
import math

import catalog
import cosmo
import config
import fig
import txt
import lin

class UseError(Exception):
  def __init__(self, value):
    self.value = value
  def __str__(self):
    return repr(self.value)

class xi_2pt(object):

  @staticmethod
  def xi_2pt_tomo(cata,bins,catb=None,k=None,ga=None,gb=None,corr='GG',maska=None,maskb=None,wa=None,wb=None,ran=True,mock=False,erron=True,jkmask=None,label0='',plot=False,conj=False):

    theta=np.zeros(((len(bins)-1),(len(bins)-1),cata.tbins))
    xip=np.zeros(((len(bins)-1),(len(bins)-1),cata.tbins))
    xim=np.zeros(((len(bins)-1),(len(bins)-1),cata.tbins))
    xip_im=np.zeros(((len(bins)-1),(len(bins)-1),cata.tbins))
    xim_im=np.zeros(((len(bins)-1),(len(bins)-1),cata.tbins))
    xiperr=np.zeros(((len(bins)-1),(len(bins)-1),cata.tbins))
    ximerr=np.zeros(((len(bins)-1),(len(bins)-1),cata.tbins))
    xip_imerr=np.zeros(((len(bins)-1),(len(bins)-1),cata.tbins))
    xim_imerr=np.zeros(((len(bins)-1),(len(bins)-1),cata.tbins))
    xipchi2=np.zeros(((len(bins)-1),(len(bins)-1),cata.tbins))
    ximchi2=np.zeros(((len(bins)-1),(len(bins)-1),cata.tbins))
    xip_imchi2=np.zeros(((len(bins)-1),(len(bins)-1),cata.tbins))
    xim_imchi2=np.zeros(((len(bins)-1),(len(bins)-1),cata.tbins))

    maska=catalog.CatalogMethods.check_mask(cata.coadd,maska)
    if catb is not None:
      maskb=catalog.CatalogMethods.check_mask(catb.coadd,maskb)

    for i in range(len(bins)-1):
      for j in range(len(bins)-1):
        if (i>j)&(catb is None):
          continue
        if catb is not None:
          theta[i,j,:],[xip[i,j,:],xim[i,j,:],xip_im[i,j,:],xim_im[i,j,:]],[xiperr[i,j,:],ximerr[i,j,:],xip_imerr[i,j,:],xim_imerr[i,j,:]],[xipchi2[i,j,:],ximchi2[i,j,:],xip_imchi2[i,j,:],xim_imchi2[i,j,:]]=xi_2pt.xi_2pt(cata,catb=catb,k=k,ga=ga,gb=gb,corr=corr,maska=maska&(cata.zp>bins[i])&(cata.zp<bins[i+1]),maskb=maskb&(catb.zp>bins[j])&(catb.zp<bins[j+1]),wa=wa,wb=wb,ran=ran,mock=mock,erron=erron,jkmask=jkmask,label0=label0,plot=plot,conj=conj)
        else:
          theta[i,j,:],[xip[i,j,:],xim[i,j,:],xip_im[i,j,:],xim_im[i,j,:]],[xiperr[i,j,:],ximerr[i,j,:],xip_imerr[i,j,:],xim_imerr[i,j,:]],[xipchi2[i,j,:],ximchi2[i,j,:],xip_imchi2[i,j,:],xim_imchi2[i,j,:]]=xi_2pt.xi_2pt(cata,catb=catb,k=k,ga=ga,gb=gb,corr=corr,maska=maska&(cata.zp>bins[j])&(cata.zp<bins[j+1]),maskb=None,wa=wa,wb=wb,ran=ran,mock=mock,erron=erron,jkmask=jkmask,label0=label0,plot=plot,conj=conj)

    return theta,[xip,xim,xip_im,xim_im],[xiperr,ximerr,xip_imerr,xim_imerr],[xipchi2,ximchi2,xip_imchi2,xim_imchi2]

  @staticmethod
  def xi_2pt(cata,catb=None,k=None,ga=None,gb=None,corr='GG',maska=None,maskb=None,wa=None,wb=None,ran=True,mock=False,erron=True,jkmask=None,label0='',plot=False,conj=False):
    """
    This is a flexible convenience wrapper for interaction with treecorr to work on CatalogStore objects. Some basic examples are given in corr_tests() of the main testsuite.py. g1, g2 correctly by c1, c2 if ellipticities and cat.bs is true. Correction by sensitivity, 1+m applied if cat.bs=True. Weighting applied if cat.wt is true. Other config properties for treecorr stored in CatalogStore object. See catalog.py or config.py. Not all correlation types fully integrated or tested. For example, only one kappa value is currently possible. Will be updated in future as useful.

    Use:

    :cata, catb:    CatalogStore - Must supply both cata, catb (can be same reference) if NG or NK correlation. Otherwise catb is optional.
    :k:             str - Array name in cata, catb to use for kappa correlation. 
    :ga, gb:        str - Array names for g1, g2 treecorr inputs. If None assume e1, e2.
    :corr:          str - Type of correlation for treecorr.
    :maska, maskb:  [bool] - Masking array to apply to input catalogs.
    :wa, wb:        [float] - Additional weights to apply after cat.w is used. Combined as e.g., w=sqrt(cat.w*wa).
    :ran:           bool - Use randoms in correlation calculation. If True, assumes cat.ran_ra, cat.ran_dec exist.
    :mock:          bool - If mock catalog from sims. Used when calculating covariances from sims, not currently migrated from SV code.
    :erron:         bool - Calculate jackknife or sim cov errors. If False, uses treecorr error outputs. Not currently migrated from SV code. When implemented requires cat.calc_err in ('jk', 'mock').
    :jkmask:        [bool] - For jk, mock cov calculation loop over regions/sims.
    :label0:        str - Additional (optional) label string used in some outputs.
    :plot:          bool - Plot output?

    Output (len cat.tbins):

    :theta:         [float] - Treecorr np.exp(meanlogr)
    :out:           ([float]x4) - Output of signal e.g., (xi+,xi-,xi+im,x-im). For correlations with only one xi output, (xi,0.,xi_im,0.).
    :err:           ([float]x4) - Same but for sqrt(var).
    :chi2:          ([float]x4) - Same but for chi^2 if using jk or sim covariance.
    :conj:          bool - Conjugate calculation.

    """

    maska=catalog.CatalogMethods.check_mask(cata.coadd,maska)
    jkmask=catalog.CatalogMethods.check_mask(cata.coadd,jkmask)

    maska0=maska&jkmask

    if wa is None:
      wa=np.ones(len(cata.coadd))

    e1,e2,w,ms=lin.linear_methods.get_lin_e_w_ms(cata,xi=True,mock=mock,mask=maska0,w1=wa)

    if catb is None:
      if corr not in ['GG','NN','KK']:
        raise UseError('Must supply both cata,catb for NG,NK correlations.')

    if ga is not None:
      e1=getattr(cata,ga+'1')[maska]
      e2=getattr(cata,ga+'2')[maska]
    else:
      ga='e'
    if catb is None:
      gb=ga
    if conj:
      e2=-e2

    if (corr=='GG')|((catb!=None)&(corr=='KG')):
      catxa=treecorr.Catalog(g1=e1, g2=e2, w=w, ra=cata.ra[maska0], dec=cata.dec[maska0], ra_units='deg', dec_units='deg')
      catma=treecorr.Catalog(k=ms, w=w, ra=cata.ra[maska0], dec=cata.dec[maska0], ra_units='deg', dec_units='deg')

    elif (corr=='NN')|((catb!=None)&(corr in ['NG','NK'])):
      catxa=treecorr.Catalog(w=w, ra=cata.ra[maska0], dec=cata.dec[maska0], ra_units='deg', dec_units='deg')
      if ran:
        catra=treecorr.Catalog(w=w, ra=cata.ran_ra[maska0], dec=cata.ran_dec[maska0], ra_units='deg', dec_units='deg')

    elif corr=='KK':
      if k is None:
        raise UseError('Must specify k for KK correlation.')
      if k not in dir(cata):
        raise UseError('Unknown k field specified.')
      catxa=treecorr.Catalog(k=getattr(cata, k)[maska0], w=w, ra=cata.ra[maska0], dec=cata.dec[maska0], ra_units='deg', dec_units='deg')

    if catb is not None:

      maskb=catalog.CatalogMethods.check_mask(catb.coadd,maskb)

      if wb is None:
        wb=np.ones(len(catb.coadd))

      e1,e2,w,ms=lin.linear_methods.get_lin_e_w_ms(catb,xi=True,mock=mock,mask=maskb,w1=wb)

      if gb is not None:
        e1=getattr(catb,gb+'1')[maskb]
        e2=getattr(catb,gb+'2')[maskb]
      else:
        gb='e'
      if conj:
        e2=-e2

      if corr in ['GG','NG','KG']:
        catxb=treecorr.Catalog(g1=e1, g2=e2, w=w, ra=catb.ra[maskb], dec=catb.dec[maskb], ra_units='deg', dec_units='deg')
        catmb=treecorr.Catalog(k=ms, w=w, ra=catb.ra[maskb], dec=catb.dec[maskb], ra_units='deg', dec_units='deg')
      elif corr=='NN':
        catxb=treecorr.Catalog(w=w, ra=catb.ra[maskb], dec=catb.dec[maskb], ra_units='deg', dec_units='deg')
        if ran:
          catrb=treecorr.Catalog(w=w, ra=catb.ran_ra[maskb], dec=catb.ran_dec[maskb], ra_units='deg', dec_units='deg')
      elif corr in ['KK','NK']:
        if k is None:
          raise UseError('Must specify k for KK correlation.')
        if k not in dir(catb):
          raise UseError('Unknown k field specified.')
        catxb=treecorr.Catalog(k=getattr(catb, k)[maskb], w=w, ra=catb.ra[maskb], dec=catb.dec[maskb], ra_units='deg', dec_units='deg')

    xim=None
    xip_im=None
    xim_im=None
    ximerr=None
    xiperr_im=None
    ximerr_im=None
    if corr=='GG':
      gg = treecorr.GGCorrelation(nbins=cata.tbins, min_sep=cata.sep[0], max_sep=cata.sep[1], sep_units='arcmin',bin_slop=cata.slop,verbose=0)
      kk = treecorr.KKCorrelation(nbins=cata.tbins, min_sep=cata.sep[0], max_sep=cata.sep[1], sep_units='arcmin',bin_slop=cata.slop,verbose=0)
      if catb is None:
        gg.process(catxa)
        kk.process(catma)
      else:
        gg.process(catxa,catxb)
        kk.process(catma,catmb)

      xip = gg.xip/kk.xi
      xim = gg.xim/kk.xi
      xiperr = ximerr = np.sqrt(gg.varxi)
      xip_im = gg.xip_im/kk.xi
      xim_im = gg.xim_im/kk.xi
      theta = np.exp(gg.meanlogr)

    elif corr=='NN':
      nn = treecorr.NNCorrelation(nbins=cata.tbins, min_sep=cata.sep[0], max_sep=cata.sep[1], sep_units='arcmin',bin_slop=cata.slop,verbose=0)
      if ran:
        nr = treecorr.NNCorrelation(nbins=cata.tbins, min_sep=cata.sep[0], max_sep=cata.sep[1], sep_units='arcmin',bin_slop=cata.slop,verbose=0)
        rr = treecorr.NNCorrelation(nbins=cata.tbins, min_sep=cata.sep[0], max_sep=cata.sep[1], sep_units='arcmin',bin_slop=cata.slop,verbose=0)

      if catb is None:
        nn.process(catxa)
        xip=nn.npairs
        xiperr=np.sqrt(nn.npairs)
        if ran:
          nr.process(catxa,catra)
          rr.process(catra)
        xip,xiperr=nn.calculateXi(rr,nr)
        xiperr=np.sqrt(xiperr)
      else:
        rn = treecorr.NNCorrelation(nbins=cata.tbins, min_sep=cata.sep[0], max_sep=cata.sep[1], sep_units='arcmin',bin_slop=cata.slop,verbose=0)
        nn.process(catxa,catxb)
        xip=nn.npairs
        xiperr=np.sqrt(nn.npairs)
        if ran:
          nr.process(catxa,catrb)
          nr.process(catra,catxb)
          rr.process(catra,catrb)
        xip,xiperr=nn.calculateXi(rr,nr,rn)
        xiperr=np.sqrt(xiperr)
      theta=np.exp(nn.meanlogr)

    elif corr=='KK':

      kk = treecorr.KKCorrelation(nbins=cata.tbins, min_sep=cata.sep[0], max_sep=cata.sep[1], sep_units='arcmin',bin_slop=cata.slop,verbose=0)
      if catb is None:
        kk.process(catxa)
      else:
        kk.process(catxa,catxb)
      xip=kk.xi
      xiperr=np.sqrt(kk.varxi)
      theta=np.exp(kk.meanlogr)

    elif corr=='KG':

      kg = treecorr.KGCorrelation(nbins=cata.tbins, min_sep=cata.sep[0], max_sep=cata.sep[1], sep_units='arcmin',bin_slop=cata.slop,verbose=0)
      kk = treecorr.KKCorrelation(nbins=cata.tbins, min_sep=cata.sep[0], max_sep=cata.sep[1], sep_units='arcmin',bin_slop=cata.slop,verbose=0)
      kg.process(catxa,catxb)
      kk.process(catxa,catmb)
      xip=kg.xi/kk.xi
      xiperr=np.sqrt(kg.varxi)
      xip_im=kg.xi_im/kk.xi
      theta=np.exp(kg.meanlogr)

    elif corr=='NG':

      ng = treecorr.NGCorrelation(nbins=cata.tbins, min_sep=cata.sep[0], max_sep=cata.sep[1], sep_units='arcmin',bin_slop=cata.slop,verbose=0)
      nk = treecorr.NKCorrelation(nbins=cata.tbins, min_sep=cata.sep[0], max_sep=cata.sep[1], sep_units='arcmin',bin_slop=cata.slop,verbose=0)
      ng.process(catxa,catxb)
      nk.process(catxa,catmb)
      xip=ng.xi/nk.xi
      xiperr=np.sqrt(ng.varxi)
      xip_im=ng.xi_im/nk.xi
      if ran:
        rg = treecorr.NGCorrelation(nbins=cata.tbins, min_sep=cata.sep[0], max_sep=cata.sep[1], sep_units='arcmin',bin_slop=cata.slop,verbose=0)
        rk = treecorr.NKCorrelation(nbins=cata.tbins, min_sep=cata.sep[0], max_sep=cata.sep[1], sep_units='arcmin',bin_slop=cata.slop,verbose=0)
        rg.process(catra,catxb)
        rk.process(catra,catmb)
        xip,xip_im,xiperr=ng.calculateXi(rg)
        tmpa,tmp=nk.calculateXi(rk)
        if np.sum(tmpa)==0:
          tmpa=np.ones(len(xip))
        xip/=tmpa
        xiperr=np.sqrt(xiperr)
        xip_im/=tmpa
      theta=np.exp(ng.meanlogr)

    elif corr=='NK':

      nk = treecorr.NKCorrelation(nbins=cata.tbins, min_sep=cata.sep[0], max_sep=cata.sep[1], sep_units='arcmin',bin_slop=cata.slop,verbose=0)
      nk.process(catxa,catxb)
      xip=nk.xi
      xiperr=np.sqrt(nk.varxi)
      if ran:
        rk = treecorr.NKCorrelation(nbins=cata.tbins, min_sep=cata.sep[0], max_sep=cata.sep[1], sep_units='arcmin',bin_slop=cata.slop,verbose=0)
        rk.process(catra,catxb)
        xip,xiperr=nk.calculateXi(rk)
        xiperr=np.sqrt(xiperr)
      theta=np.exp(nk.meanlogr)

    out=[xip,xim,xip_im,xim_im]
    err=[xiperr,ximerr,xiperr,ximerr]
    chi2=[0.,0.,0.,0.]

    if erron:
      kwargs={'catb':catb,'k':k,'corr':corr,'maska':maska,'maskb':maskb,'wa':wa,'wb':wb,'ran':ran}
      if catb is None:
        if corr in ['KK','NK','KG']:
          label='xi_2pt_'+cata.name+'_'+k+'_'+corr+'_'+label0
        else:
          label='xi_2pt_'+cata.name+'_'+corr+'_'+label0
      else:
        if corr in ['KK','NK','KG']:
          label='xi_2pt_'+cata.name+'-'+catb.name+'_'+k+'_'+corr+'_'+label0
        else:
          label='xi_2pt_'+cata.name+'-'+catb.name+'_'+corr+'_'+label0
      if cata.calc_err=='jk':
        err,chi2=jackknife_methods.jk(cata,xi_2pt.xi_2pt,[xip,xim,xip_im,xim_im],label,**kwargs)
      elif cata.calc_err=='mock':
        ggperr,ggmerr,chi2p,chi2m,ceerr,cberr,cechi2,cbchi2=BCC_Methods.jk_iter_xi(cat,ggp,ggm,ce,cb,mask,w,cosebi=cosebi,parallel=parallel)

    if plot:
      fig.plot_methods.fig_create_xi(cata,catb,corr,theta,out,err,k,ga,gb)

    return theta,out,err,chi2

  @staticmethod
  def psf_alpha(cat,mask=None):

    mask=catalog.CatalogMethods.check_mask(cat.coadd,mask)

    theta,gpout,gperr,chi2=xi_2pt.xi_2pt(cat,corr='GG',ga='psf',maska=mask,plot=False)
    theta,ppout,pperr,chi2=xi_2pt.xi_2pt(cat,cat,corr='GG',gb='psf',maska=mask,plot=False)
    theta,gpout2,gperr2,chi2=xi_2pt.xi_2pt(cat,corr='GG',ga='psf',maska=mask,plot=False,conj=True)
    theta,ppout2,pperr2,chi2=xi_2pt.xi_2pt(cat,cat,corr='GG',gb='psf',maska=mask,plot=False,conj=True)

    e1,e2=lin.linear_methods.calc_mean_stdev_rms_e(cat,mask=mask,full=False)
    psf1=lin.linear_methods.calc_mean_stdev_rms(cat,'psf1',mask=mask,full=False)
    psf2=lin.linear_methods.calc_mean_stdev_rms(cat,'psf2',mask=mask,full=False)
    e=e1+1j*e2
    psf=psf1+1j*psf2

    a=ppout[0]+1j*ppout[2]-np.abs(e)**2
    b=ppout2[0]+1j*ppout2[2]-e**2
    c=gpout[0]+1j*gpout[2]-np.conj(e)*psf
    d=gpout2[0]+1j*gpout2[2]-e*psf

    alphap=(a*np.conj(c)-np.conj(b)*d)/(np.conj(a)*a-np.conj(b)*b)
    alpham=(np.conj(a)*d-b*np.conj(c))/(np.conj(a)*a-np.conj(b)*b)

    alpha0=xi_2pt.calc_alpha(gpout[0],ppout[0],e1,e2,psf1,psf2)

    txt.write_methods.heading('Xi PSF alpha calculation results',cat,label='xi_alpha',create=True)
    txt.write_methods.write_append('theta  '+str(theta),cat,label='xi_alpha',create=False)
    txt.write_methods.write_append('gp+  '+str(gpout[0]),cat,label='xi_alpha',create=False)
    txt.write_methods.write_append('gp-  '+str(gpout[1]),cat,label='xi_alpha',create=False)
    txt.write_methods.write_append('pp+  '+str(ppout[0]),cat,label='xi_alpha',create=False)
    txt.write_methods.write_append('pp-  '+str(ppout[1]),cat,label='xi_alpha',create=False)
    txt.write_methods.write_append('gp err  '+str(gperr[0]),cat,label='xi_alpha',create=False)
    txt.write_methods.write_append('pp err  '+str(pperr[0]),cat,label='xi_alpha',create=False)
    txt.write_methods.write_append('alpha+ (real)  '+str(alphap.real),cat,label='xi_alpha',create=False)
    txt.write_methods.write_append('alpha+ (imag)  '+str(alphap.imag),cat,label='xi_alpha',create=False)
    txt.write_methods.write_append('alpha- (real)  '+str(alpham.real),cat,label='xi_alpha',create=False)
    txt.write_methods.write_append('alpha- (imag)  '+str(alpham.imag),cat,label='xi_alpha',create=False)
    txt.write_methods.write_append('alpha0  '+str(alpha0),cat,label='xi_alpha',create=False)
    txt.write_methods.write_append('e1  '+str(e1),cat,label='xi_alpha',create=False)
    txt.write_methods.write_append('e2  '+str(e2),cat,label='xi_alpha',create=False)
    txt.write_methods.write_append('psf1  '+str(psf1),cat,label='xi_alpha',create=False)
    txt.write_methods.write_append('psf2  '+str(psf2),cat,label='xi_alpha',create=False)

    fig.plot_methods.fig_create_xi_alpha(cat,theta,gpout,ppout,gperr,pperr,alphap,alpham,alpha0)

    return 

  @staticmethod
  def calc_alpha(gp,pp,e1,e2,psf1,psf2):

    return (gp-e1*psf1-e2*psf2)/(pp-psf1**2-psf2**2)    


  @staticmethod
  def create_shear_rm_cat(cat,cat2):
    rm10s=catalog.CatalogStore('y1_rm_shear',setup=False)

    m1,s1,m2,s2=catalog.CatalogMethods.sort(cat.coadd,cat2.coadd)
    rm10s.e1=(cat2.e1[m2])[s2]
    rm10s.e2=(cat2.e2[m2])[s2]
    rm10s.ra=(cat.ra[m1])[s1]
    rm10s.dec=(cat.dec[m1])[s1]
    rm10s.zp=(cat.zp[m1])[s1]
    rm10s.coadd=(cat.coadd[m1])[s1]

    return rm10s



  @staticmethod
  def ia_estimatora(cat,cat2,dlos=100.,rbins=5,rmin=.1,rmax=200.,logr=True,usempi=False,comm=None):

    import numpy.random as rand
    from mpi4py import MPI

    def rote(cat,cat2,i,j):
      # rotate cat2 (j) to cat (i)
      # Pxi=(-cat._y[i],cat._x[i]-cat._z[i],0)
      # jxi=(cat2._y[j]*cat._z[i]-cat2._z[j]*cat._y[i],cat2._z[j]*cat._x[i]-cat2._x[j]*cat._z[i],cat2._x[j]*cat._y[i]-cat2._y[j]*cat._x[i])
      # y=Pxi[0]*cat2._x[j]+Pxi[1]*cat2._y[j]
      # x=Pxi[0]*jxi[0]+Pxi[1]*jxi[1]
      # tdphi=2.*np.arctan2(y,x)
      x=np.sin(cat.ra[i]-cat2.ra[j])*np.cos(cat.dec[i])
      y=np.cos(cat.dec[i])*np.sin(cat2.dec[j])-np.sin(cat.dec[i])*np.cos(cat2.dec[j])*np.cos(cat.ra[i]-cat2.ra[j])
      tdphi=2.*np.arctan2(y,x)
      if cat.bs:
        e1=(cat2.e1[j]-cat2.c1[j])*np.cos(tdphi)+(cat2.e2[j]-cat2.c2[j])*np.sin(tdphi)
        e2=(cat2.e1[j]-cat2.c1[j])*np.sin(tdphi)-(cat2.e2[j]-cat2.c2[j])*np.cos(tdphi)
      else:
        e1=cat2.e1[j]*np.cos(tdphi)+cat2.e2[j]*np.sin(tdphi)
        e2=cat2.e1[j]*np.sin(tdphi)-cat2.e2[j]*np.cos(tdphi)
      return e1,e2

    def pairs(cat,cat2,i,dlos,r):
      # finds all pairs around cat[i]
      j=np.where(np.abs(cat2.r-cat.r[i])<=dlos)[0]
      sep=physsep(cat,cat2,i,j)
      # print sep
      sepmask=(sep<r[-1])&(sep>r[0])
      if np.sum(sepmask)>0:
        sep=sep[sepmask]
        # print sep
        j=j[sepmask]
        bins=np.digitize(sep,r)-1
      else:
        npairs=np.zeros(len(r)-1)
        sep=np.array([])
        j=np.array([])
        bins=np.array([])
      return j,bins,sep

    def angsep(cat,cat2,i,j):
      dist=np.sqrt((cat._x[i]-cat2._x[j])**2+(cat._y[i]-cat2._y[j])**2+(cat._z[i]-cat2._z[j])**2)/2.
      return 2.*np.arcsin(dist)

    def physsep(cat,cat2,i,j): 
      return np.sqrt(cat.r[i]*cat2.r[j])*angsep(cat,cat2,i,j)

    def ang2xyz(cat):
      if hasattr(cat,'_x'):
        return
      cat._cosdec = np.cos(cat.dec)
      cat._sindec = np.sin(cat.dec)
      cat._cosra = np.cos(cat.ra)
      cat._sinra = np.sin(cat.ra)
      cat._x = cat._cosdec * cat._cosra
      cat._y = cat._cosdec * cat._sinra
      cat._z = cat._sindec
      return

    ang2xyz(cat)
    ang2xyz(cat2)

    if logr:
      r=np.logspace(np.log(rmin),np.log(rmax),rbins+1,base=np.exp(1))
    else:
      r=np.linspace(rmin,rmax,rbins+1)

    npairs=np.zeros(rbins)
    r0=np.zeros(rbins)
    dep=np.zeros(rbins)
    dex=np.zeros(rbins)
    for i in range(len(cat.ra)):
      if i%10000==0:
        print i
      # if i>0:
      #   break
      j,bins,sep=pairs(cat,cat2,i,dlos,r)
      # print sep
      if len(j)==0:
        continue
      et,ex=rote(cat,cat2,i,j)
      for ri in range(rbins):
        mask=bins==ri
        if np.sum(mask)==0:
          continue
        npairs[ri]+=np.sum(mask)
        r0[ri]+=np.sum(sep[mask])
        # print sep[mask],r0[ri]
        dep[ri]+=np.sum(et[mask])
        dex[ri]+=np.sum(ex[mask])

    r0/=npairs
    dep=dep/npairs*2.*dlos
    dex=dex/npairs*2.*dlos

    return r0,dep,dex,npairs

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
    xip0=self.data['shear_xi','xiplus_'+str(i+1)+'_'+str(j+1)]
    xim0=self.data['shear_xi','ximinus_'+str(i+1)+'_'+str(j+1)]
    print xip0,xim0

    if theta is None:
      self.theta=theta0
      self.xip=xip0
      self.xim=xim0
    else:
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

class bandpowers(object):

  def __init__(self, nt=1000,tmin0=1.,tmax0=400.,nell=20,lmin=100.,lmax=3000.,load=True):

    if load==False:

      self.comm = MPI.COMM_WORLD
      self.rank = self.comm.Get_rank()
      self.size = self.comm.Get_size()

    else:

      self.rank=0
      self.size=1

    print self.rank
    print self.size

    self.nt=nt
    self.tmin0=tmin0*np.pi/180./60.
    self.tmax0=tmax0*np.pi/180./60.

    self.nell=nell
    self.lmin=lmin
    self.lmax=lmax

    self.tmin,self.tmax=self.tbounds()

    if load:

      self.Fpa=np.load('Fpa.npy')
      self.Fpb=np.load('Fpb.npy')

      self.Fma=np.load('Fma.npy')
      self.Fmb=np.load('Fmb.npy')

      self.norm=np.load('Mnorm.npy')

      self.Mp=np.load('Mp.npy')
      self.Mm=np.load('Mm.npy')

      self.A0=np.load('A0.npy')
      self.A4=np.load('A4.npy')

    else:

      self._init_integrals()
      np.save('Fpa.npy',self.Fpa)
      np.save('Fpb.npy',self.Fpb)
      np.save('Fma.npy',self.Fma)
      np.save('Fmb.npy',self.Fmb)
      np.save('Mnorm.npy',self.norm)
      np.save('Mp.npy',self.Mp)
      np.save('Mm.npy',self.Mm)
      np.save('A0.npy',self.A0)
      np.save('A4.npy',self.A4)

    self.Fp=self.Fplus()
    self.Fm=self.Fminus()
    self.FpEB=self.FplusEB()
    self.FmEB=self.FminusEB()

  def _init_integrals(self):

    self.Fpa,self.Fpb=self.Fp()
    self.Fma,self.Fmb=self.Fm()
    self.norm=self.Mnorm()

    Mp,Mm=self.M()

    if self.size>1:

      x=np.zeros((self.nt*self.nt))
      y=np.zeros((self.nt*self.nt))

      self.comm.Reduce([Mp, MPI.DOUBLE],[x, MPI.DOUBLE],op=MPI.SUM,root=0)
      self.comm.Reduce([Mm, MPI.DOUBLE],[y, MPI.DOUBLE],op=MPI.SUM,root=0)
      Mp=x
      Mm=y

    if self.rank==0:
      self.Mp=Mp.reshape((self.nt,self.nt))
      self.Mm=Mm.reshape((self.nt,self.nt))

    A0,A4=self.A()

    if self.size>1:

      x=np.zeros((self.nt*self.nell))
      y=np.zeros((self.nt*self.nell))

      self.comm.Reduce([A0, MPI.DOUBLE],[x, MPI.DOUBLE],op=MPI.SUM,root=0)
      self.comm.Reduce([A4, MPI.DOUBLE],[y, MPI.DOUBLE],op=MPI.SUM,root=0)
      A0=x
      A4=y

    if self.rank==0:
      self.A0=A0.reshape((self.nt,self.nell))
      self.A4=A4.reshape((self.nt,self.nell))

    return

  def tbounds(self):

    tmin=np.logspace(np.log(self.tmin0),np.log(self.tmax0),self.nt+1,base=np.exp(1))[:-1]
    tmax=np.logspace(np.log(self.tmin0),np.log(self.tmax0),self.nt+1,base=np.exp(1))[1:]

    return tmin,tmax

  def window_theta_geometric(self,t,i):

    if i>self.nt:
      print 'Index must be less than number theta bins '+str(self.nt)
      return

    return 2.*t/(self.tmax[i]**2-self.tmin[i]**2)

  def tint0(self,ell):

    bessel=self.tmax*scipy.special.jv(1,np.outer(ell,self.tmax))-self.tmin*scipy.special.jv(1,np.outer(ell,self.tmin))
    tint=2.*bessel/(self.tmax**2-self.tmin**2)

    return tint

  def tint4(self,ell):
    
    bessela=(np.outer(self.tmax,ell*ell)-np.outer(8./self.tmax,np.ones(len(ell))))*scipy.special.jv(1,np.outer(self.tmax,ell))
    besselb=(np.outer(self.tmin,ell*ell)-np.outer(8./self.tmin,np.ones(len(ell))))*scipy.special.jv(1,np.outer(self.tmin,ell))
    besselc=8.*ell*(scipy.special.jv(2,np.outer(self.tmax,ell))-scipy.special.jv(2,np.outer(self.tmin,ell)))

    tint=2.*(bessela-besselb-besselc)/(np.outer(self.tmax**2-self.tmin**2,ell*ell))

    return tint.T

  def lm(self,j):

    dl=(self.lmax-self.lmin)/(self.nell)
    return dl*j+self.lmin

  def window_ell_lognormal(self,ell,j):

    sig=np.log(self.lm(j+1.)/self.lm(j))/2.

    return np.exp(-((np.log(ell)-np.log(self.lm(j+.5)))/sig)**2/2.)/np.sqrt(2.*np.pi)/sig

  def Fp(self):

    def func(t,i):
      return self.window_theta_geometric(t,i)

    # Fpa=[scipy.integrate.romberg(func,self.tmin[i],self.tmax[i],args=[i]) for i in range(self.nt)]

    #From analytic integral:
    Fpa=np.ones(self.nt)   

    def func(t,i):
      return self.window_theta_geometric(t,i)*t**2

    # Fpb=[scipy.integrate.romberg(func,self.tmin[i],self.tmax[i],args=[i]) for i in range(self.nt)]

    #From analytic integral:
    Fpb=np.array([(self.tmax[i]**2+self.tmin[i]**2)/2. for i in range(self.nt)])

    return Fpa,Fpb

  def Fm(self):

    def func(t,i):
      return self.window_theta_geometric(t,i)/t**2

    # Fma=[scipy.integrate.romberg(func,self.tmin[i],self.tmax[i],args=[i]) for i in range(self.nt)]

    #From analytic integral:
    Fma=np.array([2.*(np.log(self.tmax[i])-np.log(self.tmin[i]))/(self.tmax[i]**2-self.tmin[i]**2) for i in range(self.nt)])

    def func(t,i):
      return self.window_theta_geometric(t,i)/t**4

    # Fmb=[scipy.integrate.romberg(func,self.tmin[i],self.tmax[i],args=[i]) for i in range(self.nt)]

    #From analytic integral:
    Fmb=np.array([1./(self.tmax[i]**2*self.tmin[i]**2) for i in range(self.nt)])

    return Fma,Fmb

  def Mnorm(self):

    def func(t,i):
      return self.window_theta_geometric(t,i)**2/t

    # return [scipy.integrate.romberg(func,self.tmin[i],self.tmax[i],args=[i]) for i in range(self.nt)]

    return np.array([2./(self.tmax[i]**2-self.tmin[i]**2) for i in range(self.nt)])

  def M(self):
    import time

    def func0(t1,t2,i,j):
      return self.window_theta_geometric(t1,i)*self.window_theta_geometric(t2,j)*(4./t2**2-12.*t1**2/t2**4)*(np.sign(t2-t1)+1)/2.

    def func(i,j):
      # mathematica output of integral - phi is j (k in paper)
      if (i==j):
        return (8.*self.tmax[j]**2*self.tmin[j]**2*(1.-np.log(self.tmax[j]/self.tmin[j]))-2.*self.tmax[j]**4-6.*self.tmin[j]**4)/(self.tmax[j]**3-self.tmax[j]*self.tmin[j]**2)**2
      elif (j>i):
        return 8.*np.log(self.tmax[j]/self.tmin[j])/(self.tmax[j]**2-self.tmin[j]**2)-6.*(self.tmax[i]**2+self.tmin[i]**2)/self.tmax[j]**2/self.tmin[j]**2
      else:
        return 0.

    # def func(i,k):
    #   if k>i:
    #     return 2.0/(self.tmax[i]*self.tmax[i] - self.tmin[i]*self.tmin[i])*(2.0*(self.tmax[i]*self.tmax[i] - self.tmin[i]*self.tmin[i])*np.log(self.tmax[j]/self.tmin[j]) + 3.0/2.0*(np.power(self.tmax[i],4.0) - np.power(self.tmin[i],4.0))*(1.0/self.tmax[j]/self.tmax[j] - 1.0/self.tmin[j]/self.tmin[j]))
    #   elif k==i:
    #     return 2.0/(self.tmax[i]*self.tmax[i] - self.tmin[i]*self.tmin[i])*(-0.5*(self.tmax[k]*self.tmax[k] - self.tmin[k]*self.tmin[k]) - 2.0*self.tmin[i]*self.tmin[i]*np.log(self.tmax[k]/self.tmin[k]) - 3.0/2.0*np.power(self.tmin[i],4.0)*(1.0/self.tmax[k]/self.tmax[k] - 1.0/self.tmin[k]/self.tmin[k]))
    #   else:
    #     return 0.

    # Mp=np.zeros((self.nt*self.nt))
    # start=time.time()
    # for k in range(self.nt):
    #   # if k%self.size!=self.rank:
    #   #   continue
    #   if k>10:
    #     break
    #   print k,time.time()-start
    #   for i in range(self.nt):
    #     Mp[k*self.nt+i]=scipy.integrate.dblquad(func0,self.tmin[i],self.tmax[i],lambda x: self.tmin[k],lambda x: self.tmax[k],args=(k,i))[0]
    #     if i==k:
    #       Mp[k*self.nt+i]+=1.
    #     Mp[k*self.nt+i]/=self.norm[k]

    Mp=np.zeros((self.nt*self.nt))
    for k in range(self.nt):
      for i in range(self.nt):
        Mp[k*self.nt+i]=func(i,k)/self.norm[k]
        if i==k:
          Mp[k*self.nt+i]+=1.

    def func0(t1,t2,i,j):
      return self.window_theta_geometric(t1,i)*self.window_theta_geometric(t2,j)*(4./t1**2-12.*t2**2/t1**4)*(np.sign(t1-t2)+1)/2.

    def func(i,j):
      # mathematica output of integral
      if (i==j):
        return (8.*self.tmax[j]**2*self.tmin[j]**2*(1.-np.log(self.tmax[j]/self.tmin[j]))-2.*self.tmax[j]**4-6.*self.tmin[j]**4)/(self.tmax[j]**3-self.tmax[j]*self.tmin[j]**2)**2
      elif (j<i):
        return 8.*np.log(self.tmax[i]/self.tmin[i])/(self.tmax[i]**2-self.tmin[i]**2)-6.*(self.tmax[j]**2+self.tmin[j]**2)/self.tmax[i]**2/self.tmin[i]**2
      else:
        return 0.

    # def func(i,k):
    #   if k>i:
    #     return 2.0/(self.tmax[i]*self.tmax[i] - self.tmin[i]*self.tmin[i])* (2.0*(self.tmax[k]*self.tmax[k] - self.tmin[k]*self.tmin[k])*np.log(self.tmax[i]/self.tmin[i]) + 3.0/2.0*(np.power(self.tmax[k],4.0) - np.power(self.tmin[k],4.0))* (1.0/self.tmax[i]/self.tmax[i] - 1.0/self.tmin[i]/self.tmin[i]))
    #   elif k==i:
    #     return 2.0/(self.tmax[i]*self.tmax[i] - self.tmin[i]*self.tmin[i]) *(0.5*(-1.0*self.tmax[k]*self.tmax[k] + self.tmin[k]*self.tmin[k]*(4.0 - 3.0*self.tmin[k]*self.tmin[k]/self.tmax[i]/self.tmax[i] - 4.0*np.log(self.tmax[i]/self.tmin[k]))))
    #   else:
    #     return 0.

    # Mm=np.zeros((self.nt*self.nt))
    # for k in range(self.nt):
    #   if k>10:
    #     break
    #   # if k%self.size!=self.rank:
    #   #   continue
    #   print k,time.time()-start
    #   for i in range(self.nt):
    #     Mm[k*self.nt+i]=scipy.integrate.dblquad(func0,self.tmin[i],self.tmax[i],lambda x: self.tmin[k],lambda x: self.tmax[k],args=(k,i))[0]
    #     if i==k:
    #       Mm[k*self.nt+i]+=1.
    #     Mm[k*self.nt+i]/=self.norm[k]

    Mm=np.zeros((self.nt*self.nt))
    for k in range(self.nt):
      for i in range(self.nt):
        Mm[k*self.nt+i]=func(i,k)/self.norm[k]
        if i==k:
          Mm[k*self.nt+i]+=1.

    return Mp,Mm

  def A(self):
    import time
    import src.a_int_cython as a_int

    def func0(t,ell,i,j):
      return self.window_theta_geometric(t,i)*self.window_ell_lognormal(ell,j)*scipy.special.j0(t*ell)*ell

    # after doing int_tmin^tmax dt t*ell*j0(t*ell) = besselpart

    def func(ell,i,j):
      return self.tint0(ell,i)*self.window_ell_lognormal(ell,j)

    A0=np.zeros((self.nell*self.nt))
    A0a=np.zeros((self.nell*self.nt))
    start=time.time()
    for k in range(self.nell):
      if k%self.size!=self.rank:
        continue
      print k,time.time()-start
      for i in range(self.nt):
        # A0a[i*self.nell+k]=2.*scipy.integrate.quad(func,0.,1e6,args=(i,k),limit=2000)[0]/self.norm[i]
        A0[i*self.nell+k]=2.*a_int.A0_integral(float(k),self.lmin,self.lmax,self.nell,self.tmin[i],self.tmax[i],1.e-8,0.,1.e5)/self.norm[i]

    def func0(t,ell,i,j):
      return self.window_theta_geometric(t,i)*self.window_ell_lognormal(ell,j)*scipy.special.jv(4,t*ell)*ell

    # after doing int_tmin^tmax dt t*ell*j4(t*ell) = besselpart

    def func(ell,i,j):
      return self.tint4(ell,i)*self.window_ell_lognormal(ell,j)

    A4=np.zeros((self.nell*self.nt))
    A4a=np.zeros((self.nell*self.nt))
    for k in range(self.nell):
      if k%self.size!=self.rank:
        continue
      print k,time.time()-start
      for i in range(self.nt):
        #A4[i*self.nell+k]=2.*scipy.integrate.quad(func,0.,np.inf,args=(i,k),epsabs=1.e-6, epsrel=1.e-6,limit=2000)[0]/self.norm[i]
        # lims=np.linspace(0.,1.e6,1001)
        # for x in range(100):
        #   A4a[i*self.nell+k]+=2.*scipy.integrate.quad(func,lims[x],lims[x+1],args=(i,k),limit=2000)[0]/self.norm[i]
        A4[i*self.nell+k]+=2.*a_int.A4_integral(float(k),self.lmin,self.lmax,self.nell,self.tmin[i],self.tmax[i],1.e-5,0.,1.e5)/self.norm[i]

    return A0,A4

  def Fplus(self):

    IMM=scipy.linalg.inv(np.identity(self.nt)-np.dot(self.Mm,self.Mp))
    AMF=self.A0-np.dot(self.Mm,self.A4)

    return np.dot(IMM,AMF)

  def Fminus(self):

    return self.A4-np.dot(self.Mp,self.Fp)

  def FplusEB(self):

    fpanorm=self.Fpa/np.sqrt(np.sum(self.Fpa*self.Fpa))
    fpbnorm=self.Fpb-self.Fpa*np.sum(self.Fpa*self.Fpb)/np.sum(self.Fpa*self.Fpa)
    fpbnorm/=np.sqrt(np.sum(fpbnorm*fpbnorm))

    IFF=np.identity(self.nt)-np.outer(fpanorm,fpanorm)-np.outer(fpbnorm,fpbnorm)
    IMM=scipy.linalg.inv(np.identity(self.nt)+np.dot(np.dot(self.Mm,self.Mp),IFF))

    fp=np.dot(IMM,self.A0)
    fp-=np.outer(fpanorm,np.dot(fpanorm,fp))
    fp-=np.outer(fpbnorm,np.dot(fpbnorm,fp))

    return fp

  def FminusEB(self):

    return np.dot(self.Mp,self.FpEB)

  def Wplus(self,ell):

    if hasattr(ell, "__len__"):
      return np.dot(self.tint0(ell),self.Fp)/2.+np.dot(self.tint4(ell),self.Fm)/2.
    else:
      return (np.dot(self.tint0(np.array([ell])),self.Fp)/2.+np.dot(self.tint4(np.array([ell])),self.Fm)/2.)

  def Wminus(self,ell):

    if hasattr(ell, "__len__"):
      return np.dot(self.tint0(ell),self.Fp)/2.-np.dot(self.tint4(ell),self.Fm)/2.
    else:
      return (np.dot(self.tint0(np.array([ell])),self.Fp)/2.-np.dot(self.tint4(np.array([ell])),self.Fm)/2.)

  def WplusEB(self,ell):

    if hasattr(ell, "__len__"):
      return np.dot(self.tint0(ell),self.FpEB)/2.+np.dot(self.tint4(ell),self.FmEB)/2.
    else:
      return (np.dot(self.tint0(np.array([ell])),self.FpEB)/2.+np.dot(self.tint4(np.array([ell])),self.FmEB)/2.)

  def WminusEB(self,ell):

    if hasattr(ell, "__len__"):
      return np.dot(self.tint0(ell),self.FpEB)/2.-np.dot(self.tint4(ell),self.FmEB)/2.
    else:
      return (np.dot(self.tint0(np.array([ell])),self.FpEB)/2.-np.dot(self.tint4(np.array([ell])),self.FmEB)/2.)

  def bandpowers(self,xip,xim,xiperr=None,ximerr=None,blind=True):

    if len(np.shape(xip))==1:
      cplus=np.zeros(self.nell)
      cminus=np.zeros(self.nell)
      cpluserr=np.zeros(self.nell)
      cminuserr=np.zeros(self.nell)
    else:
      cplus=np.zeros((len(xip),len(xip),self.nell))
      cminus=np.zeros((len(xip),len(xip),self.nell))
      cpluserr=np.zeros((len(xip),len(xip),self.nell))
      cminuserr=np.zeros((len(xip),len(xip),self.nell))

    if not blind:
      cplus=np.dot(xip,self.Fp)/2.+np.dot(xim,self.Fm)/2.
    cminus=np.dot(xip,self.Fp)/2.-np.dot(xim,self.Fm)/2.
    if xiperr is not None:
      cpluserr=np.dot(xip**2,self.Fp**2)/4.+np.dot(xim**2,self.Fm**2)/4.
      cminuserr=np.dot(xip**2,self.Fp**2)/4.-np.dot(xim**2,self.Fm**2)/4.

    return cplus,cminus,np.sqrt(cpluserr),np.sqrt(cminuserr)

  def bandpowersEB(self,xip,xim,xiperr=None,ximerr=None,blind=True):

    if len(np.shape(xip))==1:
      cplus=np.zeros(self.nell)
      cminus=np.zeros(self.nell)
      cpluserr=np.zeros(self.nell)
      cminuserr=np.zeros(self.nell)
    else:
      cplus=np.zeros((len(xip),len(xip),self.nell))
      cminus=np.zeros((len(xip),len(xip),self.nell))
      cpluserr=np.zeros((len(xip),len(xip),self.nell))
      cminuserr=np.zeros((len(xip),len(xip),self.nell))

    if not blind:
      cplus=np.dot(xip,self.FpEB)/2.+np.dot(xim,self.FmEB)/2.
    cminus=np.dot(xip,self.FpEB)/2.-np.dot(xim,self.FmEB)/2.

    if xiperr is not None:
      cpluserr=np.dot(xip**2,self.FpEB**2)/4.+np.dot(xim**2,self.FmEB**2)/4.
      cminuserr=np.dot(xip**2,self.FpEB**2)/4.-np.dot(xim**2,self.FmEB**2)/4.

    return cplus,cminus,np.sqrt(cpluserr),np.sqrt(cminuserr)

  def bandpowers_theory(self,Pe):

    def funcp(ell,j):
      return self.Wplus(ell)[0,j]*Pe(ell)*ell/2./np.pi

    def funcm(ell,j):

      return self.Wminus(ell)[j]*Pe(ell)*ell/2./np.pi

    cplus=[scipy.integrate.quad(funcp,0.,1e6,args=(j),limit=2000)[0] for j in range(self.nell)]
    cminus=[scipy.integrate.quad(funcm,0.,1e6,args=(j),limit=2000)[0] for j in range(self.nell)]

    return cplus,cminus

  def bandpowers_theory(self,Pe):


    return cplus,cminus

# for i in range(20):
#   plt.figure(figsize=(8,2))
#   plt.plot((bp.tmin+bp.tmax)/2./np.pi*180.*60.,np.zeros(bp.nt),color='k',label='')
#   plt.plot((bp.tmin+bp.tmax)/2./np.pi*180.*60.,bp.Fp[:,i]/np.max(bp.Fp[:,i]),label='plus')
#   plt.plot((bp.tmin+bp.tmax)/2./np.pi*180.*60.,bp.Fm[:,i]/np.max(bp.Fm[:,i]),label='minus')
#   plt.xscale('log')
#   plt.xlim((0.00029088820866572174/np.pi*180.*6[0.,0.11635528346628866/np.pi*180.*60.))
#   plt.ylim((-2.5,2.5))
#   plt.legend(loc='lower left',ncol=1, frameon=False,prop={'size':12},framealpha=0.2)
#   plt.savefig('tmp'+str(i)+'.png')
#   plt.close()

# ell=np.logspace(np.log10(1),np.log10(4000),1000)
# tmp=bp.WplusEB(ell)
# for i in range(bp.nell):
#   plt.figure(figsize=(8,2))
#   plt.plot(ell,np.zeros(len(ell)),color='k',label='')
#   plt.plot(ell,tmp[:,i]/np.max(tmp[:,i]),label='plus')
#   #plt.plot(ell,bp.Wminus(ell)[i]/np.max(bp.Wminus(ell)[i]),label='minus')
#   plt.xscale('log')
#   plt.xlim((1,4000))
#   plt.ylim((-2.5,2.5))
#   plt.legend(loc='lower left',ncol=1, frameon=False,prop={'size':12},framealpha=0.2)
#   plt.savefig('tmp'+str(i)+'.png')
#   plt.close()

class corr_methods(object):

  @staticmethod
  def get_jk_cov(xi,func):

    xi1=np.zeros_like(xi[0,:,:])
    for i in xrange(len(xi1)):
      xi1[i,:]=func(np.sum(xi,axis=1)-xi[:,i,:])

    cov=np.zeros((len(xi1[0,:]),len(xi1[0,:])))
    for i in xrange(len(xi1[0,:])):
      for j in xrange(len(xi1[0,:])):
        cov[i,j]=np.sum((xi1[i,:]-np.mean(xi1,axis=0))*(xi1[j,:]-np.mean(xi1,axis=0)))*(len(xi1)-1.)/len(xi1)

    return cov

  @staticmethod
  def proj_corr(xi):
    return xi[0]/xi[1]-xi[2]/xi[3]

class runs(object):

  @staticmethod
  def rm_ia(
    rmdfile=config.redmagicdirnersc+'y1a1_gold_1.0.2b-full_redmapper_v6.4.11_redmagic_highdens_0.5-10_e.fit',
    rmlfile=config.redmagicdirnersc+'y1a1_gold_1.0.2b-full_redmapper_v6.4.11_redmagic_highlum_1.0-04_e.fit',
    name='y1_rm',
    corrtype='dense',
    zlims=None,
    mlims=None,
    dpi=60.,
    slop=0.01,
    bins=10,
    sep=[.5,100.],
    nran=50000000):

    if corrtype!='lum':
      rmd=catalog.CatalogStore(name+'_dense',cattype='gal',cols=['coadd','ra','dec','zp','e1','e2','c1','c2','m','w','mabs'],catfile=rmdfile,release='y1',ranfile=rmdfile[:-5]+'randoms.fit')
      rmd.r=cosmo.chi(rmd.zp)
      try:
        rmd.ran_r=np.load(config.redmagicdirnersc+'dense_ran_r.npy')
      except:
        rmd.ran_r=cosmo.chi(rmd.ran_zp)
        np.save(config.redmagicdirnersc+'dense_ran_r.npy',rmd.ran_r)
      try:
        rmd.reg=np.load(config.redmagicdirnersc+'highdens_regs.npy')
        rmd.ran_reg=np.load(config.redmagicdirnersc+'highdens_ran_regs.npy')
      except:
        print 'need regions files'
        return
      rmd.ran_r=rmd.ran_r[:nran]
      rmd.ran_ra=rmd.ran_ra[:nran]
      rmd.ran_dec=rmd.ran_dec[:nran]
      rmd.ran_zp=rmd.ran_zp[:nran]
      rmd.ran_reg=rmd.ran_reg[:nran]

    if corrtype!='dense':
      rml=catalog.CatalogStore(name+'_lum',cattype='gal',cols=['coadd','ra','dec','zp','e1','e2','c1','c2','m','w','mabs'],catfile=rmlfile,release='y1',ranfile=rmlfile[:-5]+'randoms.fit')
      rml.r=cosmo.chi(rml.zp)
      try:
        rml.ran_r=np.load(config.redmagicdirnersc+'lum_ran_r.npy')
      except:
        rml.ran_r=cosmo.chi(rml.ran_zp)
        np.save(config.redmagicdirnersc+'lum_ran_r.npy',rml.ran_r)
      try:
        rml.reg=np.load(config.redmagicdirnersc+'highlum_regs.npy')
        rml.ran_reg=np.load(config.redmagicdirnersc+'highlum_ran_regs.npy')
      except:
        print 'need regions files'
        return
      rml.ran_r=rml.ran_r[:nran]
      rml.ran_ra=rml.ran_ra[:nran]
      rml.ran_dec=rml.ran_dec[:nran]
      rml.ran_zp=rml.ran_zp[:nran]
      rml.ran_reg=rml.ran_reg[:nran]

    if corrtype=='dense':
      pos=rmd
      shape=rmd
    elif corrtype=='lum':
      pos=rml
      shape=rml
    elif corrtype=='cross1':
      pos=rmd
      shape=rml
    elif corrtype=='cross2':
      pos=rml
      shape=rmd
    else:
      print 'need dense, lum, cross1, or cross2 corrtype'

    maske=shape.e1!=-9999
    maskd=np.ones(len(pos.ra)).astype(bool)
    maskr=np.ones(len(pos.ran_ra)).astype(bool)
    if zlims is not None:
      maske=maske&(shape.zp>zlims[0])&(shape.zp<=zlims[1])
      maskd=maskd&(pos.zp>zlims[0])&(pos.zp<=zlims[1])
      maskr=maskr&(pos.ran_zp>zlims[0])&(pos.ran_zp<=zlims[1])
    if mlims is not None:
      maske=maske&(shape.mabs_3>mlims[0])&(shape.mabs_3<=mlims[1])
      maskd=maskd&(pos.mabs_3>mlims[0])&(pos.mabs_3<=mlims[1])

    cate=treecorr.Catalog(g1=shape.e1[maske]-shape.c1[maske], g2=shape.e2[maske]-shape.c2[maske], w=shape.w[maske], ra=shape.ra[maske], dec=shape.dec[maske], r=shape.r[maske], ra_units='deg', dec_units='deg')
    catm=treecorr.Catalog(k=(1.+shape.m[maske]), w=shape.w[maske], ra=shape.ra[maske], dec=shape.dec[maske], r=shape.r[maske], ra_units='deg', dec_units='deg')

    nreg=int(np.max(pos.reg)+1)
    r=np.zeros((4,nreg,bins))
    xi=np.zeros((4,nreg,bins))
    xi_im=np.zeros((4,nreg,bins))
    weight=np.zeros((4,nreg,bins))
    for i in range(nreg):
      maskdjk=(pos.reg[maskd]==i)
      w=np.ones(len(maskdjk))
      w[~maskdjk]=0.
      maskrjk=(pos.ran_reg[maskr]==i)
      ran_w=np.ones(len(maskrjk))
      ran_w[~maskrjk]=0.

      catd=treecorr.Catalog(ra=pos.ra[maskd], dec=pos.dec[maskd], r=pos.r[maskd], w=w, ra_units='deg', dec_units='deg')
      catr=treecorr.Catalog(ra=pos.ran_ra[maskr], dec=pos.ran_dec[maskr], r=pos.ran_r[maskr], w=ran_w, ra_units='deg', dec_units='deg')

      de = treecorr.NGCorrelation(nbins=bins, min_sep=sep[0], max_sep=sep[1], min_rpar = -dpi, max_rpar = dpi, bin_slop=slop, verbose=0)
      dm = treecorr.NKCorrelation(nbins=bins, min_sep=sep[0], max_sep=sep[1], min_rpar = -dpi, max_rpar = dpi, bin_slop=slop, verbose=0)
      re = treecorr.NGCorrelation(nbins=bins, min_sep=sep[0], max_sep=sep[1], min_rpar = -dpi, max_rpar = dpi, bin_slop=slop, verbose=0)
      rm = treecorr.NKCorrelation(nbins=bins, min_sep=sep[0], max_sep=sep[1], min_rpar = -dpi, max_rpar = dpi, bin_slop=slop, verbose=0)

      de.process_cross(catd,cate,metric='Rperp')
      dm.process_cross(catd,catm,metric='Rperp')
      re.process_cross(catr,cate,metric='Rperp')
      rm.process_cross(catr,catm,metric='Rperp')

      r[:,i,:]=[de.meanr,dm.meanr,re.meanr,rm.meanr]
      weight[:,i,:]=[de.weight,dm.weight,re.weight,rm.weight]
      xi[:,i,:]=[de.xi,dm.xi,re.xi,rm.xi]
      xi_im[:,i,:]=[de.xi_im,dm.xi,re.xi_im,rm.xi]

    np.savetxt('r.txt',r[0,:,:])
    np.savetxt('weight.txt',weight[0,:,:])
    np.savetxt('xi_0.txt',xi[0,:,:])
    np.savetxt('xi_1.txt',xi[1,:,:])
    np.savetxt('xi_2.txt',xi[2,:,:])
    np.savetxt('xi_3.txt',xi[3,:,:])
    r0=np.sum(r[0,:,:],axis=0)/np.sum(weight[0,:,:],axis=0)
    wgp=corr_methods.proj_corr(np.sum(xi,axis=1))*2.*dpi
    wgx=corr_methods.proj_corr(np.sum(xi_im,axis=1))*2.*dpi
    varwgp=np.sqrt(np.diagonal(corr_methods.get_jk_cov(xi,corr_methods.proj_corr)))*2.*dpi
    varwgx=np.sqrt(np.diagonal(corr_methods.get_jk_cov(xi_im,corr_methods.proj_corr)))*2.*dpi
    print r0,wgp,varwgp
    print r0,wgx,varwgx

    if zlims is None:
      zlabel='_zlims_None'
    else:
      zlabel='_zlims_'+'_'+str(zlims[0])+'-'+str(zlims[1])

    if mlims is None:
      mlabel='_mlims_None'
    else:
      mlabel='_mlims_'+'_'+str(mlims[0])+'-'+str(mlims[1])

    label=name+'_'+corrtype+zlabel+mlabel+'_dpi_'+str(dpi)+'_bins_'+str(bins)+'_sep_'+str(sep[0])+'-'+str(sep[1])+'_nran_'+str(nran)+'_jk_'+str(nreg)
    np.savetxt(label+'.txt',np.vstack((r0,wgp,varwgp,wgx,varwgx)).T)
    fig.plot_methods.plot_IA(r0,[wgp,wgx],[varwgp,varwgx],label)

    return

  @staticmethod
  def rmp_ia(
    rmdfile=config.redmagicdirnersc+'y1a1_gold_1.0.2b-full_redmapper_v6.4.11_redmagic_highdens_0.5-10_e.fit',
    rmlfile=config.redmagicdirnersc+'y1a1_gold_1.0.2b-full_redmapper_v6.4.11_redmagic_highdens_0.5-10_e.fit',
    rmpfile=config.redmapperdirnersc+'y1a1_gold_1.0.2b-full_run_redmapper_v6.4.11_lgt5_desformat_catalog.fit',
    name='y1_rmp',
    corrtype='dense',
    zlims=None,
    mlims=None,
    richlims=None,
    dpi=60.,
    slop=0.01,
    bins=10,
    sep=[.5,100.]):

    if corrtype!='lum':
      rmd=catalog.CatalogStore(name+'_dense',cattype='gal',cols=['coadd','ra','dec','zp','e1','e2','c1','c2','m','w','mabs'],catfile=rmdfile,release='y1',ranfile=rmdfile[:-5]+'randoms.fit')
      rmd.r=cosmo.chi(rmd.zp)
      try:
        rmd.ran_r=np.load(config.redmagicdirnersc+label+'_ran_r.npy')
      except:
        np.save(config.redmagicdirnersc+label+'_dense_ran_r.npy',cosmo.chi(rmd.zp))
        rmd.ran_r=np.load(config.redmagicdirnersc+label+'_ran_r.npy')

    if corrtype!='dense':
      rml=catalog.CatalogStore(name+'_lum',cattype='gal',cols=['coadd','ra','dec','zp','e1','e2','c1','c2','m','w','mabs'],catfile=rmlfile,release='y1',ranfile=rmlfile[:-5]+'randoms.fit')
      rml.r=cosmo.chi(rml.zp)
      try:
        rml.ran_r=np.load(config.redmagicdirnersc+label+'_ran_r.npy')
      except:
        np.save(config.redmagicdirnersc+label+'_lum_ran_r.npy',cosmo.chi(rml.zp))
        rml.ran_r=np.load(config.redmagicdirnersc+label+'_ran_r.npy')

      rmp=catalog.CatalogStore(name+'_cluster',cattype='redmapper',cols=None,catfile=rmlfile,release='y1',ranfile=rmlfile[:-5]+'randoms.fit')
      rmp.r=cosmo.chi(rmp.zp)
      try:
        rmp.ran_r=np.load(config.redmapperdirnersc+label+'_ran_r.npy')
      except:
        np.save(config.redmapperdirnersc+label+'_lum_ran_r.npy',cosmo.chi(rmp.zp))
        rmp.ran_r=np.load(config.redmapperdirnersc+label+'_ran_r.npy')

    if corrtype=='dense':
      pos=rmp
      shape=rmd
    elif corrtype=='lum':
      pos=rmp
      shape=rml
    else:
      print 'need dense or lum corrtype'

    maske=shape.e1!=-9999
    maskd=np.ones(len(pos.ra)).astype(bool)
    if zlims is not None:
      maske=maske&(shape.zp>zlims[0])&(shape.zp<=zlims[1])
      maskd=maskd&(pos.zp>zlims[0])&(pos.zp<=zlims[1])
    if mlims is not None:
      maske=maske&(shape.mabs_3>mlims[0])&(shape.mabs_3<=mlims[1])
      maskd=maskd&(pos.mabs_3>mlims[0])&(pos.mabs_3<=mlims[1])
    if richlims is not None:
      maskd=maskd&(pos.rich>richlims[0])&(pos.rich<=richlims[1])

    cate=treecorr.Catalog(g1=shape.e1[maske]-shape.c1[maske], g2=shape.e2[maske]-shape.c2[maske], w=shape.w[maske], ra=shape.ra[maske], dec=shape.dec[maske], r=shape.r[maske], ra_units='deg', dec_units='deg')
    catm=treecorr.Catalog(k=(1.+shape.m[maske]), w=shape.w[maske], ra=shape.ra[maske], dec=shape.dec[maske], r=shape.r[maske], ra_units='deg', dec_units='deg')

    catd=treecorr.Catalog(ra=pos.ra, dec=pos.dec, r=pos.r, ra_units='deg', dec_units='deg')
    catr=treecorr.Catalog(ra=pos.ran_ra, dec=pos.ran_dec, r=pos.ran_r, ra_units='deg', dec_units='deg')

    de = treecorr.NGCorrelation(nbins=bins, min_sep=sep[0], max_sep=sep[1], min_rpar = -dpi, max_rpar = dpi, bin_slop=slop, verbose=0)
    dm = treecorr.NKCorrelation(nbins=bins, min_sep=sep[0], max_sep=sep[1], min_rpar = -dpi, max_rpar = dpi, bin_slop=slop, verbose=0)
    re = treecorr.NGCorrelation(nbins=bins, min_sep=sep[0], max_sep=sep[1], min_rpar = -dpi, max_rpar = dpi, bin_slop=slop, verbose=0)
    rm = treecorr.NKCorrelation(nbins=bins, min_sep=sep[0], max_sep=sep[1], min_rpar = -dpi, max_rpar = dpi, bin_slop=slop, verbose=0)

    de.process(catd,cate,metric='Rperp')
    dm.process(catd,catm,metric='Rperp')
    # re.process(catr,cate,metric='Rperp')
    # rm.process(catr,catm,metric='Rperp')

    wgp=de.xi/dm.xi-re.xi/rm.xi
    wgx=de.xi_im/dm.xi-re.xi_im/rm.xi
    varxi=np.sqrt(de.varxi)

    if zlims is None:
      zlabel='_zlims_None'
    else:
      zlabel='_zlims_'+'_'+str(zlims[0])+'-'+str(zlims[1])

    if mlims is None:
      mlabel='_mlims_None'
    else:
      mlabel='_mlims_'+'_'+str(mlims[0])+'-'+str(mlims[1])

    if richlims is None:
      richlabel='_richlims_None'
    else:
      richlabel='_richlims_'+'_'+str(richlims[0])+'-'+str(richlims[1])

    fig.plot_methods.plot_IA(np.exp(de.meanlogr),[wgp,wgx],[varxi,varxi],name+'_'+corrtype+zlabel+mlabel+'_dpi_'+str(dpi)+'_bins_'+str(bins)+'_sep_'+str(sep[0])+'-'+str(sep[1]))
    return
