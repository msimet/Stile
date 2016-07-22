import numpy as np
import fitsio as fio

import catalog
import config
import fig
import txt
import lin
import corr

class field(object):

  @staticmethod
  def loop_epoch_stuff(loop,maxloop=64,catdir='/share/des/disc2/y1/im3shape/single_band/r/y1v1/complete/epoch/',catname='i3epoch',mask=None,label='',plot=False):
    """
    """
    import glob

    # whisker store
    y0=[]
    x0=[]
    m0=[]
    e0=[]
    w0=[]
    e10=[]
    e20=[]
    psf10=[]
    psf20=[]
    psf0=[]
    # whisker2 store
    y1=[]
    x1=[]
    m1=[]
    w1=[]
    e11=[]
    e21=[]
    e1=[]
    psf11=[]
    psf21=[]
    psf1=[]

    coadd=np.load('coadd.npy')
    nbc0=np.load('/home/troxel/destest/i3nbcv1.npy')
    for ii in range(len(glob.glob(catdir))):
      print maxloop,loop,ii%maxloop
      if ii%maxloop!=loop-1:
        break
      print ii
      i3epoch=catalog.CatalogStore('y1_i3_r_epoch_v1',cutfunc=None,cattype=catname,cols=None,catdir=catdir,release='y1',maxrows=1000000,maxiter=50,exiter=ii)
      i3epoch.wt=True
      i3epoch.bs=True
      x=np.in1d(i3epoch.coadd,coadd,assume_unique=False)
      catalog.CatalogMethods.match_cat(i3epoch,x)
      nbc=nbc0
      a=np.argsort(nbc[:,0])
      mask=np.diff(nbc[a,0])
      mask=mask==0
      mask=~mask
      mask=a[mask]
      nbc=nbc[mask]
      x=np.in1d(nbc[:,0],np.unique(i3epoch.coadd),assume_unique=False)
      nbc=nbc[x,:]
      epocharg=np.argsort(i3epoch.coadd)
      nbcarg=np.argsort(nbc[:,0])
      nbc=nbc[nbcarg,:]
      catalog.CatalogMethods.match_cat(i3epoch,epocharg)
      diff=np.diff(i3epoch.coadd)
      diff=np.where(diff!=0)[0]+1
      diff=np.append([0],diff)
      diff=np.append(diff,[None])

      i3epoch.m=np.zeros(len(i3epoch.coadd))
      i3epoch.c1=np.zeros(len(i3epoch.coadd))
      i3epoch.c2=np.zeros(len(i3epoch.coadd))
      i3epoch.w=np.zeros(len(i3epoch.coadd))
      for i in range(len(diff)-1):
        if i%1000==0:
          print i
        i3epoch.m[diff[i]:diff[i+1]]=nbc[i,1]
        i3epoch.c1[diff[i]:diff[i+1]]=nbc[i,2]
        i3epoch.c2[diff[i]:diff[i+1]]=nbc[i,3]
        i3epoch.w[diff[i]:diff[i+1]]=nbc[i,4]

      tmp=[y0,x0,m0,w0,e10,e20,e0,psf10,psf20,psf0]
      for i,x in enumerate(field.whisker_loop(i3epoch)):
        print 'nums',len(tmp[i]),x,tmp[i],x
        if ii==0:
          tmp[i]=x
        else:
          tmp[i]=np.mean(np.vstack((tmp[i],x)).T,axis=1)
      tmp2=[y1,x1,m1,w1,e11,e21,e1,psf11,psf21,psf1]
      for i,x in enumerate(field.whisker_loopb(i3epoch)):
        if ii==0:
          tmp2[i]=x
        else:
          tmp2[i]=np.mean(np.vstack((tmp2[i],x)).T,axis=1)
  
    np.save('epoch_loop_'+str(loop-1)+'.npy',np.vstack((tmp[0],tmp[1],tmp[2],tmp[3],tmp[4],tmp[5],tmp[6],tmp[7],tmp[8],tmp[9])).T)
    np.save('epoch_loopb_'+str(loop-1)+'.npy',np.vstack((tmp2[0],tmp2[1],tmp2[2],tmp2[3],tmp2[4],tmp2[5],tmp2[6],tmp2[7],tmp2[8],tmp2[9])).T)

    return 

  @staticmethod
  def loop_epoch_stuff_finalise(label='',plot=False):
    """
    """

    pos0=0.5*np.arctan2(e20/m0,e10/m0)
    psfpos0=0.5*np.arctan2(psf20/w0,psf10/w0)
    e0/=m0
    psf0/=w0
    fig.plot_methods.plot_whisker(y0,x0,np.sin(pos0)*e0,np.cos(pos0)*e0,name=i3epoch.name,label='shear'+label,scale=.01,key=r'$\langle e\rangle$')
    fig.plot_methods.plot_whisker(y0,x0,np.sin(psfpos0)*psf0,np.cos(psfpos0)*psf0,name=i3epoch.name,label='psf'+label,scale=.01,key=r'$\langle$ PSF $e\rangle$')

    return

  @staticmethod
  def whisker_loop(cat,mask=None,label='',plot=False):
    """
    Calculate whisker plot for e and psf e over field of view.
    """

    mask=catalog.CatalogMethods.check_mask(cat.coadd,mask)

    if not hasattr(cat, 'ra'):
      cat.ra,cat.dec=field_methods.get_field_pos(cat)

    #x,y=field_methods.get_field_pos(cat)

    cx=field_methods.ccd_centres()[:,1]
    cy=field_methods.ccd_centres()[:,0]

    dc=2048./4.

    x0=[]
    y0=[]
    pos0=[]
    psfpos0=[]
    e0=[]
    m0=[]
    psf0=[]
    pos1=[]
    psfpos1=[]
    e1=[]
    psf1=[]
    for i in range(len(cx)):
      if (i==1)|(i==30)|(i==60):
        continue
      print 'chip',i
      #pos1=2.*(cat.pos[mask&(cat.ccd==i)]-np.pi/2.)
      mask0=mask&(cat.ccd==i)
      e1=cat.e1[mask0]
      e2=cat.e2[mask0]
      if cat.bs:
        e1-=cat.c1[mask0]
        e2-=cat.c2[mask0]
        m=cat.m[mask0]
      if cat.wt:
        w=cat.w[mask0]
      else:
        w=np.ones(np.sum(mask0))
      psf1=cat.psf1[mask0]
      psf2=cat.psf2[mask0]
      x1=cat.row[mask0]
      y1=cat.col[mask0]
      for j in xrange(4):
        for k in xrange(8):
          x0=np.append(x0,cx[i]-field_methods.ccdx/2.+(j+.5)*field_methods.ccdx/8.)
          y0=np.append(y0,cy[i]-field_methods.ccdy/2.+(k+.5)*field_methods.ccdy/4.)
          mask1=(x1>k*dc)&(x1<=(k+1.)*dc)&(y1>j*dc)&(y1<=(j+1.)*dc)
          e10=np.sum(e1[mask1]*w[mask1])
          e20=np.sum(e2[mask1]*w[mask1])
          psf10=np.sum(psf1[mask1]*w[mask1])
          psf20=np.sum(psf2[mask1]*w[mask1])
          w0=np.sum(w[mask1])
          e0=np.sum(np.sqrt(e1[mask1]**2.+e2[mask1]**2.)*w[mask1])
          if cat.bs:
            m0=np.sum((1.+m[mask1])*w[mask1])
          else:
            m0=1.
          psf0=np.sum(np.sqrt(psf1[mask1]**2.+psf2[mask1]**2.)*w[mask1])

    return y0,x0,m0,w0,e10,e20,e0,psf10,psf20,psf0

  @staticmethod
  def whisker_loopb(cat,mask=None,label='',plot=False):
    """
    Calculate whisker plot for e and psf e over field of view.
    """

    mask=catalog.CatalogMethods.check_mask(cat.coadd,mask)

    if not hasattr(cat, 'ra'):
      cat.ra,cat.dec=field_methods.get_field_pos(cat)

    #x,y=field_methods.get_field_pos(cat)

    cx=field_methods.ccd_centres()[:,1]
    cy=field_methods.ccd_centres()[:,0]

    dc=2048./4.

    x0=[]
    y0=[]
    pos0=[]
    psfpos0=[]
    e0=[]
    m0=[]
    psf0=[]
    pos1=[]
    psfpos1=[]
    e1=[]
    psf1=[]
    for i in range(len(cx)):
      if (i==1)|(i==30)|(i==60):
        continue
      print 'chip',i
      #pos1=2.*(cat.pos[mask&(cat.ccd==i)]-np.pi/2.)
      mask0=mask&(cat.ccd==i)
      e1=cat.e1[mask0]
      e2=cat.e2[mask0]
      if cat.bs:
        e1-=cat.c1[mask0]
        e2-=cat.c2[mask0]
        m=cat.m[mask0]
      if cat.wt:
        w=cat.w[mask0]
      else:
        w=np.ones(np.sum(mask0))
      psf1=cat.psf1[mask0]
      psf2=cat.psf2[mask0]
      x1=cat.row[mask0]
      y1=cat.col[mask0]
      for j in xrange(40):
        for k in xrange(80):
          x0=np.append(x0,cx[i]-field_methods.ccdx/2.+(j+.5)*field_methods.ccdx/8.)
          y0=np.append(y0,cy[i]-field_methods.ccdy/2.+(k+.5)*field_methods.ccdy/4.)
          mask1=(x1>k*dc)&(x1<=(k+1.)*dc)&(y1>j*dc)&(y1<=(j+1.)*dc)
          e10=np.sum(e1[mask1]*w[mask1])
          e20=np.sum(e2[mask1]*w[mask1])
          psf10=np.sum(psf1[mask1]*w[mask1])
          psf20=np.sum(psf2[mask1]*w[mask1])
          w0=np.sum(w[mask1])
          e0=np.sum(np.sqrt(e1[mask1]**2.+e2[mask1]**2.)*w[mask1])
          if cat.bs:
            m0=np.sum((1.+m[mask1])*w[mask1])
          else:
            m0=1.
          psf0=np.sum(np.sqrt(psf1[mask1]**2.+psf2[mask1]**2.)*w[mask1])

    return y0,x0,m0,w0,e10,e20,e0,psf10,psf20,psf0

  @staticmethod
  def whisker(cat,mask=None,label='',plot=False):
    """
    Calculate whisker plot for e and psf e over field of view.
    """

    mask=catalog.CatalogMethods.check_mask(cat.coadd,mask)

    if not hasattr(cat, 'ra'):
      cat.ra,cat.dec=field_methods.get_field_pos(cat)

    #x,y=field_methods.get_field_pos(cat)

    cx=field_methods.ccd_centres()[:,1]
    cy=field_methods.ccd_centres()[:,0]

    dc=2048./4.

    x0=[]
    y0=[]
    pos0=[]
    psfpos0=[]
    e0=[]
    m0=[]
    psf0=[]
    pos1=[]
    psfpos1=[]
    e1=[]
    psf1=[]
    for i in range(len(cx)):
      if (i==1)|(i==30)|(i==60):
        continue
      print 'chip',i
      #pos1=2.*(cat.pos[mask&(cat.ccd==i)]-np.pi/2.)
      mask0=mask&(cat.ccd==i)
      e1=cat.e1[mask0]
      e2=cat.e2[mask0]
      if cat.bs:
        e1-=cat.c1[mask0]
        e2-=cat.c2[mask0]
        m=cat.m[mask0]
      if cat.wt:
        w=cat.w[mask0]
      else:
        w=np.ones(np.sum(mask0))
      psf1=cat.psf1[mask0]
      psf2=cat.psf2[mask0]
      x1=cat.row[mask0]
      y1=cat.col[mask0]
      # pos1=np.append(pos1,np.mean(0.5*np.arctan2(e2,e1)))
      # psfpos1=np.append(psfpos1,np.mean(0.5*np.arctan2(psf2,psf1)))
      # e1=np.append(e1,np.mean(np.sqrt(e1**2.+e2**2.)))
      # psf1=np.append(psf1,np.mean(np.sqrt(psf1**2.+psf2**2.)))
      for j in xrange(4):
        for k in xrange(8):
          x0=np.append(x0,cx[i]-field_methods.ccdx/2.+(j+.5)*field_methods.ccdx/8.)
          y0=np.append(y0,cy[i]-field_methods.ccdy/2.+(k+.5)*field_methods.ccdy/4.)
          mask1=(x1>k*dc)&(x1<=(k+1.)*dc)&(y1>j*dc)&(y1<=(j+1.)*dc)
          pos0=np.append(pos0,0.5*np.arctan2(np.average(e2[mask1],weights=w[mask1]),np.average(e1[mask1],weights=w[mask1])))
          psfpos0=np.append(psfpos0,0.5*np.arctan2(np.average(psf2[mask1],weights=w[mask1]),np.average(psf1[mask1],weights=w[mask1])))
          e0=np.append(e0,np.average(np.sqrt(e1[mask1]**2.+e2[mask1]**2.),weights=w[mask1]))
          if cat.bs:
            m0=np.append(m0,np.average(1.+m[mask1]))
          else:
            m0=np.append(m0,1.)
          psf0=np.append(psf0,np.average(np.sqrt(psf1[mask1]**2.+psf2[mask1]**2.),weights=w[mask1]))

    # fig.plot_methods.plot_whisker(cy,cx,np.sin(pos1)*e1,np.cos(pos1)*e1,name=cat.name,label='shear2',scale=.01,key=r'$\langle e\rangle$')
    # fig.plot_methods.plot_whisker(cy,cx,np.sin(psfpos1)*psf1,np.cos(psfpos1)*psf1,name=cat.name,label='psf2',scale=.01,key=r'$\langle$ PSF $e\rangle$')
    if plot:
      fig.plot_methods.plot_whisker(y0,x0,np.sin(pos0)*e0/m0,np.cos(pos0)*e0/m0,name=cat.name,label='shear'+label,scale=.01,key=r'$\langle e\rangle$')
      fig.plot_methods.plot_whisker(y0,x0,np.sin(psfpos0)*psf0,np.cos(psfpos0)*psf0,name=cat.name,label='psf'+label,scale=.01,key=r'$\langle$ PSF $e\rangle$')

    return y0,x0,m0,pos0,e0,psfpos0,psf0

  @staticmethod
  def whisker_chip(cat,mask=None):
    """
    Calculate whisker plot for e and psf e over each chip.
    """

    mask=catalog.CatalogMethods.check_mask(cat.coadd,mask)

    #x,y=field_methods.get_field_pos(cat)

    cx=field_methods.ccd_centres()[:,1]
    cy=field_methods.ccd_centres()[:,0]

    dc=2048./20.

    e1=cat.e1[mask]
    e2=cat.e2[mask]
    psf1=cat.psf1_exp[mask]
    psf2=cat.psf2_exp[mask]
    x1=cat.row[mask]
    y1=cat.col[mask]
    for i in xrange(len(cx)):
      print 'chip', i
      x0=[]
      y0=[]
      pos0=[]
      psfpos0=[]
      e0=[]
      psf0=[]
      for j in xrange(20):
        for k in xrange(40):
          x0=np.append(x0,j*dc)
          y0=np.append(y0,k*dc)
          mask1=(x1>k*dc)&(x1<=(k+1.)*dc)&(y1>j*dc)&(y1<=(j+1.)*dc)
          pos0=np.append(pos0,0.5*np.arctan2(np.mean(e2[mask1]),np.mean(e1[mask1])))
          psfpos0=np.append(psfpos0,0.5*np.arctan2(np.mean(psf2[mask1]),np.mean(psf1[mask1])))
          e0=np.append(e0,np.sqrt(np.mean(e1[mask1])**2.+np.mean(e2[mask1])**2.))
          psf0=np.append(psf0,np.sqrt(np.mean(psf1[mask1])**2.+np.mean(psf2[mask1])**2.))

      fig.plot_methods.plot_whisker(y0,x0,np.sin(pos0)*e0,np.cos(pos0)*e0,name=cat.name,label='chip_'+str(i)+'_shear',scale=.01,key=r'$\langle e\rangle$',chip=True)
      fig.plot_methods.plot_whisker(y0,x0,np.sin(psfpos0)*psf0,np.cos(psfpos0)*psf0,name=cat.name,label='chip_'+str(i)+'_psf',scale=.01,key=r'$\langle$ PSF $e\rangle$',chip=True)

    return

  @staticmethod
  def footprint(cat,mask=None):
    """
    Calculate and plot object number density over field of view.
    """

    mask=catalog.CatalogMethods.check_mask(cat.coadd,mask)

    if not hasattr(cat, 'fx'):
      cat.fx,cat.fy=field_methods.get_field_pos(cat)

    fig.plot_methods.plot_field_footprint(cat.fx,cat.fy,cat.name,label='field',bins=1)

    return

  @staticmethod
  def loop_submit_sp():
    from popen2 import popen2
    import subprocess as sp
    import time

    for i in range(40):
      if i<20:
        node='compute-0-17.local'
      else:
        node='compute-0-19.local'

      p = sp.Popen('qsub', shell=True, bufsize=1, stdin=sp.PIPE, stdout=sp.PIPE, close_fds=True)
      output,input = p.stdout, p.stdin

      job_string = """#!/bin/bash
      #PBS -l nodes=%s:ppn=1
      #PBS -l walltime=48:00:00
      #PBS -N sp_%s
      #PBS -o sp_%s.log
      #PBS -j oe
      #PBS -m abe 
      #PBS -M michael.troxel@manchester.ac.uk
      module use /home/zuntz/modules/module-files
      module load python
      module use /etc/modulefiles/
      cd $PBS_O_WORKDIR
      python testsuite.py 3 %s""" % (node,str(i),str(i),str(i))    

      output,outputerr=p.communicate(input=job_string)

      time.sleep(0.1)

    return


  @staticmethod
  def build_special_points(chunk):
    """
    Used to build parts of catalog of special points.
    """

    import re

    dchunk=int(fio.FITS(config.wcsfile)[-1].get_nrows())/40
    ia=dchunk*chunk
    print ia
    ib=dchunk*(chunk+1)
    if chunk==39:
      ib=int(fio.FITS(config.wcsfile)[-1].get_nrows()) 
    print ib

    tmp=fio.FITS(config.wcsfile)[-1][ia:ib]

    with open(config.y1blacklist) as f:
      lines = f.readlines()
    blexp=[]
    blccd=[]
    for line in lines:
      blexp=np.append(blexp,int(re.compile('\w+').findall(line)[1][6:]))
      blccd=np.append(blccd,int(re.compile('\w+').findall(line)[2]))

    image=np.empty(tmp.shape, dtype=tmp.dtype.descr + [('naxis1',int)]+[('naxis2',int)])
    for name in tmp.dtype.names:
      image[name]=tmp[name]

    image['naxis1']=np.ones(len(image))*2048
    image['naxis2']=np.ones(len(image))*4096

    for i in range(ib-ia):
      if image['expnum'][i] in blexp:
        if image['ccdnum'][i] in blccd[blexp==image['expnum'][i]]:
          continue
      print i
      print str(image['expnum'][i])+' '+str(image['ccdnum'][i])
      line=str(i)+' '+str(image['expnum'][i])+' '+str(image['ccdnum'][i])+' '
      radec=field_methods.translate_to_wcs([[1024,0,2048,0,2048],[2048,0,4096,0,4096]],image[i])
      # if field_methods.get_coadd_tile(radec[0],radec[1],tiles=tiles) in image['tilename'][i]:
      for i in range(5):
        line+=str(radec[0][i])+' '+str(radec[1][i])+' '

      with open('y1a1_special_points_'+str(chunk)+'.txt','a') as f:
        f.write(line+'\n')

    return

  @staticmethod
  def build_special_points_fits(sp=None): 
    """
    Combines parts of special points catalog into single fits catalog.
    """

    import fitsio as fio

    name=['center','ll','ul','lr','ur']
    tiles=fio.FITS(config.coaddtiles)[-1].read()
    wcs=fio.FITS(config.wcsfile)[-1].read()
    a=np.sort(np.unique(wcs['expnum']))
    b=np.sort(np.unique(wcs['ccdnum']))-1
    store=np.empty((len(a)*len(b)*5),dtype=[('exposure',int)]+[('ccd',int)]+[('type',int)]+[('ra','f8')]+[('dec','f8')])
    print len(store)
    # for i in range(len(a)):
    #   store['exposure'][i*len(b):(i+1)*len(b)]=a[i]
    #   for j in range(len(b)):
    #     store['ccd'][i*len(b)+j]=b[j]

    if sp is None:
      for i in range(40):
        print i
        tmp=np.genfromtxt('y1a1_special_points_'+str(i)+'.txt',names=['index','exposure','ccd','racenter','deccenter','rall','decll','raul','decul','ralr','declr','raur','decur'])
        if i==0:
          sp=tmp
        else:
          sp=np.append(sp,tmp)
        a=np.argsort(sp,order=('exposure','ccd'))
        sp=sp[a]

    ind0=-1
    for i in range(len(sp)):
      if i>0:
        if (sp['ccd'][i]==sp['ccd'][i-1])&(sp['exposure'][i]==sp['exposure'][i-1]):
          continue
      ind=[]
      for j in range(100):
        if i+j+1==len(sp):
          break
        if (sp['ccd'][i]==sp['ccd'][i+j+1])&(sp['exposure'][i]==sp['exposure'][i+j+1]):
          ind.append(j)
        else:
          break
      for k in range(5):
        ind0+=1
        if ind0==len(store):
          print i,len(sp)
          break
        store['exposure'][ind0]=sp['exposure'][i]
        store['ccd'][ind0]=sp['ccd'][i]
        store['type'][ind0]=k
        store['ra'][ind0]=np.mean(sp['ra'+name[k]][i:i+j+1])
        store['dec'][ind0]=np.mean(sp['dec'+name[k]][i:i+j+1])

    # for i in range(len(a)):
    #   if i%1000==0:
    #     print i
    #     store['exposure'][ind0]=sp['exposure'][i]
    #     store['ccd'][ind0]=-1
    #     store['type'][ind0]=-1
    #     store['ra'][ind0]=np.mean(sp['ra'+name[k]][i:i+j+1])
    #     store['dec'][ind0]=np.mean(sp['dec'+name[k]][i:i+j+1])
    #   store['exposure'][len(a)*len(b)+i]=a[i]
    #   store['ccd'][len(a)*len(b)+i]=-1
    #   store['type'][len(a)*len(b)+i]=-1
    #   mask=(store['exposure']==store['exposure'][len(a)*len(b)+i])&(store['type']==0)&((store['ccd']==27)|(store['ccd']==34))
    #   store['ra'][len(a)*len(b)+i]=np.mean(store['ra'][mask])
    #   store['dec'][len(a)*len(b)+i]=np.mean(store['dec'][mask])

    store=store[:ind0]

    fio.write(config.spointsfile,store,clobber=True)

    return

  @staticmethod
  def mean_shear(cat,mask):
    """
    Calculate and plot mean shear as a function of pixel row/column.
    """

    if not hasattr(cat, 'ra'):
      cat.ra,cat.dec=field_methods.get_field_pos(cat)

    cat.lbins=50
    split.split_methods.split_gals_lin_along(cat,'ra',mask=mask,plot=True,label='field-row',fit=True)
    split.split_methods.split_gals_lin_along(cat,'ra',mask=mask,plot=True,label='field-col',fit=True)


  @staticmethod
  def corr_points(cat,mask):
    """
    Calculate and plot tangential shear and mean shear around special points in catalog.
    """

    import fitsio as fio

    mask=catalog.CatalogMethods.check_mask(cat.coadd,mask)

    fits=fio.FITS(config.spointsfile)
    tmp=fits[-1].read()
    pointings=catalog.CatalogStore('field_points',setup=False)
    pointings.coadd=np.arange(len(tmp))
    pointings.ra=tmp['ra']
    pointings.dec=tmp['dec']
    pointings.type=tmp['type']
    pointings.ccd=tmp['ccd']
    pointings.tbins=50
    pointings.sep=np.array([.1,500.])

    mask1=tmp['ccd']==-1
    theta,out,err,chi2=corr.xi_2pt.xi_2pt(pointings,catb=cat,k=None,ga=None,gb=None,corr='NG',maska=mask1,maskb=mask,wa=None,wb=None,ran=False,mock=False,erron=True,jkmask=None,label0='',plot=False)
    fig.plot_methods.plot_field_corr(cat,theta,out,err,'centre')
    theta,out,err,chi2=corr.xi_2pt.xi_2pt(pointings,catb=cat,k='e1',ga=None,gb=None,corr='NK',maska=mask1,maskb=mask,wa=None,wb=None,ran=False,mock=False,erron=True,jkmask=None,label0='',plot=False)
    theta,out2,err2,chi2=corr.xi_2pt.xi_2pt(pointings,catb=cat,k='e2',ga=None,gb=None,corr='NK',maska=mask1,maskb=mask,wa=None,wb=None,ran=False,mock=False,erron=True,jkmask=None,label0='',plot=False)
    fig.plot_methods.plot_field_corr2(cat,theta,out,err,out2,err2,'centre')

    mask1=tmp['type']==0
    theta,out,err,chi2=corr.xi_2pt.xi_2pt(pointings,catb=cat,k=None,ga=None,gb=None,corr='NG',maska=mask1,maskb=mask,wa=None,wb=None,ran=False,mock=False,erron=True,jkmask=None,label0='',plot=False)
    fig.plot_methods.plot_field_corr(cat,theta,out,err,'chip-centre')
    theta,out,err,chi2=corr.xi_2pt.xi_2pt(pointings,catb=cat,k='e1',ga=None,gb=None,corr='NK',maska=mask1,maskb=mask,wa=None,wb=None,ran=False,mock=False,erron=True,jkmask=None,label0='',plot=False)
    theta,out2,err2,chi2=corr.xi_2pt.xi_2pt(pointings,catb=cat,k='e2',ga=None,gb=None,corr='NK',maska=mask1,maskb=mask,wa=None,wb=None,ran=False,mock=False,erron=True,jkmask=None,label0='',plot=False)
    fig.plot_methods.plot_field_corr2(cat,theta,out,err,out2,err2,'chip-centre')

    mask1=tmp['type']==1
    theta,out,err,chi2=corr.xi_2pt.xi_2pt(pointings,catb=cat,k=None,ga=None,gb=None,corr='NG',maska=mask1,maskb=mask,wa=None,wb=None,ran=False,mock=False,erron=True,jkmask=None,label0='',plot=False)
    fig.plot_methods.plot_field_corr(cat,theta,out,err,'chip-cornera')
    theta,out,err,chi2=corr.xi_2pt.xi_2pt(pointings,catb=cat,k='e1',ga=None,gb=None,corr='NK',maska=mask1,maskb=mask,wa=None,wb=None,ran=False,mock=False,erron=True,jkmask=None,label0='',plot=False)
    theta,out2,err2,chi2=corr.xi_2pt.xi_2pt(pointings,catb=cat,k='e2',ga=None,gb=None,corr='NK',maska=mask1,maskb=mask,wa=None,wb=None,ran=False,mock=False,erron=True,jkmask=None,label0='',plot=False)
    fig.plot_methods.plot_field_corr2(cat,theta,out,err,out2,err2,'chip-cornera')

    mask1=tmp['type']==2
    theta,out,err,chi2=corr.xi_2pt.xi_2pt(pointings,catb=cat,k=None,ga=None,gb=None,corr='NG',maska=mask1,maskb=mask,wa=None,wb=None,ran=False,mock=False,erron=True,jkmask=None,label0='',plot=False)
    fig.plot_methods.plot_field_corr(cat,theta,out,err,'chip-cornerb')
    theta,out,err,chi2=corr.xi_2pt.xi_2pt(pointings,catb=cat,k='e1',ga=None,gb=None,corr='NK',maska=mask1,maskb=mask,wa=None,wb=None,ran=False,mock=False,erron=True,jkmask=None,label0='',plot=False)
    theta,out2,err2,chi2=corr.xi_2pt.xi_2pt(pointings,catb=cat,k='e2',ga=None,gb=None,corr='NK',maska=mask1,maskb=mask,wa=None,wb=None,ran=False,mock=False,erron=True,jkmask=None,label0='',plot=False)
    fig.plot_methods.plot_field_corr2(cat,theta,out,err,out2,err2,'chip-cornerb')

    mask1=tmp['type']==3
    theta,out,err,chi2=corr.xi_2pt.xi_2pt(pointings,catb=cat,k=None,ga=None,gb=None,corr='NG',maska=mask1,maskb=mask,wa=None,wb=None,ran=False,mock=False,erron=True,jkmask=None,label0='',plot=False)
    fig.plot_methods.plot_field_corr(cat,theta,out,err,'chip-cornerc')
    theta,out,err,chi2=corr.xi_2pt.xi_2pt(pointings,catb=cat,k='e1',ga=None,gb=None,corr='NK',maska=mask1,maskb=mask,wa=None,wb=None,ran=False,mock=False,erron=True,jkmask=None,label0='',plot=False)
    theta,out2,err2,chi2=corr.xi_2pt.xi_2pt(pointings,catb=cat,k='e2',ga=None,gb=None,corr='NK',maska=mask1,maskb=mask,wa=None,wb=None,ran=False,mock=False,erron=True,jkmask=None,label0='',plot=False)
    fig.plot_methods.plot_field_corr2(cat,theta,out,err,out2,err2,'chip-cornerc')

    mask1=tmp['type']==4
    theta,out,err,chi2=corr.xi_2pt.xi_2pt(pointings,catb=cat,k=None,ga=None,gb=None,corr='NG',maska=mask1,maskb=mask,wa=None,wb=None,ran=False,mock=False,erron=True,jkmask=None,label0='',plot=False)
    fig.plot_methods.plot_field_corr(cat,theta,out,err,'chip-cornerd')
    theta,out,err,chi2=corr.xi_2pt.xi_2pt(pointings,catb=cat,k='e1',ga=None,gb=None,corr='NK',maska=mask1,maskb=mask,wa=None,wb=None,ran=False,mock=False,erron=True,jkmask=None,label0='',plot=False)
    theta,out2,err2,chi2=corr.xi_2pt.xi_2pt(pointings,catb=cat,k='e2',ga=None,gb=None,corr='NK',maska=mask1,maskb=mask,wa=None,wb=None,ran=False,mock=False,erron=True,jkmask=None,label0='',plot=False)
    fig.plot_methods.plot_field_corr2(cat,theta,out,err,out2,err2,'chip-cornerd')

    mask1=tmp['type']>0
    theta,out,err,chi2=corr.xi_2pt.xi_2pt(pointings,catb=cat,k=None,ga=None,gb=None,corr='NG',maska=mask1,maskb=mask,wa=None,wb=None,ran=False,mock=False,erron=True,jkmask=None,label0='',plot=False)
    fig.plot_methods.plot_field_corr(cat,theta,out,err,'chip-corner')
    theta,out,err,chi2=corr.xi_2pt.xi_2pt(pointings,catb=cat,k='e1',ga=None,gb=None,corr='NK',maska=mask1,maskb=mask,wa=None,wb=None,ran=False,mock=False,erron=True,jkmask=None,label0='',plot=False)
    theta,out2,err2,chi2=corr.xi_2pt.xi_2pt(pointings,catb=cat,k='e2',ga=None,gb=None,corr='NK',maska=mask1,maskb=mask,wa=None,wb=None,ran=False,mock=False,erron=True,jkmask=None,label0='',plot=False)
    fig.plot_methods.plot_field_corr2(cat,theta,out,err,out2,err2,'chip-corner')

    return theta,out,err,chi2    


class field_methods(object):
  """
  Utilities for doing pixel and chip calculations.
  """

  chip_centres = {

  'N7':[16.908,191.670],
  'N6':[16.908,127.780],
  'N5':[16.908,63.890],
  'N4':[16.908,0.],
  'N3':[16.908,-63.890],
  'N2':[16.908,-127.780],
  'N1':[16.908,-191.670],
  'N13':[50.724,159.725],
  'N12':[50.724,95.835],
  'N11':[50.724,31.945],
  'N10':[50.724,-31.945],
  'N9':[50.724,-95.835],
  'N8':[50.724,-159.725],
  'N19':[84.540,159.725],
  'N18':[84.540,95.835],
  'N17':[84.540,31.945],
  'N16':[84.540,-31.945],
  'N15':[84.540,-95.835],
  'N14':[84.540,-159.725],
  'N24':[118.356,127.780],
  'N23':[118.356,63.890],
  'N22':[118.356,0.],
  'N21':[118.356,-63.890],
  'N20':[118.356,-127.780],
  'N28':[152.172,95.835],
  'N27':[152.172,31.945],
  'N26':[152.172,-31.945],
  'N25':[152.172,-95.835],
  'N31':[185.988,63.890],
  'N30':[185.988,0.],
  'N29':[185.988,-63.890],
  'S7':[-16.908,191.670],
  'S6':[-16.908,127.780],
  'S5':[-16.908,63.890],
  'S4':[-16.908,0.],
  'S3':[-16.908,-63.890],
  'S2':[-16.908,-127.780],
  'S1':[-16.908,-191.670],
  'S13':[-50.724,159.725],
  'S12':[-50.724,95.835],
  'S11':[-50.724,31.945],
  'S10':[-50.724,-31.945],
  'S9':[-50.724,-95.835],
  'S8':[-50.724,-159.725],
  'S19':[-84.540,159.725],
  'S18':[-84.540,95.835],
  'S17':[-84.540,31.945],
  'S16':[-84.540,-31.945],
  'S15':[-84.540,-95.835],
  'S14':[-84.540,-159.725],
  'S24':[-118.356,127.780],
  'S23':[-118.356,63.890],
  'S22':[-118.356,0.],
  'S21':[-118.356,-63.890],
  'S20':[-118.356,-127.780],
  'S28':[-152.172,95.835],
  'S27':[-152.172,31.945],
  'S26':[-152.172,-31.945],
  'S25':[-152.172,-95.835],
  'S31':[-185.988,63.890],
  'S30':[-185.988,0.],
  'S29':[-185.988,-63.890]
  }

  ccdid=['S29','S30','S31','S25','S26','S27','S28','S20','S21','S22','S23','S24','S14','S15','S16','S17','S18','S19','S8','S9','S10','S11','S12','S13','S1','S2','S3','S4','S5','S6','S7','N1','N2','N3','N4','N5','N6','N7','N8','N9','N10','N11','N12','N13','N14','N15','N16','N17','N18','N19','N20','N21','N22','N23','N24','N25','N26','N27','N28','N29','N30','N31']

  ccdx=4096.*15.e-6*1000.
  ccdy=2048.*15.e-6*1000.

  @staticmethod
  def ccd_centres():

    centrex=[]
    centrey=[]
    for i,x in enumerate(field_methods.ccdid):
      centrex=np.append(centrex,field_methods.chip_centres.get(x,None)[1])
      centrey=np.append(centrey,field_methods.chip_centres.get(x,None)[0])

    return np.vstack((centrex,centrey)).T

  @staticmethod
  def ccd_corners():

    cornersx=[]
    cornersy=[]
    for i,x in enumerate(field_methods.ccdid):
      centrex=np.append(centrex,field_methods.chip_centres.get(x,None)[0]-field_methods.ccdx/2.) # lower left
      centrex=np.append(centrex,field_methods.chip_centres.get(x,None)[0]-field_methods.ccdx/2.) # lower right
      centrex=np.append(centrex,field_methods.chip_centres.get(x,None)[0]+field_methods.ccdx/2.) # upper left
      centrex=np.append(centrex,field_methods.chip_centres.get(x,None)[0]+field_methods.ccdx/2.) # upper right

      centrey=np.append(centrey,field_methods.chip_centres.get(x,None)[1]-field_methods.ccdy/2.)
      centrey=np.append(centrey,field_methods.chip_centres.get(x,None)[1]+field_methods.ccdy/2.)
      centrey=np.append(centrey,field_methods.chip_centres.get(x,None)[1]-field_methods.ccdy/2.)
      centrey=np.append(centrey,field_methods.chip_centres.get(x,None)[1]+field_methods.ccdy/2.)

    return np.vstack((centrex,centrey)).T

  @staticmethod
  def ccd_to_field(ccd,ccdx,ccdy):

    centre=field_methods.ccd_centres()

    centrex=(centre[:,0])[[ccd]]
    centrey=(centre[:,1])[[ccd]]

    return ccdx*15e-6*1000+centrex,ccdy*15e-6*1000+centrey 

  @staticmethod
  def get_field_pos(cat):

    x,y=field_methods.ccd_to_field(cat.ccd,cat.row-field_methods.ccdx/2.,cat.col-field_methods.ccdy/2.)

    return x,y 

  @staticmethod
  def translate_to_wcs(pos,image):

    from esutil import wcsutil
    
    wcs=wcsutil.WCS(image, longpole=180.0, latpole=90.0, theta0=90.0)
    ra,dec=wcs.image2sky(pos[0],pos[1])

    return ra,dec

  @staticmethod
  def get_coadd_tile(ra,dec,tiles=None):

    if tiles is None:
      tiles=fio.FITS(config.coaddtiles)[-1].read()

    tmp=tiles['TILENAME'][(ra<tiles['URAUR'])&(dec<tiles['UDECUR'])&(ra>tiles['URALL'])&(dec>tiles['UDECLL'])]
    if len(tmp)==0:
      tmp=tiles['TILENAME'][((ra+360)<tiles['URAUR'])&(dec<tiles['UDECUR'])&((ra+360)>tiles['URALL'])&(dec>tiles['UDECLL'])]

    return tmp[0].rstrip()

  @staticmethod
  def get_radec_coadd_tiles(tiles=None,tiles0=None,file=config.coaddtiles):

    if tiles is None:
      tiles=fio.FITS(file)[-1].read()

    if tiles0 is None:
      mask=np.ones(len(tiles)).astype(bool)
    else:
      mask=np.in1d(np.core.defchararray.strip(tiles['TILENAME']),tiles0,assume_unique=False)

    return tiles,np.vstack(((tiles['URAUR'][mask]+tiles['URALL'][mask])/2.,(tiles['UDECUR'][mask]+tiles['UDECLL'][mask])/2.)).T
