
import numpy as np
import matplotlib
matplotlib.use ('agg')
import matplotlib.pyplot as plt
import os
import inspect

# if "NERSC_HOST" in os.environ:
#   os.environ['PATH']+=':/usr/common/software/latex/2015/2015/bin/x86_64-linux'
#   os.environ['LATEX_DIR']='/usr/common/software/latex/2015/2015'

dirname=os.path.split(__file__)[0]
style_file=os.path.join(dirname, "SVA1StyleSheet.mplstyle")
plt.style.use(style_file)
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import pylab
import healpy as hp

import catalog
import config
import pz

class plot_methods(object):
  """
  Plotting routines used by various modules.
  """

  @staticmethod
  def plot_hist(x1,bins=config.cfg.get('hbins',500),name='',label='',tile='',w=None):

    print 'hist ',label,tile

    if tile!='':
      bins/=10

    plt.figure()
    if (w is None)|(tile!=''):
      plt.hist(x1,bins=bins,histtype='stepfilled')
    else:
      plt.hist(x1,bins=bins,alpha=0.25,normed=True,label='unweighted',histtype='stepfilled')
      plt.hist(x1,bins=bins,alpha=0.25,normed=True,weights=w,label='weighted',histtype='stepfilled')
    plt.ylabel(r'$n$')
    s=config.lbl.get(label,label.replace('_','-'))
    if config.log_val.get(label,False):
      s='log '+s
    plt.xlabel(s+' '+tile)
    plt.minorticks_on()
    if tile!='':
      name='tile_'+tile+'_'+name
    plt.legend(loc='upper right',ncol=2, frameon=True,prop={'size':12})
    plt.savefig('plots/hist/hist_'+name+'_'+label.replace('_','-')+'.png', bbox_inches='tight')
    plt.close()

    return

  @staticmethod
  def plot_comp_hist(x1,x2,bins=config.cfg.get('hbins',500),name='',name2='',label='',w1=None,w2=None,range=None,normed=True):

    plt.figure()
    if w1 is None:
      plt.hist(x1,bins=bins,alpha=.25,label=name,normed=normed,histtype='stepfilled',range=range)
      plt.hist(x2,bins=bins,alpha=.25,label=name2,normed=normed,histtype='stepfilled',range=range)
    else:
      plt.hist(x1,bins=bins,alpha=.25,label=name,normed=normed,histtype='stepfilled',weights=w1,range=range)
      plt.hist(x2,bins=bins,alpha=.25,label=name2,normed=normed,histtype='stepfilled',weights=w2,range=range)
    plt.ylabel(r'$n$')
    s=config.lbl.get(label,label.replace('_','-'))
    if config.log_val.get(label,False):
      s='log '+s
    plt.xlabel(s)
    plt.minorticks_on()
    plt.legend(loc='upper right',ncol=2, frameon=False,prop={'size':12},framealpha=0.2)
    plt.savefig('plots/hist/hist_'+name+'_'+name2+'_'+label.replace('_','-')+'.png', bbox_inches='tight')
    plt.close()

    return

  @staticmethod
  def plot_2D_hist(x1,y1,bins=config.cfg.get('hbins',500),xname='',yname='',xlabel='',ylabel='',xtile='',ytile=''):

    print 'hist 2D',xlabel,ylabel

    plt.figure()
    plt.hist2d(x1,y1,bins=bins)
    s=config.lbl.get(xlabel,'')
    if config.log_val.get(xlabel,False):
      s='log '+s
    plt.xlabel(s+' '+xtile)
    s=config.lbl.get(ylabel,'')
    if config.log_val.get(ylabel,False):
      s='log '+s
    plt.ylabel(s+' '+ytile)
    plt.minorticks_on()
    if xtile!='':
      xname='tile_'+xtile+'_'+xname
    plt.savefig('plots/hist/hist_2D_'+xname+'_'+yname+'_'+xlabel.replace('_','-')+'_'+ylabel.replace('_','-')+'.png', bbox_inches='tight')
    plt.close()

    plt.figure()
    plt.hist2d(np.abs(x1),np.abs(y1),bins=bins,norm=LogNorm())
    s=config.lbl.get(xlabel,'')
    if config.log_val.get(xlabel,False):
      s='log '+s
    plt.xlabel(s+' '+xtile)
    s=config.lbl.get(ylabel,'')
    if config.log_val.get(ylabel,False):
      s='log '+s
    plt.ylabel(s+' '+ytile)   
    plt.minorticks_on()
    plt.savefig('plots/hist/hist_2D_'+xname+'_'+yname+'_'+xlabel.replace('_','-')+'_'+ylabel.replace('_','-')+'_abslog.png', bbox_inches='tight')
    plt.close()

    return

  @staticmethod
  def plot_hexbin(x1,ra,dec,bins=config.cfg.get('hexbins',20),name='',label='',tile=''):

    s82=(dec>-10)
    spta=(~s82)&(ra<0)
    sptc=(~s82)&(ra>50)
    sptb=(~s82)&(~spta)&(~sptc)

    plot_methods.plot_hexbin_base(x1,ra[s82],dec[s82],label=label,bins=bins,part='s82',name=name,tile=tile)
    plot_methods.plot_hexbin_base(x1,ra[spta],dec[spta],label=label,bins=bins,part='spta',name=name,tile=tile)
    plot_methods.plot_hexbin_base(x1,ra[sptb],dec[sptb],label=label,bins=bins,part='sptb',name=name,tile=tile)
    plot_methods.plot_hexbin_base(x1,ra[sptc],dec[sptc],label=label,bins=bins,part='sptc',name=name,tile=tile)

    return

  @staticmethod
  def plot_hexbin_base(x1,ra,dec,bins=config.cfg.get('hexbins',20),name='',label='',part='',tile=''):

    ra1=np.max(ra)
    ra0=np.min(ra)
    dec1=np.max(dec)
    dec0=np.min(dec)

    plt.figure()

    plt.hexbin(ra,dec,x1,gridsize=(int((ra1-ra0)*bins),int((dec1-dec0)*bins)), cmap=plt.cm.afmhot,linewidths=(0,))
    cb = plt.colorbar(orientation='horizontal')
    plt.xlabel('RA')
    plt.ylabel('Dec')
    plt.xlim((ra0-0.1*(ra1-ra0),ra1+0.1*(ra1-ra0)))
    plt.ylim((dec0-0.1*(dec1-dec0),dec1+0.1*(dec1-dec0)))
    plt.minorticks_on()
    s=config.lbl.get(label,'')
    if config.log_val.get(label,False):
      s='log '+s
    if tile!='':
      name='tile_'+tile+'_'+name
    cb.set_label(s)
    plt.gca().set_aspect('equal', 'box')
    plt.savefig('plots/footprint/hexbin_'+name+'_'+label+'_'+part+'.png', dpi=500,bbox_inches='tight')
    plt.close()

    # plt.figure()
    # plt.hexbin(cat.ra[mask],cat.dec[mask],x1,gridsize=bins,bins='log', cmap=plt.cm.afmhot,linewidth=0)
    # cb = plt.colorbar()
    # plt.xlabel('RA')
    # plt.ylabel('Dec')
    # plt.minorticks_on()
    # s=config.lbl.get(label,'')
    # if config.log_val.get(label,False):
    #   s='log '+s
    # cb.set_label(s)
    # plt.gca().set_aspect('equal', 'box')
    # plt.savefig('plots/footprint/hexbin_'+name+'_'+label+'_'+part+'_log.png', dpi=500,bbox_inches='tight')
    # plt.close()

    return


  @staticmethod
  def plot_field_footprint(x,y,name,label='',bins=config.cfg.get('footbins',10)):

    dra=np.max(x)-np.min(x)
    ddec=np.max(y)-np.min(y)

    plt.figure()
    plt.hist2d(x,y,bins=(int(dra*bins),int(ddec*bins)),range=((np.min(x)-.1*dra,np.max(x)+.1*dra),(np.min(y)-.1*ddec,np.max(y)+.1*ddec)),normed=True, cmap=plt.cm.afmhot)#cmax=1.2e-5,cmin=0.5e-5,
    cb = plt.colorbar(orientation='horizontal')
    plt.gca().set_aspect('equal', 'box')
    plt.savefig('plots/footprint/field_'+name+'_'+label+'.png', dpi=500, bbox_inches='tight')
    plt.close()

    plt.figure()
    plt.hist2d(x,y,bins=(int(dra*bins),int(ddec*bins)),range=((np.min(x)-.1*dra,np.max(x)+.1*dra),(np.min(y)-.1*ddec,np.max(y)+.1*ddec)),normed=True,norm=LogNorm(), cmap=plt.cm.afmhot)
    #cb = plt.colorbar()
    plt.gca().set_aspect('equal', 'box')
    plt.savefig('plots/footprint/field_'+name+'_'+label+'_log.png', dpi=500, bbox_inches='tight')
    plt.close()

    return

  @staticmethod
  def plot_footprint(ra,dec,name,label='',bins=config.cfg.get('footbins',100),cap=None):

    s82=(dec>-10)
    spta=(~s82)&(ra<0)
    sptc=(~s82)&(ra>50)
    sptb=(~s82)&(~spta)&(~sptc)

    plot_methods.plot_footprint_base(ra[s82],dec[s82],name,label=label,bins=bins,part='s82')
    plot_methods.plot_footprint_base(ra[spta],dec[spta],name,label=label,bins=bins,part='spta')
    plot_methods.plot_footprint_base(ra[sptb],dec[sptb],name,label=label,bins=bins,part='sptb')
    plot_methods.plot_footprint_base(ra[sptc],dec[sptc],name,label=label,bins=bins,part='sptc')

    return

  @staticmethod
  def plot_footprint_base(ra,dec,name,label='',bins=config.cfg.get('footbins',100),part='',cap=None):

    # if not hasattr(cat, 'gdmask'):
    #   cat.gdmask=hp.reorder(hp.read_map(config.golddir+'y1a1_gold_1.0.1_wide_footprint_4096.fit'),inp='ring',out='nested')
    #   cat.badmask=hp.reorder(hp.read_map(config.golddir+'y1a1_gold_1.0.1_wide_badmask_4096.fit'),inp='ring',out='nested')
    # if not hasattr(cat,'pix'):
    #   cat.pix=radec_to_hpix(ra,dec,nside=4096,nest=True)

    # mask0=(cat.gdmask>=1)&(cat.badmask==0)

    # hpmap=np.ones((12*4096**2))*hp.UNSEEN
    # hpmap[(cat.gdmask>=1)]=0
    # cnt=np.zeros(12*4096**2)
    # cnt[:np.max(cat.pix[mask])+1]=np.bincount(cat.pix[mask])
    # hpmap[mask0]=cnt[mask0]
    # hp.cartview(hpmap,latra=config.dec_lim.get(part),lonra=config.ra_lim.get(part),nest=True,xsize=10000,title=label)
    # plt.savefig('plots/footprint/footprint_'+cat.name+'_'+label+'_'+part+'.png', dpi=1000,bbox_inches='tight')
    # plt.close()
    print 'inside plot func',os.getpid()

    dra=config.ra_lim.get(part)[1]-config.ra_lim.get(part)[0]
    ddec=config.dec_lim.get(part)[1]-config.dec_lim.get(part)[0]

    print 'inside plot func - before fig',os.getpid()

    plt.figure()
    tmp=config.ra_lim.get(part),config.dec_lim.get(part)
    a=plt.hist2d(ra,dec,bins=(int(dra*bins),int(ddec*bins)),range=tmp,normed=True, cmap=plt.cm.afmhot,cmax=cap)

    print 'inside plot func - mid fig',os.getpid()
    cb = plt.colorbar(orientation='horizontal')
    plt.gca().set_aspect('equal', 'box')
    print 'inside plot func - before savefig',os.getpid()
    plt.savefig('plots/footprint/footprint_'+name+'_'+label+'_'+part+'.png', dpi=500, bbox_inches='tight')
    print 'inside plot func - after savefig',os.getpid()
    plt.close()
    print 'end plot func',os.getpid()

    # plt.figure()
    # plt.hist2d(cat.ra[mask],cat.dec[mask],bins=(int(dra*bins),int(ddec*bins)),range=((np.min(cat.ra[mask])-.1*dra,np.max(cat.ra[mask])+.1*dra),(np.min(cat.dec[mask])-.1*ddec,np.max(cat.dec[mask])+.1*ddec)),normed=True,norm=LogNorm(), cmap=plt.cm.afmhot)
    # #cb = plt.colorbar()
    # plt.gca().set_aspect('equal', 'box')
    # plt.savefig('plots/footprint/footprint_'+cat.name+'_'+label+'_log.png', dpi=500, bbox_inches='tight')
    # plt.close()    

    return

  @staticmethod
  def plot_whisker(x,y,e1,e2,name='',label='',scale=.01,key='',chip=False):

    plt.figure()
    Q = plt.quiver(x,y,e1,e2,units='width',pivot='middle',headwidth=0,width=.0005)
    if chip:
      plt.quiverkey(Q,0.2,0.125,scale,str(scale)+' '+key,labelpos='E',coordinates='figure',fontproperties={'weight': 'bold'})
      plt.xlim((-250,4250))
      plt.ylim((-200,2100))
    else:
      plt.quiverkey(Q,0.2,0.2,scale,str(scale)+' '+key,labelpos='E',coordinates='figure',fontproperties={'weight': 'bold'})
    plt.savefig('plots/footprint/whisker_'+name+'_'+label+'.png', dpi=500,bbox_inches='tight')
    plt.close()

    return

  @staticmethod
  def get_filename_str(cat):
    return cat.name+'_bs-'+str(cat.bs)+'_wt-'+str(cat.wt)

  @staticmethod
  def plot_lin_split(x,e1,e2,e1err,e2err,m1,m2,b1,b2,name,val,log=False,label='',e=True,val2=None,trend=True):

    plt.figure()
    if e:
      l1=r'$\langle e_1 \rangle$'
      l2=r'$\langle e_2 \rangle$'
      plt.errorbar(x,e1,yerr=e1err,marker='o',linestyle='',color='r',label=r'$\langle e_1 \rangle$')
    else:
      l1=r'$\langle e_1 \rangle$'      
      plt.errorbar(x,e1,yerr=e1err,marker='o',linestyle='',color='r',label='')
    if trend:
      plt.errorbar(x,m1*x+b1,marker='',linestyle='-',color='r')
    if e:
      plt.errorbar(x+(x[1]-x[0])/5.,e2,yerr=e2err,marker='o',linestyle='',color='b',label=r'$\langle e_2 \rangle$')
      if trend:
        plt.errorbar(x,m2*x+b2,marker='',linestyle='-',color='b')
      plt.ylabel(r'$\langle e \rangle$')
    else:
      plt.ylabel(r'$\langle $'+config.lbl.get(val2,val2)+r'$ \rangle$')
    if e:
      plt.legend(loc='lower right',ncol=1, frameon=True,prop={'size':12})
      plt.axhline(.004,color='k')
      plt.axhline(-.004,color='k')
    if config.log_val.get(val,False):
      plt.xlabel('log '+config.lbl.get(val,val.replace('_','-')))
    else:
      plt.xlabel(config.lbl.get(val,val.replace('_','-')))
    y1=np.min(np.minimum(e1,e2))
    if e:   
      y2=np.max(np.maximum(e1,e2))
    else:
      y2=y1
    # plt.ylim((np.min([y1-(y2-y1)/10.,-.005]),np.max([y2+(y2-y1)/10.,.005])))
    plt.minorticks_on()
    if val2 is not None:
      val+='-'+val2
    plt.savefig('plots/split/lin_split_'+name+'_'+val+'_'+label.replace('_','-')+'.png', bbox_inches='tight')
    plt.close()

    return

  @staticmethod
  def plot_2pt_split_sub(cat,val,split,n,yl,xi,i,log,blind=True):

    if blind:
      bf=1.+(np.random.rand(1)[0]-.5)*.4
    else:
      bf=1.

    plt.figure(0)
    ax=plt.subplot(3,3,n)
    #ax.fill_between([1,100],-1,1,facecolor='gray',alpha=0.25)
    #ax.fill_between([1,100],-2,2,facecolor='gray',alpha=0.2)
    plt.errorbar(xi[0],np.zeros((len(xi[0]))),marker='',linestyle='-',color='k',alpha=.8)
    # plt.errorbar(xi[0],xi[10][i]*np.ones((len(xi[0]))),marker='',linestyle='-',color='g')
    plt.errorbar(xi[0],(xi[1][i]-xi[2][i])/xi[2][i],marker='v',linestyle='',color='g')#yerr=xi[7][i]/xi[2][i]
    # plt.errorbar(xi[0],xi[11][i]*np.ones((len(xi[0]))),marker='',linestyle='-',color='r')
    # plt.errorbar(xi[0]*1.1,(xi[3][i]-xi[1][i])/xi[2][i],marker='o',linestyle='',color='r')#,yerr=xi[8][i]/xi[2][i]
    # plt.errorbar(xi[0],xi[12][i]*np.ones((len(xi[0]))),marker='',linestyle='-',color='b')
    plt.errorbar(xi[0]*1.2,(xi[3][i]-xi[2][i])/xi[2][i],marker='^',linestyle='',color='b')#,yerr=xi[9][i]/xi[2][i]
    plt.xlabel(r'$\theta$ (arcmin)')
    plt.xscale('log')
    plt.ylim(-2.5,2.5)
    if i==1:
      plt.xlim(10,100)
    else:
      plt.xlim(1,100)
    ax.set_xticklabels([])
    plt.ylabel(r'$\Delta '+yl+r'/'+yl+r'$')
    # ax.axes.get_yaxis().set_ticks([])

    ax=plt.subplot(3,3,3+n)
    s=config.lbl.get(val,val)
    if log:
      s='log '+s
    plt.errorbar(xi[0]*1.1,xi[0]*xi[1][i]*bf,yerr=xi[0]*xi[4][i],marker='v',linestyle='',color='g',label=s.replace('_','-')+r'$<$'+str(np.around(split,2)))
    plt.errorbar(xi[0],xi[0]*xi[2][i]*bf,yerr=xi[0]*xi[5][i],marker='o',linestyle='',color='r',label='All')# (upper-lower in top)
    plt.errorbar(xi[0]*1.2,xi[0]*xi[3][i]*bf,yerr=xi[0]*xi[6][i],marker='^',linestyle='',color='b',label=s.replace('_','-')+r'$>$'+str(np.around(split,2)))
    plt.xlabel(r'$\theta$ (arcmin)')
    plt.xscale('log')
    if n==1:
      leg=plt.legend(loc='upper left',ncol=1, frameon=False,prop={'size':12},framealpha=0.2)
    ax.set_xticklabels([])
    plt.ylabel(r'$\theta\times'+yl+r'$')
    if i==1:
      plt.xlim(10,100)
    else:
      plt.xlim(1,100)
    if n<3:
      plt.ylim(0,4e-4)
    else:
      plt.ylim(0,4e-3)
    ax.axes.get_yaxis().set_ticks([])

    ax=plt.subplot(3,3,6+n)
    plt.errorbar(xi[0]*1.1,xi[1][i]*bf,yerr=xi[4][i],marker='v',linestyle='',color='g')
    plt.errorbar(xi[0],xi[2][i]*bf,yerr=xi[5][i],marker='o',linestyle='',color='r')
    plt.errorbar(xi[0]*1.2,xi[3][i]*bf,yerr=xi[6][i],marker='^',linestyle='',color='b')
    plt.xlabel(r'$\theta$ (arcmin)')
    plt.ylabel(r'$'+yl+r'$')
    plt.xscale('log')
    plt.yscale('log')
    if i==1:
      plt.xlim(10,100)
    else:
      plt.xlim(1,100)
    if n<3:
      plt.ylim(5e-7,1e-4)
    else:
      plt.ylim(2e-5,2e-3)
    ax.axes.get_yaxis().set_ticks([])
    ax.tick_params('y', length=0, width=0, which='both')
    # ax.YAxis().set_ticks_position('none')

    return

  @staticmethod
  def plot_2pt_split(xi,gt,cat,val,split,log):

    plt.figure(0,figsize=(15,10))

    for i in range(3):

      if i==0:
        plot_methods.plot_2pt_split_sub(cat,val,split,i+1,r'\xi_{+}',xi,0,log)
      elif i==1:
        plot_methods.plot_2pt_split_sub(cat,val,split,i+1,r'\xi_{-}',xi,1,log)
      elif i==2:
        plot_methods.plot_2pt_split_sub(cat,val,split,i+1,r'\gamma_{t}',gt,0,log) 

    #plt.minorticks_on()
    plt.subplots_adjust(hspace=0,wspace=.4)
    plt.savefig('plots/split/2pt_split_'+plot_methods.get_filename_str(cat)+'_'+val.replace('_','-')+'.png', bbox_inches='tight')
    plt.close(0)

    return

  @staticmethod
  def plot_field_corr(cat,theta,out,err,label):

    plt.figure()
    plt.errorbar(theta,theta*out[0],yerr=theta*err[0],marker='o',linestyle='',color='r',label=r'$\gamma_{t}$')
    plt.errorbar(theta,theta*out[2],yerr=theta*err[2],marker='o',linestyle='',color='b',label=r'$\gamma_{x}$')
    if 'chip' not in label:
      plt.axvline(x=5.25*60, linewidth=1, color='k')
    elif 'corner' in label:
      plt.axvline(x=0.75*60, linewidth=1, color='k')
      plt.axvline(x=0.15*60, linewidth=1, color='k')
      plt.axvline(x=0.765*60, linewidth=1, color='k')
    elif 'centre' in label:
      plt.axvline(x=0.75*60/2., linewidth=1, color='k')
      plt.axvline(x=0.15*60/2., linewidth=1, color='k')
      plt.axvline(x=0.765*60/2., linewidth=1, color='k')
    plt.ylabel(r'$\theta \times\gamma$')
    plt.xlabel(r'$\theta$ (arcmin)')
    plt.ylim((-.005,.005))
    plt.xscale('log')
    plt.minorticks_on()
    plt.legend(loc='upper right',ncol=1, frameon=True,prop={'size':12})
    plt.savefig('plots/xi/field_'+label+'_'+cat.name+'.png', bbox_inches='tight')
    plt.close()

    plt.figure()
    plt.errorbar(theta[out[0]>0],out[0][out[0]>0],yerr=err[0][out[0]>0],marker='o',linestyle='',color='r',label=r'$\gamma_{t}$')
    plt.errorbar(theta[out[2]>0],out[2][out[2]>0],yerr=err[2][out[2]>0],marker='o',linestyle='',color='b',label=r'$\gamma_{x}$')
    if np.sum(out[0]<0):
      plt.errorbar(theta[out[0]<0],-out[0][out[0]<0],yerr=err[0][out[0]<0],marker='x',linestyle='',color='r',label='')
    if np.sum(out[2]<0):
      plt.errorbar(theta[out[2]<0],-out[2][out[2]<0],yerr=err[2][out[2]<0],marker='x',linestyle='',color='b',label='')
    if 'chip' not in label:
      plt.axvline(x=5.25*60, linewidth=1, color='k')
    elif 'corner' in label:
      plt.axvline(x=0.75*60, linewidth=1, color='k')
      plt.axvline(x=0.15*60, linewidth=1, color='k')
      plt.axvline(x=0.765*60, linewidth=1, color='k')
    elif 'center' in label:
      plt.axvline(x=0.75*60/2., linewidth=1, color='k')
      plt.axvline(x=0.15*60/2., linewidth=1, color='k')
      plt.axvline(x=0.765*60/2., linewidth=1, color='k')
    plt.ylabel(r'$\gamma$')
    plt.xlabel(r'$\theta$ (arcmin)')
    plt.yscale('log')
    plt.xscale('log')
    plt.minorticks_on()
    plt.legend(loc='upper right',ncol=1, frameon=True,prop={'size':12})
    plt.savefig('plots/xi/field_'+label+'_'+cat.name+'_log.png', bbox_inches='tight')
    plt.close()

    return


  @staticmethod
  def plot_field_corr2(cat,theta,out,err,out2,err2,label):

    plt.figure()
    plt.errorbar(theta,theta*out[0],yerr=theta*err[0],marker='o',linestyle='',color='r',label=r'$e_1$')
    plt.errorbar(theta,theta*out2[0],yerr=theta*err2[0],marker='o',linestyle='',color='b',label=r'$e_2$')
    if 'chip' not in label:
      plt.axvline(x=5.25*60, linewidth=1, color='k')
    elif 'corner' in label:
      plt.axvline(x=0.75*60, linewidth=1, color='k')
      plt.axvline(x=0.15*60, linewidth=1, color='k')
      plt.axvline(x=0.765*60, linewidth=1, color='k')
    elif 'centre' in label:
      plt.axvline(x=0.75*60/2., linewidth=1, color='k')
      plt.axvline(x=0.15*60/2., linewidth=1, color='k')
      plt.axvline(x=0.765*60/2., linewidth=1, color='k')
    plt.ylabel(r'$\langle e \rangle$')
    plt.xlabel(r'$\theta$ (arcmin)')
    plt.ylim((-.005,.005))
    plt.xscale('log')
    plt.minorticks_on()
    plt.legend(loc='upper right',ncol=1, frameon=True,prop={'size':12})
    plt.savefig('plots/xi/field_'+label+'_'+cat.name+'_mean_e.png', bbox_inches='tight')
    plt.close()

    return

  @staticmethod
  def fig_create_xi(cat,catb,corr,theta,out,err,k,ga,gb):

    plt.figure()
    if (corr=='GG')|(corr=='NG'):
      plt.errorbar(theta,theta*out[0],yerr=theta*err[0],marker='o',linestyle='',color='r',label=r'$\xi_{+}$')
      if corr=='GG':
        plt.errorbar(theta,theta*out[1],yerr=theta*err[1],marker='o',linestyle='',color='b',label=r'$\xi_{-}$')
    plt.ylabel(r'$\theta \xi$')
    plt.xlabel(r'$\theta$ (arcmin)')
    plt.xscale('log')

    plt.minorticks_on()
    plt.legend(loc='upper right',ncol=1, frameon=True,prop={'size':12})
    if catb is None:
      plt.savefig('plots/xi/xi_'+corr+'_'+ga+'_'+gb+'_'+cat.name+'_bs-'+str(cat.bs)+'.png', bbox_inches='tight')
    else:
      plt.savefig('plots/xi/xi_'+corr+'_'+ga+'_'+gb+'_'+cat.name+'_'+catb.name+'_bs-'+str(cat.bs)+'.png', bbox_inches='tight')
    plt.close()

    plt.figure()
    if (corr=='GG')|(corr=='NG'):
      plt.errorbar(theta[out[0]>0],out[0][out[0]>0],yerr=err[0][out[0]>0],marker='o',linestyle='',color='r',label=r'$\xi_{+}$')
      if corr=='GG':
        plt.errorbar(theta[out[1]>0],out[1][out[1]>0],yerr=err[1][out[1]>0],marker='o',linestyle='',color='b',label=r'$\xi_{-}$')
      if np.sum(out[0]<0):
        plt.errorbar(theta[out[0]<0],-out[0][out[0]<0],yerr=err[0][out[0]<0],marker='x',linestyle='',color='r',label='')
      if corr=='GG':
        if np.sum(out[1]<0):
          plt.errorbar(theta[out[1]<0],-out[1][out[1]<0],yerr=err[1][out[1]<0],marker='x',linestyle='',color='b',label='')
    plt.ylabel(r'$\xi$')
    plt.xlabel(r'$\theta$ (arcmin)')
    plt.yscale('log')
    plt.xscale('log')

    plt.minorticks_on()
    plt.legend(loc='upper right',ncol=1, frameon=True,prop={'size':12})
    if catb is None:
      plt.savefig('plots/xi/xi_'+corr+'_'+ga+'_'+gb+'_'+cat.name+'_bs-'+str(cat.bs)+'_log.png', bbox_inches='tight')
    else:
      plt.savefig('plots/xi/xi_'+corr+'_'+ga+'_'+gb+'_'+cat.name+'_'+catb.name+'_bs-'+str(cat.bs)+'_log.png', bbox_inches='tight')
    plt.close()

    return

  @staticmethod
  def fig_create_xi_alpha(cat,theta,gpout,ppout,gperr,pperr,alphap,alpham,alpha0):

    ax=plt.subplot(2,1,1)
    plt.errorbar(theta,gpout[0],yerr=gperr[0],marker='.',linestyle='',color='r',label=r'$\xi^{gp}$')
    plt.errorbar(theta,ppout[0],yerr=pperr[0],marker='.',linestyle='',color='b',label=r'$\xi^{pp}$')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim((.1,500))
    # plt.ylim(5e-7,6e-4)
    ax.set_xticklabels([])
    ax.minorticks_on()
    plt.legend(loc='upper right',ncol=2, frameon=True,prop={'size':12})
    plt.ylabel(r'$|\xi_{+}|$')

    ax=plt.subplot(2,1,2)
    plt.errorbar(theta,alphap.real,marker='',linestyle='-',color='b',label=r'$\alpha_{p}$')
    plt.errorbar(theta,alphap.imag,marker='',linestyle=':',color='b',label=r'$\alpha_{p}$')
    plt.errorbar(theta,alpham.real,marker='',linestyle='-',color='r',label=r'$\alpha_{m}$')
    plt.errorbar(theta,alpham.imag,marker='',linestyle=':',color='r',label=r'$\alpha_{m}$')
    plt.errorbar(theta,alpha0,marker='',linestyle='-',color='k',label=r'$\alpha_{0}$')
    plt.xscale('log')
    plt.yscale('linear')
    plt.xlim((.1,500))
    plt.ylim(-.5,.5)
    plt.ylabel(r'$\alpha$')
    plt.xlabel(r'$\theta$ (arcmin)')    
    plt.legend(loc='upper left',ncol=2, frameon=True,prop={'size':12})
    plt.subplots_adjust(hspace=0,wspace=0)
    plt.savefig('plots/xi/xi_alpha_'+cat.name+'_bs-'+str(cat.bs)+'.png', bbox_inches='tight')
    plt.close()

    return

  @staticmethod
  def plot_pz_corr(test,pz0,label='',boot=False,ylim=0.3):

    from astropy.table import Table
    from matplotlib.ticker import MultipleLocator, FormatStrFormatter
    import fitsio as fio
    import math

    def get_cov(file,ihdu):
      cov=fio.FITS(file)['COVMAT']
      a=cov.read_header()['STRT_'+str(ihdu)]
      try:
        b=cov.read_header()['STRT_'+str(ihdu+1)]
      except ValueError:
        b=len(cov.read())
      cov=cov.read()[a:b,a:b]
      
      return cov

    colors=['k','r','g','b','c','y']
    col=['r','g','b','c','y','b']

    for j,hdu in enumerate(['shear_cl','shear_galaxy_cl','galaxy_cl']):

      fig, ax = plt.subplots()
      for i,pz1 in enumerate(pz0):

        spec=fio.FITS(config.pztestdir+test+'/out/spec_'+pz1.name+'.fits.gz')[hdu].read()
        pza=fio.FITS(config.pztestdir+test+'/out/'+pz1.name+'.fits.gz')[hdu].read()

        spec_notomo=fio.FITS(config.pztestdir+test+'/out/spec_notomo_'+pz1.name+'.fits.gz')[hdu].read()
        pz_notomo=fio.FITS(config.pztestdir+test+'/out/notomo_'+pz1.name+'.fits.gz')[hdu].read()

        cov=get_cov(config.pztestdir+test+'/out/spec_'+pz1.name+'.fits.gz',j)
        cov=np.sqrt(cov)/np.sqrt(len(np.unique(spec['ANG'])))

        notomocov=get_cov(config.pztestdir+test+'/out/spec_notomo_'+pz1.name+'.fits.gz',j)
        notomocov=np.sqrt(notomocov)/np.sqrt(len(np.unique(spec_notomo['ANG'])))

        bins=np.vstack((spec['BIN1'],spec['BIN2'])).T[np.arange(0,len(spec),len(np.unique(spec['ANG'])))]
        ratio=np.zeros((len(bins)+1))
        ratio2=np.zeros((len(bins)+1))
        ratio3=np.zeros((len(bins)+1))
        sig0=np.zeros((len(bins)+1))

        print hdu,pz1.name,np.shape(notomocov),np.shape(spec_notomo['VALUE'])

        ratio[0]=np.mean((pz_notomo['VALUE']-spec_notomo['VALUE'])/spec_notomo['VALUE'])
        ratio2[0]=np.max((pz_notomo['VALUE']-spec_notomo['VALUE'])/spec_notomo['VALUE'])
        ratio3[0]=np.min((pz_notomo['VALUE']-spec_notomo['VALUE'])/spec_notomo['VALUE'])
        sig0[0]=np.mean(np.diagonal(notomocov)/spec_notomo['VALUE'])

        for k in range(len(bins)):
          mask=(spec['BIN1']==bins[k,0])&(spec['BIN2']==bins[k,1])
          ratio[k+1]=np.mean((pza['VALUE'][mask]-spec['VALUE'][mask])/spec['VALUE'][mask])
          ratio2[k+1]=np.max((pza['VALUE'][mask]-spec['VALUE'][mask])/spec['VALUE'][mask])
          ratio3[k+1]=np.min((pza['VALUE'][mask]-spec['VALUE'][mask])/spec['VALUE'][mask])
          sig0[k+1]=np.mean(cov[mask,mask]/spec['VALUE'][mask])

        if boot:
          sig=pz.pz_spec_validation.calc_bootstrap(pz1,test,hdu)
        else:
          sig=0.

        print hdu,pz1.name,ratio,ratio2,ratio3

        plt.errorbar(np.arange(len(bins)+1),ratio,yerr=sig,marker='o',linestyle='',color=col[i],label=pz1.name)

      binlabels=[' ','2D']
      for i in range(len(bins)):
        binlabels=np.append(binlabels,[str(bins[i,0])+str(bins[i,1])])
      binlabels=np.append(binlabels,[' '])

      plt.fill_between(0-.4+.8*np.arange(100)/100.,-sig0[0]*np.ones(100),sig0[0]*np.ones(100),interpolate=True,color='k',alpha=0.2)
      for i in range(len(bins)):
        plt.fill_between(i+1-.4+.8*np.arange(100)/100.,-sig0[i+1]*np.ones(100),sig0[i+1]*np.ones(100),interpolate=True,color='k',alpha=0.2)
      plt.plot(np.arange(len(bins)+3)-1,np.zeros((len(bins)+3)), marker='', linestyle='-',color='k',label='')
      ax.xaxis.set_major_locator(MultipleLocator(1.))
      ax.yaxis.set_major_locator(MultipleLocator(ylim/3.))
      plt.xticks(np.arange(len(bins)+3)-1,binlabels)
      plt.ylim((-ylim,ylim))
      plt.xlim((-1,len(bins)+1))
      plt.ylabel(r'$\Delta C_{\ell}/C_{\ell}(\textrm{spec})$')
      plt.xlabel(r'Bin pairs')

      props = dict(boxstyle='square', lw=1.2,facecolor='white', alpha=1.)

      ax.text(0.82, 0.95, label, transform=ax.transAxes, fontsize=14,
          verticalalignment='top', bbox=props)

      plt.legend(loc='upper left',ncol=2, frameon=True,prop={'size':12})
      plt.savefig('plots/photoz/pz_'+hdu+'_'+test+'.png',bbox_inches='tight')
      plt.close()

    return

  @staticmethod
  def plot_pz_param(test,pz0,bins=3,testtype='wl',boot=False):

    col=['r','g','b','c','y','b']

    if testtype=='lss':
      params=['ggl_bias_vals--b0','ggl_bias_vals--b1','ggl_bias_vals--b2','ggl_bias_vals--b3','ggl_bias_vals--b4','ggl_bias_vals--b5','ggl_bias_vals--b6']
      name=['b0','b1','b2','b3','b4','b5','b6']
      scaling=[0.7,1.,1.2]
    elif testtype=='wl':
      params=['cosmological_parameters--sigma8_input']
      name=[r'$\sigma_{8}$']
      scaling=[0.7,.8,1.]
    elif testtype=='tcp':
      params=['cosmological_parameters--sigma8_input']
      name=[r'$\sigma_{8}$']
      scaling=[0.5,.8,1.1]

    for j,param in enumerate(params[:bins]):

      plt.figure(figsize=(12,12))
      for i,pz1 in enumerate(pz0):
        notomo=np.genfromtxt(config.pztestdir+test+'/out/'+testtype+'_spec_notomo_'+pz1.name+'_notomo_'+pz1.name+'_means.txt',names=True,dtype=None)
        tomo=np.genfromtxt(config.pztestdir+test+'/out/'+testtype+'_spec_'+pz1.name+'_'+pz1.name+'_means.txt',names=True,dtype=None)

        if boot:
          sig=pz.pz_spec_validation.calc_bootstrap_param(pz1,test,param,testtype)
          if j==0:
            sig0=pz.pz_spec_validation.calc_bootstrap_param(pz1,test,param,testtype,notomo=True)
        else:
          sig=0.
          sig0=0.

        plt.axvspan(scaling[1], scaling[1],alpha=0.2, color='gray')
        plt.errorbar(tomo['mean'][tomo['parameter']==param],i*2.,xerr=tomo['std_dev'][tomo['parameter']==param], fmt=col[i]+'.')
        plt.errorbar(scaling[1],i*2.,xerr=sig, fmt='k.')
        plt.text(scaling[2], i*2., name[j]+' -- Tomographic '+pz1.name, color=col[i], family='serif', weight='bold')
        plt.errorbar(tomo['mean'][tomo['parameter']==param],i*2.+1,xerr=tomo['std_dev'][tomo['parameter']==param], fmt=col[i]+'.')
        plt.text(scaling[2], i*2.+1, name[j]+'  -- 2D '+pz1.name, color=col[i], family='serif', weight='bold')
        plt.errorbar(scaling[1],i*2.+1,xerr=sig0, fmt='k.')
      plt.ylim((-1,len(pz0)*2+1))
      plt.xlim((scaling[0], scaling[2]+.2))
      plt.xlabel(name[j])
      plt.tick_params(axis='y',which='both',left='off',right='off',labelleft='off')
      plt.savefig('plots/photoz/pz_param_'+param+'_'+test+'.png',bbox_inches='tight')
      plt.close()

    return

  @staticmethod
  def plot_nofz(pz0,test,spec=True,name='source'):

    col=['k','r','g','b','r','g','b']
    ax=plt.subplot(2,1,1)
    for i in xrange(len(getattr(pz0,'pz'+name))-1):
      plt.plot(getattr(pz0,'bin'+name),getattr(pz0,'pz'+name)[i+1,:],color=col[i+1],linestyle='-',linewidth=1.,drawstyle='steps-mid',label='')
      plt.axvline(x=np.average(getattr(pz0,'bin'+name),weights=getattr(pz0,'pz'+name)[i+1,:]), ymin=0., ymax = 1, linewidth=2, color=col[i+1])
      print i+1,np.average(getattr(pz0,'bin'+name),weights=getattr(pz0,'pz'+name)[i+1,:])-np.average(getattr(pz0,'bin'+name),weights=getattr(pz0,'spec'+name)[i+1,:])
      if spec:
        plt.plot(getattr(pz0,'bin'+name),getattr(pz0,'spec'+name)[i+1,:],color=col[i+1],linestyle=':',linewidth=3.,drawstyle='steps-mid',label='')
        plt.axvline(x=np.average(getattr(pz0,'bin'+name),weights=getattr(pz0,'spec'+name)[i+1,:]), ymin=0., ymax = 1, linewidth=1, color=col[i+1])
    ax.set_xticklabels([])
    plt.ylabel(r'$n(z)$')
    ax=plt.subplot(2,1,2)
    plt.plot(getattr(pz0,'bin'+name),getattr(pz0,'pz'+name)[0,:],color=col[0],linestyle='-',linewidth=1.,drawstyle='steps-mid',label='')
    plt.axvline(x=np.average(getattr(pz0,'bin'+name),weights=getattr(pz0,'pz'+name)[0,:]), ymin=0., ymax = 1, linewidth=2, color=col[0])
    print 0,np.average(getattr(pz0,'bin'+name),weights=getattr(pz0,'pz'+name)[0,:])-np.average(getattr(pz0,'bin'+name),weights=getattr(pz0,'spec'+name)[0,:])
    if spec:
      plt.plot(getattr(pz0,'binspec'),getattr(pz0,'spec'+name)[0,:],color=col[0],linestyle=':',linewidth=3.,drawstyle='steps-mid',label='')
      plt.axvline(x=np.average(getattr(pz0,'binspec'),weights=getattr(pz0,'spec'+name)[0,:]), ymin=0., ymax = 1, linewidth=1, color=col[0])
    plt.xlabel(r'$z$')
    plt.ylabel(r'$n(z)$')
    # plt.xscale('log')
    # plt.xlim((0,5.))
    plt.subplots_adjust(hspace=0,wspace=0)
    plt.savefig('plots/photoz/pz_nofz_'+test+'.png',bbox_inches='tight')
    plt.close()

    return

  @staticmethod
  def plot_nofz_comp_pz(pzlist,pztypes=[''],label='',spec=True,notomo=False):

    col=['k','r','g','b','c','r','g']

    plt.figure(figsize=(14,16))
    for i in xrange(len(getattr(pzlist[0],'pz'+pztypes[0]))):
      for j,pztype in enumerate(pztypes):
        if notomo&(i>0):
          continue
        ax=plt.subplot(len(getattr(pzlist[0],'pz'+pztype)),len(pztypes),i*len(pztypes)+j+1)
        for pz0i,pz0 in enumerate(pzlist):
          pzz=getattr(pz0,'pz'+pztype)
          if spec:
            if pz0i==0:
              specz=getattr(pz0,'spec'+pztype)
            plt.plot(pz0.bin,pzz[i,:],color=col[pz0i+1],linestyle='-',linewidth=1.,drawstyle='steps-mid',label=pz0.name)
            plt.axvline(x=np.average(pz0.bin,weights=pzz[i,:]), ymin=0., ymax = 1, linewidth=2, color=col[pz0i+1])
            if pz0i==0:
              plt.plot(pzlist[0].bin,specz[i,:],color=col[0],linestyle=':',linewidth=2.,drawstyle='steps-mid',label='')
              plt.axvline(x=np.average(pz0.bin,weights=specz[i,:]), ymin=0., ymax = 1, linewidth=2, color=col[0])
          else:
            plt.plot(pz0.bin,pzz[i,:],color=col[pz0i],linestyle='-',linewidth=1.,drawstyle='steps-mid',label=pz0.name)
            plt.axvline(x=np.average(pz0.bin,weights=pzz[i,:]), ymin=0., ymax = 1, linewidth=2, color=col[pz0i])          
          if spec:
            print i,np.average(pz0.bin,weights=pzz[i,:])-np.average(pzlist[0].bin,weights=specz[0,:])
        props = dict(boxstyle='square', lw=1.2,facecolor='white', alpha=1.)
        if i==0:
          ax.text(0.73, 0.95, 'Non-tomographic', transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=props)
        else:
          ax.text(0.9, 0.95, 'Bin '+str(i), transform=ax.transAxes, fontsize=14, verticalalignment='top', bbox=props)
        if j==0:
          plt.ylabel(r'$n(z)$')
        ax.minorticks_on()
        if i<len(getattr(pzlist[0],'pz'+pztype))-1:
          ax.set_xticklabels([])
        if j>0:
          ax.set_yticklabels([])
        # plt.xscale('log')
        plt.xlim((0,1.5))
        if i==0:
          plt.title(pztype)
      plt.xlabel(r'$z$')
    plt.legend(loc='upper left',ncol=2, frameon=True,prop={'size':12})
    plt.subplots_adjust(hspace=0,wspace=0)
    if label!='':
      label+='_'
    plt.savefig('plots/photoz/pz_nofz_'+label+str(len(getattr(pzlist[0],'pz'+pztype))-1)+'_weight-'+str(pz0.wt)+'.png',bbox_inches='tight')
    plt.close()

    return


  @staticmethod
  def sig_crit_spec(pz0,bins,bootstrap,point,lensbins,edge):

    gs = gridspec.GridSpec(bins+1,1)
    plt.figure(figsize=(8,20))

    col=['k','r','g','b','c','y']

    for ibin in xrange(bins+1):
      print ibin
      ax1=plt.subplot(gs[ibin,0])        
      ax1.minorticks_on()
      for ipz,pz in enumerate(pz0):

        z_l=np.load('text/spec_'+str(ibin)+'_'+pz.name+'_invsigcrit_z.npy')
        if pz.wt:
          data=np.load('text/spec_'+str(ibin)+'_'+pz.name+'_invsigcrit_dat_weighted.npy')
        else:
          data=np.load('text/spec_'+str(ibin)+'_'+pz.name+'_invsigcrit_dat.npy')
        if bootstrap:
          if pz.wt:
            var=np.load('text/spec_'+str(ibin)+'_'+pz.name+'_invsigcrit_var_weighted.npy')
          else:
            var=np.load('text/spec_'+str(ibin)+'_'+pz.name+'_invsigcrit_var.npy')
        else:
          var=np.zeros(len(data))

        ls='o'
        if (ibin==0):
          ax1.errorbar(z_l[z_l<edge[ibin-1]]+1.1*ipz*(z_l[1]-z_l[0])/5.,data[z_l<edge[ibin-1]],yerr=var[z_l<edge[ibin-1]],linestyle='', marker=ls,color=col[ipz+1],label=pz.name)
        else:
          ax1.errorbar(z_l[z_l<edge[ibin-1]]+1.1*ipz*(z_l[1]-z_l[0])/5.,data[z_l<edge[ibin-1]],yerr=var[z_l<edge[ibin-1]],linestyle='', marker=ls,color=col[ipz+1],label='')
        ls='-'
        ax1.plot([-.05,1.3],[0,0],marker='',color='k',label='')

        if ibin==0:
          bintext=r'$'+str(np.around(np.min(pz.spec_full),2))+'<z_{\mathrm{source}}<'+str(np.around(np.max(pz.spec_full),2))+'$'
        else:
          bintext=r'$'+str(np.around(edge[ibin-1],2))+'<z_{\mathrm{source}}<'+str(np.around(edge[ibin],2))+'$'
        props = dict(boxstyle='square', lw=1.2,facecolor='white', alpha=1.)
        ax1.text(0.7, 0.95, bintext, transform=ax1.transAxes, fontsize=14,verticalalignment='top', bbox=props)
        
        if (ibin==bins):
          plt.xlabel(r'$z_{\mathrm{lens}}$')
        else:
          ax1.set_xticklabels([])

        plt.ylabel(r'$\Delta\langle \Sigma_{\mathrm{crit}}^{-1}\rangle/\langle \Sigma_{\mathrm{crit}}^{-1}\rangle_{\mathrm{spec}}$')
        plt.ylim((-1.,1.))
        ax1.xaxis.set_major_locator(MultipleLocator(.2))
        ax1.xaxis.set_minor_locator(MultipleLocator(.05))              
        plt.xlim((0.,1.1))
        plt.legend(loc='upper left',ncol=2, fancybox=True, shadow=True)

    gs.update(wspace=0.,hspace=0.)
    plt.savefig('plots/photoz/spec_point-'+str(point)+'_bins-'+str(bins)+'_bootstrap-'+str(bootstrap)+'_invsigcrit_weight-'+str(pz.wt)+'.png', bbox_inches='tight')
    plt.close()

    return

  @staticmethod
  def sig_crit_spec2(pz0,point):

    gs = gridspec.GridSpec(len(pz0)+1,1)
    plt.figure(figsize=(8,16))

    col=['r','g','b','c','y']

    for ipz,pz1 in enumerate(pz0):
      ax1=plt.subplot(gs[ipz,0])
      ax1.minorticks_on()
      z_l=np.load('text/spec_'+pz1.name+'_invsigcrit_zl.npy')
      if pz1.wt:
        data=np.load('text/spec_'+pz1.name+'_invsigcrit_dat_weighted.npy')
      else:
        data=np.load('text/spec_'+pz1.name+'_invsigcrit_dat.npy')

      plt.imshow(data, interpolation='bilinear', origin='lower', extent=[np.min(z_l), np.max(z_l), np.min(z_l), np.max(z_l)])
      plt.xlabel(r'$z_{\mathrm{lens}}$')
      plt.ylabel(r'$z_{\mathrm{source}}$')
      if ipz==0:
        plt.title(r'$\Delta\langle \Sigma_{\mathrm{crit}}^{-1}\rangle/\langle \Sigma_{\mathrm{crit}}^{-1}\rangle_{\mathrm{spec}}$')

      cb = plt.colorbar(orientation='vertical')
      cb.set_label(pz1.name)
    gs.update(wspace=0.,hspace=0.)
    plt.savefig('plots/photoz/spec_point-'+str(point)+'_invsigcrit_weight-'+str(pz1.wt)+'.png', bbox_inches='tight')
    plt.close()

    return

  @staticmethod
  def spec_loop_hist2d(pz0):


    for i,x in enumerate(['mc','mean','peak']):
      gs = gridspec.GridSpec(len(pz0)+1,1)
      plt.figure(figsize=(8,16))
      for ipz,pz in enumerate(pz0):
        ax1=plt.subplot(gs[ipz,0])
        ax1.minorticks_on()
        plt.hist2d(pz.spec_full,getattr(pz,'z_'+x+'_full'),weights=pz.w,bins=200,range=((0,1.5),(0,1.5)))
        plt.ylabel(r'$z_\mathrm{'+x+'}$')
        if (ipz==len(pz0)-1):
          plt.xlabel(r'$z_{\mathrm{spec}}$')
        else:
          ax1.set_xticklabels([])
        cb = plt.colorbar(orientation='vertical')
        cb.set_label(pz.name)
      gs.update(wspace=0.,hspace=0.)
      plt.savefig('plots/photoz/spec_hist_'+x+'_weighted.png', bbox_inches='tight')
      plt.close()

      gs = gridspec.GridSpec(len(pz0)+1,1)
      plt.figure(figsize=(8,16))
      for ipz,pz in enumerate(pz0):
        ax1=plt.subplot(gs[ipz,0])
        ax1.minorticks_on()
        plt.hist2d(pz.spec_full,getattr(pz,'z_'+x+'_full'),bins=200,range=((0,1.5),(0,1.5)))
        if (ipz==len(pz0)-1):
          plt.xlabel(r'$z_{\mathrm{spec}}$')
        else:
          ax1.set_xticklabels([])
        plt.ylabel(r'$z_\mathrm{'+x+'}$')
        cb = plt.colorbar(orientation='vertical')
        cb.set_label(pz.name)
      gs.update(wspace=0.,hspace=0.)
      plt.savefig('plots/photoz/spec_hist_'+x+'.png', bbox_inches='tight')
      plt.close()

    return


  @staticmethod
  def plot_pzrw(cat,pz,bins,w,label,edge):

    plt.figure(0,figsize=(10,5))
    ax=plt.subplot(1,2,1)

    col=['r','b','g']
    plt.hist(pz,bins=100,color='k',linestyle=('solid'),linewidth=1.,label='Full sample',histtype='step',normed=True)
    for i in range(cat.sbins):
      plt.hist(pz[bins==i],bins=100,color=col[i],linestyle=('solid'),linewidth=1.,label=r'$'+"{0:.2f}".format(edge[i])+'<$'+label.replace('_','-')+'$<'+"{0:.2f}".format(edge[i+1])+'$',histtype='step',weights=w[bins==i],normed=True)
      plt.hist(pz[bins==i],bins=100,color=col[i],linestyle=('dashed'),linewidth=1.,label='',histtype='step',normed=True)
    plt.legend(loc='upper right')
    #plt.ylim((0,2.5))
    plt.xlabel('z')
    plt.ylabel('n(z)')

    ax=plt.subplot(1,2,2)

    plt.axvline(x=1)
    for i in range(cat.sbins):
      plt.hist(w[bins==i],bins=50,alpha=.5,color=col[i],label=r'$'+"{0:.2f}".format(edge[i])+'<$'+label.replace('_','-')+'$<'+"{0:.2f}".format(edge[i+1])+'$',normed=True,histtype='stepfilled')
    plt.legend(loc='upper right')
    #plt.xlim((-1,4))
    plt.xlabel('w')
    plt.ylabel('n(w)')
    plt.savefig('plots/split/pzrw_'+cat.name+'_'+label.replace('_','-')+'.png', bbox_inches='tight')
    plt.close()

    return


  @staticmethod
  def plot_IA(r,out,err,label):

    col=['r','b','g','c']
    name=['gp','gx','ee','xx']
    for i in xrange(len(out)):
      plt.errorbar(r[out[i]>0]*(1.+.2*i),out[i][out[i]>0],yerr=err[i][out[i]>0],color=col[i],linestyle='',marker='o',label=name[i])
      plt.errorbar(r[out[i]<0]*(1.+.2*i),-out[i][out[i]<0],yerr=err[i][out[i]<0],color=col[i],linestyle='',marker='s',label='')
    plt.xscale('log')
    plt.yscale('log')
    # plt.legend(loc='upper right')
    plt.xlabel('R [Mpc/h]')
    plt.savefig('plots/IA_'+label+'.png', bbox_inches='tight')
    plt.close()

    return

  @staticmethod
  def plot_IA_lin(r,out,err,label):

    col=['r','b','g','c']
    name=['gp','gx','ee','xx']
    for i in xrange(len(out)):
      plt.errorbar(r*(1.+.2*i),-out[i],yerr=err[i]*np.sqrt(r),color=col[i],linestyle='',marker='o',label=name[i])
    plt.errorbar(r*(1.+.2*i),np.zeros(len(r)),color='k',linestyle='-',marker='',label='')
    plt.xscale('log')
    plt.ylim((-5,5))
    plt.legend(loc='upper right')
    plt.xlabel('R [Mpc/h]')
    plt.savefig('plots/IA_'+label+'.png', bbox_inches='tight')
    plt.close()

    return

  @staticmethod
  def imshow_symlog(my_matrix, logthresh=3):
    # From Freddie Nfbnm on stack overflow  http://stackoverflow.com/questions/11138706/

    plt.imshow( my_matrix,interpolation='nearest',origin='lower',vmin=np.min(my_matrix), vmax=np.max(my_matrix),norm=matplotlib.colors.SymLogNorm(10**-logthresh) )

    maxlog=int(np.ceil( np.log10(np.max(my_matrix)) ))
    minlog=int(np.ceil( np.log10(-np.min(my_matrix)) ))

    tick_locations=([-(10**x) for x in xrange(minlog,-logthresh-1,-1)]+[0.0]+[(10**x) for x in xrange(-logthresh,maxlog+1)] )

    plt.colorbar(ticks=tick_locations)

    return

  @staticmethod
  def bandpowers(ell,bp,bperr,bp2=None,bperr2=None):

    if len(np.shape(bp))==1:
      bins=1
    else:
      bins=len(bp)

    gs = gridspec.GridSpec(bins,bins)
    plt.figure(figsize=(10,10))

    col=['k','r','g','b','c','y']

    for i in range(bins):
      for j in range(bins):
        ax1=plt.subplot(gs[i,j])
        ax1.minorticks_on()

        ax1.errorbar(ell,bp[i,j,:],yerr=bperr[i,j,:],linestyle='',marker='o')
        if bp2 is not None:
          ax1.errorbar(ell+20,bp2[i,j,:],yerr=bperr2[i,j,:],linestyle='',marker='o')
        ax1.plot(ell,np.zeros(len(ell)),color='k')
        bintext='('+str(i)+','+str(j)+')'
        props = dict(boxstyle='square', lw=1.2,facecolor='white', alpha=1.)
        ax1.text(0.1, 0.95, bintext, transform=ax1.transAxes, fontsize=14,verticalalignment='top', bbox=props)
        plt.xlim((np.min(ell)-(ell[1]-ell[0]),np.max(ell)+(ell[1]-ell[0])))
        if j==0:
          plt.ylabel(r'$\ell(\ell+1)C_B(\ell)/2/\pi$')
        else:
          plt.ylabel('')
          ax1.set_yticklabels([])
        if i==bins-1:
          plt.xlabel(r'$\ell$')
        else:
          plt.xlabel('')
          ax1.set_xticklabels([])

    gs.update(wspace=0.,hspace=0.)
    plt.savefig('plots/xi/bandpowers_bins-'+str(bins)+'.png', bbox_inches='tight')
    plt.close()

    return
