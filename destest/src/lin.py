import numpy as np
from scipy.optimize import curve_fit
#import dill
import multiprocessing
import ctypes
import os

import catalog
import config
import fig
import txt

class UseError(Exception):
  def __init__(self, value):
    self.value = value
  def __str__(self):
    return repr(self.value)

# class single_arg_wrapper(object):
#   def __init__(self, func):
#     self.func = func

#   def __call__(self, task):
#     print 'wrap'
#     args, kwargs = task
#     print 'wrap2'
#     return self.func(*args, **kwargs)

# def run_dill_encoded(what):
#   fun, args = dill.loads(what)
#   print 'dill2',fun,args
#   return fun(*args)

# def apply_async(pool, fun, args):
#   return pool.apply_async(run_dill_encoded, (dill.dumps((fun, args)),))

# def _pickle_method(method):
#   func_name = method.im_func.__name__
#   obj = method.im_self
#   cls = method.im_class
#   return _unpickle_method, (func_name, obj, cls)

# def _unpickle_method(func_name, obj, cls):
#   for cls in cls.mro():
#     try:
#       func = cls.__dict__[func_name]
#     except KeyError:
#       pass
#     else:
#       break
#   return func.__get__(obj, cls)

# import copy_reg
# import types
# copy_reg.pickle(types.MethodType, _pickle_method, _unpickle_method)


class linear_methods(object):
  """
  Linear statistics and support functions for modifying shear.
  """

  @staticmethod
  def get_lin_e_w_ms(cat,xi=False,mock=False,mask=None,w1=None):
    """
    For a general CatalogStore object cat, return properly corrected ellipticity and weight, m/s values. Used in many functions.
    """

    if isinstance(cat,catalog.CatalogStore)|hasattr(cat,'cat'):
      mask=catalog.CatalogMethods.check_mask(cat.coadd,mask)
      if (cat.cat=='gal'):
        cattype,bs,wt,e1,e2,m1,m2,c1,c2,w=cat.cat,cat.bs,cat.wt,None,None,None,None,None,None,None
      else:
        cattype,bs,wt,e1,e2,m1,m2,c1,c2,w=cat.cat,cat.bs,cat.wt,cat.e1,cat.e2,cat.m1,cat.m2,cat.c1,cat.c2,cat.w        
    else:
      cattype,bs,wt,e1,e2,m1,m2,c1,c2,w=cat

    if mock:

      return e1[mask], e2[mask], w[mask], np.ones(np.sum(mask))

    elif cattype=='gal':

      if wt:
        return None, None, w[mask], None
      else:
        return None, None, np.ones(np.sum(mask)), None

    elif cattype==None:

      print 'None catalog type. Assuming no e, bias/sensitivity corections.'
      if wt:
        return np.ones(np.sum(mask)), np.ones(np.sum(mask)), cat.w[mask], np.ones(np.sum(mask))
      else:
        return np.ones(np.sum(mask)), np.ones(np.sum(mask)), np.ones(np.sum(mask)), np.ones(np.sum(mask))

    elif cattype not in ['i3','ng']:

      print 'Unknown catalog type. Assuming no bias/sensitivity corections.'
      if wt:
        return e1[mask], e2[mask], w[mask], np.ones(np.sum(mask))
      else:
        return e1[mask], e2[mask], np.ones(np.sum(mask)), np.ones(np.sum(mask))

    elif cattype=='i3':

      if bs:
        e1=e1-c1
        e2=e2-c2
        ms=np.sqrt((m1+1.)*(m2+1.))
      else:
        e1=e1
        e2=e2
        ms=np.ones(len(e1))     

    elif cattype=='ng':

      e1=e1
      e2=e2
      if bs:
        if xi:
          ms=np.sqrt(m1*m2)
        else:
          ms=(m1+m2)/2.
      else:  
        ms=np.ones(len(e1))

    if wt:
      w=w
    else:
      w=np.ones(len(e1))

    if w1 is not None:
      w=np.sqrt(w*w1)

    return e1[mask],e2[mask],w[mask],ms[mask]

  @staticmethod
  def calc_mean_stdev_rms_e(cat,mask=None,mock=False,full=True):
    """
    For a general CatalogStore object cat, return mean, std dev, rms of ellipticities.
    """

    if isinstance(cat,catalog.CatalogStore):
      mask=catalog.CatalogMethods.check_mask(cat.coadd,mask)

    e1,e2,w,ms=linear_methods.get_lin_e_w_ms(cat,mock=mock,mask=mask)

    wms=np.sum(w*ms)
    ww=np.sum(w**2)
    mean1=np.sum(w*e1)/wms
    mean2=np.sum(w*e2)/wms
    if full:
      std1=np.sqrt(np.sum(w*(e1-mean1)**2)/wms)
      std2=np.sqrt(np.sum(w*(e2-mean2)**2)/wms)
      rms1=np.sqrt(np.sum((w*e1)**2)/ww)
      rms2=np.sqrt(np.sum((w*e2)**2)/ww)

      return mean1,mean2,std1,std2,rms1,rms2
    else:
      return mean1,mean2

    return

  @staticmethod
  def calc_mean_stdev_rms(cat,x,mask=None,mock=False,full=True):
    """
    For a general CatalogStore object cat, return mean, std dev, rms of array x in cat.
    """

    if isinstance(cat,catalog.CatalogStore):
      mask=catalog.CatalogMethods.check_mask(cat.coadd,mask)

    w=linear_methods.get_lin_e_w_ms(cat,mock=mock,mask=mask)[2]

    mean=np.sum(w*x[mask])/np.sum(w)
    if full:
      std=np.sqrt(np.sum(w*(x[mask]-mean)**2)/np.sum(w))
      rms=np.sqrt(np.sum((w*x[mask])**2)/np.sum(w**2))
      return mean,std,rms
    else:
      return mean

    return


  @staticmethod    
  def find_bin_edges(x,nbins,w=None):
    """
    For an array x, returns the boundaries of nbins equal (possibly weighted by w) bins.
    """

    if w is None:
      xs=np.sort(x)
      r=np.linspace(0.,1.,nbins+1.)*(len(x)-1)
      return xs[r.astype(int)]

    fail=False
    ww=np.sum(w)/nbins
    i=np.argsort(x)
    k=np.linspace(0.,1.,nbins+1.)*(len(x)-1)
    k=k.astype(int)
    r=np.zeros((nbins+1))
    ist=0
    for j in xrange(1,nbins):
      # print k[j],r[j-1]
      if k[j]<r[j-1]:
        print 'Random weight approx. failed - attempting brute force approach'
        fail=True
        break
      w0=np.sum(w[i[ist:k[j]]])
      if w0<=ww:
        for l in xrange(k[j],len(x)):
          w0+=w[i[l]]
          if w0>ww:
            r[j]=x[i[l]]
            ist=l
            break
      else:
        for l in xrange(k[j],0,-1):
          w0-=w[i[l]]
          if w0<ww:
            r[j]=x[i[l]]
            ist=l
            break

    if fail:

      ist=np.zeros((nbins+1))
      ist[0]=0
      for j in xrange(1,nbins):
        wsum=0.
        for k in xrange(ist[j-1].astype(int),len(x)):
          wsum+=w[i[k]]
          if wsum>ww:
            r[j]=x[i[k-1]]
            ist[j]=k
            break

    r[0]=x[i[0]]
    r[-1]=x[i[-1]]

    return r

  @staticmethod
  def binned_mean_e(bin,cat,mask=None,mock=False):

    if isinstance(cat,catalog.CatalogStore):
      mask=catalog.CatalogMethods.check_mask(cat.coadd,mask)

    y_mean1=[]
    y_err1=[]
    y_mean2=[]
    y_err2=[]

    for i in xrange(config.cfg.get('lbins',10)):
      mask0=(bin==i)&mask
      mean1,mean2,std1,std2,rms1,rms2=linear_methods.calc_mean_stdev_rms_e(cat,mask0,mock=mock)
      y_mean1=np.append(y_mean1,mean1)
      y_err1=np.append(y_err1,std1/np.sqrt(np.sum(mask0)))
      y_mean2=np.append(y_mean2,mean2)
      y_err2=np.append(y_err2,std2/np.sqrt(np.sum(mask0)))

    return y_mean1,y_err1,y_mean2,y_err2

  @staticmethod
  def binned_mean_x(bin,x,cat,mask=None,mock=False):

    if isinstance(cat,catalog.CatalogStore):
      mask=catalog.CatalogMethods.check_mask(cat.coadd,mask)

    x_mean=[]
    x_err=[]

    for i in xrange(config.cfg.get('lbins',10)):
      mask0=(bin==i)&mask
      mean,std,rms=linear_methods.calc_mean_stdev_rms(cat,x,mask0,mock=mock)
      x_mean=np.append(x_mean,mean)
      x_err=np.append(x_err,std/np.sqrt(np.sum(mask0)))

    return x_mean,x_err


  @staticmethod
  def bin_means(x,cat,w=None,mask=None,mock=False,log=False,noe=False,y=None):
    """
    For array x in CatalogStore object cat, calculate the means of shear in equal bins of x. Returns errors in both x and y directions.
    """

    if isinstance(cat,catalog.CatalogStore):
      mask=catalog.CatalogMethods.check_mask(cat.coadd,mask)

    if w is not None:
      edge=linear_methods.find_bin_edges(x[mask],config.cfg.get('lbins',10),w[mask])
    else:
      edge=linear_methods.find_bin_edges(x[mask],config.cfg.get('lbins',10))

    xbin=np.digitize(x,edge)-1

    x_mean,x_err=linear_methods.binned_mean_x(xbin,x,cat,mask,mock=mock)
    if np.sum(np.isnan(x_mean))>0:
      return None,None,None,None,None,None
    if noe:
      if y is None:
        return x_mean,x_err
      else:
        e1_mean,e1_err=linear_methods.binned_mean_x(xbin,y,cat,mask,mock=mock)
        e2_mean=np.zeros(len(e1_mean))
        e2_err=np.zeros(len(e1_mean))
    else:
      e1_mean,e1_err,e2_mean,e2_err=linear_methods.binned_mean_e(xbin,cat,mask,mock=mock)

    return x_mean,x_err,e1_mean,e1_err,e2_mean,e2_err

class fitting(object):

  @staticmethod
  def lin_fit(x,y,sig):
    """
    Find linear fit with errors for two arrays.
    """

    def func(x,m,b):
      return m*x+b

    params=curve_fit(func,x,y,p0=(0.,0.),sigma=sig)

    m,b=params[0]
    merr,berr=np.sqrt(np.diagonal(params[1]))

    return m,b,merr,berr

class hist(object):

  @staticmethod
  def hist_tests(cat,cat2=None,cols=None,mask=None,mask2=None,p=None):
    """
    Loop over array cols, containing stored catalog column variables in CatalogStore object cat. Optionally mask the elements used.

    Produces plots of 1D histograms for each element in cols.
    """

    if p is not None:
      jobs=[]
      p=multiprocessing.Pool(processes=config.cfg.get('proc',32),maxtasksperchild=config.cfg.get('task',None))

    mask=catalog.CatalogMethods.check_mask(cat.coadd,mask,p=p)

    x1name=cat.name
    if cols is None:
      cols=catalog.CatalogMethods.get_cat_colnames(cat)
    if cat2 is not None:
      x2name=cat2.name
      mask2=catalog.CatalogMethods.check_mask(cat2.coadd,mask2,p=p)

    for val in cols:

      print 'hist',val

      x1=getattr(cat,val)
      if cat2 is not None:
        x2=getattr(cat2,val)

      if p is not None:
        if cat2 is None:
          if cat.wt:
            job=p.apply_async(hist_tests_base,[val,x1,mask,x1name],{'w1':cat.w})
          else:
            job=p.apply_async(hist_tests_base,[val,x1,mask,x1name],{})
        else:
          if cat.wt:
            job=p.apply_async(hist_tests_base,[val,x1,mask,x1name], {'w1':cat.w,'x2':x2,'mask2':mask2,'x2name':cat2.name,'w2':cat2.w})
          else:
            job=p.apply_async(hist_tests_base,[val,x1,mask,x1name], {'x2':x2,'mask2':mask2,'x2name':cat2.name})
        jobs.append(job)
      else:
        if cat2 is None:
          if cat.wt:
            hist_tests_base(val,x1,mask,x1name,w1=cat.w)
          else:
            hist_tests_base(val,x1,mask,x1name)
        else:
          if cat.wt:
            hist_tests_base(val,x1,mask,x1name,w1=cat.w,x2=x2,mask2=mask2,x2name=cat2.name,w2=cat2.w)
          else:
            hist_tests_base(val,x1,mask,x1name,x2=x2,mask2=mask2,x2name=cat2.name)

    if p is not None:
      for job in jobs:
        print job.get()

      p.close()
      p.join()

    return

  @staticmethod
  def hist_2D_tests(cat,cat2=None,colsx=None,colsy=None,mask=None,mask2=None,match_col=None,p=None):
    """
    Loop over array cols(x|y), containing stored catalog column variables in CatalogStore object cat (and cat2). Optionally mask the elements used. 

    Produces plots of 2D histograms for each cross pair in valsx and colsy. If both cat and cat2 provided, optionally provide an array name in cat with which to match the two catalogs (default value of None indicates to use the coadd ids). This is useful for matching between data releases.
    """

    if p is not None:
      jobs=[]
      p=multiprocessing.Pool(processes=config.cfg.get('proc',32),maxtasksperchild=config.cfg.get('task',None))

    jobs=[]
    mask=catalog.CatalogMethods.check_mask(cat.coadd,mask,p=p)

    if colsx is None:
      colsx=catalog.CatalogMethods.get_cat_colnames(cat)

    if cat2!=None:
      mask2=catalog.CatalogMethods.check_mask(cat2.coadd,mask2,p=p)
      if colsy is None:
        colsy=catalog.CatalogMethods.get_cat_colnames(cat2)

      if match_col is None:
        coadd=cat.coadd 
      else:
        coadd=getattr(cat,match_col)
        mask=mask&(coadd>0)

      yname=cat2.name

      s1a,s2a=catalog.CatalogMethods.sort2(coadd[mask],cat2.coadd[mask2])
      shared_array_base=multiprocessing.Array(ctypes.c_double, len(s1))
      s1=np.ctypeslib.as_array(shared_array_base.get_obj())
      s1[:]=s1a
      shared_array_base=multiprocessing.Array(ctypes.c_double, len(s1))
      s2=np.ctypeslib.as_array(shared_array_base.get_obj())
      s2[:]=s2a

    else:
      cat2=cat
      colsy=colsx

    for ix,x in enumerate(colsx):
      for iy,y in enumerate(colsy):   

        if cat2 is cat:
          if (ix>=iy):
            continue
        else:
          if ix!=iy:
            continue

        x1=getattr(cat,x)
        y1=getattr(cat2,y)

        if p is not None:
          if cat2 is cat:
            job=p.apply_async(hist_2D_tests_base,[x,y,x1,y1,cat.name,cat2.name,mask],{})
          else:
            job=p.apply_async(hist_2D_tests_base,[x,y,x1,y1,cat.name,cat2.name,mask], {'mask2':mask2,'s1':s1,'s2':s2})
          jobs.append(job)
        else:
          if cat2 is cat:
            hist_2D_tests_base(x,y,x1,y1,cat.name,cat2.name,mask)
          else:
            hist_2D_tests_base(x,y,x1,y1,cat.name,cat2.name,mask,mask2,s1=s2,s2=s2)

    if p is not None:
      for job in jobs:
        print job.get()

      p.close()
      p.join()

    return

  @staticmethod
  def tile_tests(cat):
    """
    Loop over cols that have had means computed tile-by-tile with summary_stats.tile_stats(). This expects the output from that function to exist.

    Produces plots of 1D histograms for each mean value tile-by-tile.
    """

    try:
      cat.tilecols
    except NameError:
      print 'you must first call lin.summary_stats.tile_stats(cat)'
      return

    for i,x in enumerate(cat.tilecols):

      print 'tile hist',x

      x1=cat.tilemean[:,i]
      if config.log_val.get(x,False):
        x1=np.log10(x1)
      fig.plot_methods.plot_hist(x1,name=cat.name,label=x,bins=20,tile='mean')

      x1=cat.tilestd[:,i]
      if config.log_val.get(x,False):
        x1=np.log10(x1)
      fig.plot_methods.plot_hist(x1,name=cat.name,label=x,bins=20,tile='std')

      x1=cat.tilerms[:,i]
      if config.log_val.get(x,False):
        x1=np.log10(x1)
      fig.plot_methods.plot_hist(x1,name=cat.name,label=x,bins=20,tile='rms')

    return

  @staticmethod
  def tile_tests_2D(cat,cat2=None):
    """
    Loop over cols that have had means computed tile-by-tile with summary_stats.tile_stats(). This expects the output from that function to exist.

    Produces plots of 2D histograms for each cross pair in valsx and valsy tile-by-tile.
    """

    try:
      cat.tilecols
    except NameError:
      print 'you must first call lin.summary_stats.tile_stats(cat)'
      return

    if cat2 is not None:
      try:
        cat2.tilecols
      except NameError:
        print 'you must first call lin.summary_stats.tile_stats(cat)'
        return

      if cat2.tilecols!=cat.tilecols:
        print 'tile col lists do not agree between cat and cat2'
        return
    else:
      cat2=cat

    for ix,x in enumerate(cat.tilecols):
      for iy,y in enumerate(cat2.tilecols):

        if cat2 is cat:
          if (ix>=iy):
            continue
        else:
          if ix!=iy:
            continue

        print 'tile hist 2D',x,y

        x1=cat.tilemean[:,ix]
        if config.log_val.get(x,False):
          x1=np.log10(x1)
        y1=cat2.tilemean[:,iy]
        if config.log_val.get(y,False):
          y1=np.log10(y1)

        fig.plot_methods.plot_2D_hist(x1,y1,bins=20,xname=cat.name,yname=cat2.name,xlabel=x,ylabel=y,xtile='mean',ytile='mean')

    return

class footprint(object):

  @staticmethod
  def hexbin_tests(cat,cols=None,mask=None,p=None):
    """
    Produces a set of hexbin plots (mean value in cells across the survey footprint) for each value listed in cols and stored in CatalogeStore object cat. Optionally mask the arrays listed in cols.
    """

    if p is not None:
      jobs=[]
      p=multiprocessing.Pool(processes=config.cfg.get('proc',32),maxtasksperchild=config.cfg.get('task',None))

    mask=catalog.CatalogMethods.check_mask(cat.coadd,mask,p=p)

    if cols is None:
      cols=catalog.CatalogMethods.get_cat_colnames(cat)

    for x in cols:
      x1=getattr(cat,x)

      if p is not None:
        job=p.apply_async(hexbin_tests_base,[x,x1,cat.ra,cat.dec,cat.name,mask],{})
        jobs.append(job)
      else:
        hexbin_tests_base(x,x1,cat.ra,cat.dec,cat.name,mask)

    if p is not None:
      for job in jobs:
        print job.get()

      p.close()
      p.join()

    return

  @staticmethod
  def tile_hexbin_tests(cat,mask=None,p=None):
    """
    A version of hexbin_tests that maps mean value tile-by-tile instead of by hexbin cell.
    """

    if p is not None:
      jobs=[]
      p=multiprocessing.Pool(processes=config.cfg.get('proc',32),maxtasksperchild=config.cfg.get('task',None))

    mask=catalog.CatalogMethods.check_mask(cat.coadd,mask,p=p)

    try:
      cat.tilecols
    except NameError:
      print 'you must first call lin.summary_stats.tile_stats(cat)'
      return

    for j,val in enumerate(cat.tilecols):

      if p is not None:
        job=p.apply_async(tile_hexbin_tests_base,[cat.tile,cat.tilelist,cat.tilemean,cat.tilestd,cat.tilerms,cat.ra,cat.dec,val,j,cat.name,mask],{})
        jobs.append(job)
      else:
        tile_hexbin_tests_base(cat.tile,cat.tilelist,cat.tilemean,cat.tilestd,cat.tilerms,cat.ra,cat.dec,val,j,cat.name,mask)

    if p is not None:
      for job in jobs:
        print job.get()

      p.close()
      p.join()

    return

  @staticmethod
  def footprint_tests(cat,vals,mask=None,label='',p=None):
    """
    If vals==[], produces a galaxy density plot over the survey fottprint for the catalog with mask. If vals contains a list of flag columns, it maps the density of objects that fail each flag value.
    """

    mask=catalog.CatalogMethods.check_mask(cat.coadd,mask,p=p)

    if vals==[]:

      fig.plot_methods.plot_footprint(cat.ra[mask],cat.dec[mask],cat.name,label=label)

    else:
      print 'entered error fooprint'
      for val in vals:
        if p is not None:
          jobs=[]
          p=multiprocessing.Pool(processes=config.cfg.get('proc',32),maxtasksperchild=config.cfg.get('task',None))

        print 'error fooprint',val,os.getpid()
        flag=getattr(cat,val)
        maxbit=summary_stats.n_bits_array(cat,val)
        for i in xrange(maxbit):
          print 'error fooprint',val,i,os.getpid()
          if p is not None:
            print 'error fooprint inside p check',val,i,os.getpid()
            job=p.apply_async(footprint_tests_base,[flag,val,i,cat.ra,cat.dec,cat.name,mask],{})
            jobs.append(job)
            print 'error fooprint after jobs',val,i,os.getpid()
          else:
            footprint_tests_base(flag,val,i,cat.ra,cat.dec,cat.name,mask)

        print 'error fooprint before job end loop',os.getpid()
        if p is not None:
          for job in jobs:
            print 'error fooprint before job get',job,os.getpid()
            print job.get()
            print 'error fooprint after job get',job,os.getpid()

          print 'before job close',os.getpid()
          p.close()
          print 'after job close',os.getpid()
          p.join()
          print 'after job join',os.getpid()

    return

class summary_stats(object):

  @staticmethod
  def n_bits_array(cat,arr,mask=None):
    """
    Convenience function to return maximum flag bit.
    """

    mask=catalog.CatalogMethods.check_mask(cat.coadd,mask)

    arr1=getattr(cat,arr)
    print arr1,mask

    return len(bin(np.max(arr1[mask])))-2

  @staticmethod
  def i3_flags_vals_check(cat,flags=['error','info'],mask=None,label=''):
    """
    Produce a summary of error properties in the catalog. Identifies nan or inf values in arrays and lists the distribution of objects that fail flags.
    """

    mask=catalog.CatalogMethods.check_mask(cat.coadd,mask)

    for flag in flags:
      if flag==flags[0]:
        txt.write_methods.heading(flag+' flags',cat,label='flags_dist'+label,create=True)
      else:
        txt.write_methods.heading(flag+' flags',cat,label='flags_dist'+label,create=False)
      for i in xrange(summary_stats.n_bits_array(cat,flag)):
        total=np.sum((getattr(cat,flag) & 2**i) != 0)
        unique=np.sum(getattr(cat,flag) == 2**i)
        txt.write_methods.write_append(str(i)+'  '+str(total)+'  '+str(unique)+'  '+str(np.around(1.*total/len(cat.coadd),5)),cat,label='flags_dist'+label,create=False)

    txt.write_methods.heading('checking for bad values',cat,label='flags_dist'+label,create=False)
    for x in dir(cat):
      obj = getattr(cat,x)
      if isinstance(obj,np.ndarray):
        if len(obj)==len(cat.coadd):
          if isinstance(obj[0], str)|('__' in x):
            continue
          line=x
          if np.isnan(obj[mask]).any():
            line+='  !!!NANS!!!'
          if np.isinf(obj[mask]).any():
            line+='  !!!INF!!!'
          txt.write_methods.write_append(line,cat,label='flags_dist'+label,create=False)

    return

  @staticmethod
  def val_stats(cat,cols=None,mask=None,label=''):
    """
    Produce a summary of basic properties in the catalog. Writes out the mean, std dev, and rms of cols, along with number of galaxies for some optional mask.
    """

    mask=catalog.CatalogMethods.check_mask(cat.coadd,mask)

    txt.write_methods.heading('Summary',cat,label='summary'+label,create=True)
    txt.write_methods.heading('num gals  '+str(np.sum(mask)),cat,label='summary'+label,create=False)
    line=['quantity', 'mean', 'std', 'rms', 'min', 'max']
    txt.write_methods.heading("".join(word.ljust(22) for word in line),cat,label='summary'+label,create=False)

    if cols is None:
      cols=catalog.CatalogMethods.get_cat_colnames(cat)

    for val in cols:
      if val=='e1':
        mean,e2,std,e2_std,rms,e2_rms=linear_methods.calc_mean_stdev_rms_e(cat,mask)
      elif val=='e2':
        e1,mean,e1_std,std,e1_rms,rms=linear_methods.calc_mean_stdev_rms_e(cat,mask)
      else:
        mean,std,rms=linear_methods.calc_mean_stdev_rms(cat,getattr(cat,val),mask)
      line=[val,str(mean),str(std),str(rms),str(np.min(getattr(cat,val))),str(np.max(getattr(cat,val)))]
      txt.write_methods.write_append("".join(word.ljust(22) for word in line),cat,label='summary'+label,create=False)

    return

  @staticmethod
  def tile_stats(cat,cols=None,mask=None,p=None):
    """
    Produce a summary of basic properties tile-by-tile in the catalog. Writes out the mean, std dev, and rms of the values listed in cols for some mask.
    """

    if p is not None:
      jobs=[]
      p=multiprocessing.Pool(processes=config.cfg.get('proc',32),maxtasksperchild=config.cfg.get('task',None))

    mask=catalog.CatalogMethods.check_mask(cat.coadd,mask,p=p)

    if cols is None:
      cols=catalog.CatalogMethods.get_cat_colnames(cat)

    cat.tilelist=np.unique(cat.tile)
    cat.tilecols=cols
    cat.tilenums=np.zeros(len(cat.tilelist))
    cat.tilemean=np.zeros((len(cat.tilelist),len(cols)))
    cat.tilestd=np.zeros((len(cat.tilelist),len(cols)))
    cat.tilerms=np.zeros((len(cat.tilelist),len(cols)))

    for i,x in enumerate(cat.tilelist):
      valstore=[getattr(cat,val) for j,val in enumerate(cols)]
      if p is not None:
        cat0=[cat.cat,cat.bs,cat.wt,cat.e1,cat.e2,cat.m1,cat.m2,cat.c1,cat.c2,cat.w]
        job=p.apply_async(tile_stats_base,[cat0,valstore,cat.tile,x,i,cols,mask],{})
        jobs.append(job)
      else:
        tmp,cat.tilenums[i],cat.tilemean[i,:],cat.tilestd[i,:],cat.tilerms[i,:]=tile_stats_base(cat,valstore,cat.tile,x,i,cols,mask)

    if p is not None:
      for job in jobs:
        i,a,b,c,d=job.get()
        cat.tilenums[i],cat.tilemean[i,:],cat.tilestd[i,:],cat.tilerms[i,:]=a,b,c,d

      p.close()
      p.join()

    line='#tile  numgal  '
    for val in cols:
      line+=val+'  '+val+'_std  '+val+'_rms  '

    txt.write_methods.write_append(line,cat,label='tile_stats',create=True)

    for i,x in enumerate(cat.tilelist):
      line=x+'  '+str(cat.tilenums[i])+'  '
      for j,val in enumerate(cols):
        line+=str(np.around(cat.tilemean[i,j],5))+'  '+str(np.around(cat.tilestd[i,j],5))+'  '+str(np.around(cat.tilerms[i,j],5))+'  '
      txt.write_methods.write_append(line,cat,label='tile_stats',create=False)

    return

  def load_tile_stats(cat,cols=None):

    tmp=np.genfromtxt('tile_stats_y1_i3_sv_v1.txt',names=True)
    tmp=np.genfromtxt(write_methods.get_file(cat,label=label),names=True)
    for i,x in enumerate(tmp.dtype.names):
      if i<2:
        continue
      for j,y in enumerate(tmp.dtype.names):
        if j<2:
          continue
        a=x.split('_')
        b=y.split('_')
        if len(a)>1:
          # fig.plot_methods.plot_hist(tmp[x],name=i3.name,label=a[0],bins=200,tile=a[1])
          if len(b)<1:
            fig.plot_methods.plot_2D_hist(tmp[x],tmp[y],bins=200,xname=i3.name,yname=i3.name,xlabel=a[0],ylabel=b[0],xtile=a[1],ytile=b[1])
          else:
            fig.plot_methods.plot_2D_hist(tmp[x],tmp[y],bins=200,xname=i3.name,yname=i3.name,xlabel=a[0],ylabel=b[0],xtile=a[1],ytile='mean')
        else:
          # fig.plot_methods.plot_hist(tmp[x],name=i3.name,label=a[0],bins=200,tile='mean')
          if len(b)>1:
            fig.plot_methods.plot_2D_hist(tmp[x],tmp[y],bins=200,xname=i3.name,yname=i3.name,xlabel=a[0],ylabel=b[0],xtile='mean',ytile=b[1])
          else:
            fig.plot_methods.plot_2D_hist(tmp[x],tmp[y],bins=200,xname=i3.name,yname=i3.name,xlabel=a[0],ylabel=b[0],xtile='mean',ytile='mean')



def tile_stats_base(cat,valstore,tiles,tile,i,cols,mask):

  tilemean=np.zeros(len(cols))
  tilestd=np.zeros(len(cols))
  tilerms=np.zeros(len(cols))

  mask0=mask&(tiles==tile)
  tilenums=np.sum(mask0)
  if tilenums>0:
    for j,val in enumerate(cols):
      if val=='e1':
        e1,e2,e1_std,e2_std,e1_rms,e2_rms=linear_methods.calc_mean_stdev_rms_e(cat,mask0)
        tilemean[j]=e1
        tilestd[j]=e1_std
        tilerms[j]=e1_rms
      elif val=='e2':
        e1,e2,e1_std,e2_std,e1_rms,e2_rms=linear_methods.calc_mean_stdev_rms_e(cat,mask0)
        tilemean[j]=e2
        tilestd[j]=e2_std
        tilerms[j]=e2_rms
      else:
        x1=linear_methods.calc_mean_stdev_rms(cat,valstore[j],mask0)
        tilemean[j]=x1[0]
        tilestd[j]=x1[1]
        tilerms[j]=x1[2]

  return i,tilenums,tilemean,tilestd,tilerms

def hist_tests_base(val,x1,mask,x1name,w1=None,x2=None,mask2=None,x2name=None,w2=None):

  if isinstance(x1[0], basestring):
    print 'string values'
    return  
  if (np.sum(np.isinf(x1))>0)|(np.sum(np.isnan(x1))>0):
    print 'bad values'
    return

  if config.log_val.get(val,False):
    x1=np.log10(x1[mask&(x1>0)])

  if x2 is None:
    if w1 is None:
      fig.plot_methods.plot_hist(x1,name=x1name,label=val)
    else:
      fig.plot_methods.plot_hist(x1,name=x1name,label=val,w=w1[mask])
  else:
    if (np.sum(np.isinf(x2))>0)|(np.sum(np.isnan(x2))>0):
      print 'bad values'
      return
    if config.log_val.get(val,False):
      x2=np.log10(x2[mask2])

    if w1 is None:
      fig.plot_methods.plot_comp_hist(x1,x2,name=x1name,name2=x2name,label=val)
    else:
      fig.plot_methods.plot_comp_hist(x1,x2,name=x1name,name2=x2name,label=val,w1=w1[mask],w2=w2[mask2])

  return

def hist_2D_tests_base(x,y,x1,y1,xname,yname,mask,mask2=None,s1=None,s2=None):

  x1=x1[mask]
  if config.log_val.get(x,False):
    x1=np.log10(x1)

  if s1 is None:
    y1=y1[mask]
  else:
    y1=y1[mask2]
    x1=x1[s1]
    y1=y1[s2]
  if config.log_val.get(y,False):
    y1=np.log10(y1)

  fig.plot_methods.plot_2D_hist(x1,y1,xname=xname,yname=yname,xlabel=x,ylabel=y)

  return

def hexbin_tests_base(x,x1,ra,dec,xname,mask):

  if config.log_val.get(x,False):
    x1=np.log10(x1)

  fig.plot_methods.plot_hexbin(x1[mask],ra[mask],dec[mask],name=xname,label=x)

  return

def tile_hexbin_tests_base(tiles,tilelist,tilemean,tilestd,tilerms,ra,dec,val,j,name,mask):

  x1=np.zeros(np.sum(mask))
  for i in xrange(len(tilelist)):
    mask0=(tiles[mask]==tilelist[i])
    x0=tilemean[i,j]
    if config.log_val.get(val,False):
      x0=np.log10(x0)
    x1[mask0]=np.ones(np.sum(mask0))*x0

  fig.plot_methods.plot_hexbin(x1,ra[mask],dec[mask],name=name,label=val,tile='mean')

  x1=np.zeros(np.sum(mask))
  for i in xrange(len(tilelist)):
    mask0=(tiles[mask]==tilelist[i])
    x0=tilestd[i,j]
    if config.log_val.get(val,False):
      x0=np.log10(x0)
    x1[mask0]=np.ones(np.sum(mask0))*x0

  fig.plot_methods.plot_hexbin(x1,ra[mask],dec[mask],name=name,label=val,tile='std')

  x1=np.zeros(np.sum(mask))
  for i in xrange(len(tilelist)):
    mask0=(tiles[mask]==tilelist[i])
    x0=tilerms[i,j]
    if config.log_val.get(val,False):
      x0=np.log10(x0)
    x1[mask0]=np.ones(np.sum(mask0))*x0

  fig.plot_methods.plot_hexbin(x1,ra[mask],dec[mask],name=name,label=val,tile='rms')

  return

def footprint_tests_base(flag,val,i,ra,dec,name,mask):

  print 'inside base func',os.getpid()
  if np.sum((flag & 2**i) != 0)>1000:
    fig.plot_methods.plot_footprint(ra[((flag & 2**i) != 0)&mask],dec[((flag & 2**i) != 0)&mask],name,label=val+'_'+getattr(config,val+'_name').get(i))
  print 'end of base func',os.getpid()

  return
