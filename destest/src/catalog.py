import numpy as np
import glob
import os
import fitsio as fio
try:
  import pandas as pd
except:
  print "No Pandas"
  pd=None

import config
import pickle
import multiprocessing
import ctypes

noval=999999

class CatalogStore(object):
  """
  A flexible class that reads and stores shear catalog information. An example is given for im3shape in the main testsuite.py.

  Use:

  :name:      str - A label for identifying the catalog in output. Multiple catalog classes can be defined, but each should have a unique name.
  :setup:     bool - Read in data and setup class structure? If false, only basic structure is created. Used for custom catalogs to feed into functions that expect a CatalogStore object.
  :cutfunc:   func - Pre-defined function for determining which objects in the catalog to store. See examples in CatalogMethods below.
  :cattype:   str - Currently one of 'im', 'ng', or 'gal' for im3shape, ngmix or a galaxy/lens catalog, respectively. This determines the dictionary in config.py used to match shortnames for quantities with column names in the catalog files.
  :cols:      [str] - Array of shortnames of columns to save from files. If None, saves all columns in dictionary.
  :catdir:    str - Directory to look for catalog files. Currently looks for files with *fits* extensions. Supports multiple files.
  :goldfile:  str - Common 'gold' file in combined, flatcat SV format for ngmix, im3shape. Not fully migrated from SV version of code. Will update when common file format for Y1 agreed upon.
  :catfile:   str - Point to a single catalog file. Also expects fits currently, but simple to add option for text files with column headers or h5 if useful.
  :ranfile:   str - Random file to be used with catalog type 'gal'. Currently rigid format, expecting numpy array of ra,dec.
  :jkbuild:   bool - Build jackknife regions from kmeans? Currently partly implemented - 150 regions. Can be more flexible if needed.
  :jkload:    bool - Load jackknife regions from file. Currently not fully migrated from SV code.
  :tiles:     [str] - Array of DES tile names to load, maching to multi-file fits format for catalog. Could be made more flexible to match strings in filenames generally.
  :release:   str - Release version, e.g. 'y1', 'sv'. Used for identifying release-specific files like systematics maps.

  """

  def __init__(self,name,setup=True,cutfunc=None,cattype=None,cols=None,catdir=None,goldfile=None,catfile=None,ranfile=None,jkbuild=False,jkload=False,tiles=None,release='y1',maxrows=150000000,maxiter=999999,exiter=-1,p=None,ext='*fit*',hdu=-1):

    if setup:
      # Populate catalog on object creation

      # Select column name lookup dict
      if cattype=='i3':
        table=config.i3_col_lookup
      elif cattype=='i3epoch':
        table=config.i3_epoch_col_lookup
      elif cattype=='ng':
        table=config.ng_col_lookup
      elif cattype=='gal':
        table=config.gal_col_lookup
      elif cattype=='redmapper':
        table=config.redmapper_col_lookup
      elif cattype=='truth':
        table=config.truth_col_lookup
      elif cattype=='buzzard':
        table=config.buzzard_col_lookup
      elif cattype=='psf':
        table=config.psf_col_lookup
      elif cattype=='gold':
        table=config.gold_col_lookup
      else:
        raise CatValError('No catalog type cattype specified.')

      # Choose default selections for cuts and columns to read
      if cutfunc is None:
        print 'Assuming no mask of catalog.'
        cutfunc=CatalogMethods.final_null_cuts()

      if cols is None:
        print 'Assuming all columns in dictionary.'
        cols=np.array(list(table.keys()))
      elif cols==-1:
        print 'Using all columns from file'
        cols1=None

      if goldfile is not None:
        print 'not ready to use matched catalog format in this version'
        if (i3file is None)|(ngfile is None):
          raise CatValError('Assumed flat catalog style and no im3shape or ngmix file specified.')

        cols1=[table.get(x,x) for x in cols]
        for i,x in enumerate(CatalogMethods.get_cat_cols_matched(catdir,goldfile,catfile,cols1,table,cuts,full=full,ext=ext)):
          setattr(self,cols[i],x)

      elif (catfile!=None)|(catdir!=None):
        if (catfile!=None)&(catdir!=None):
          raise CatValError('Both catfile and catdir specified.')
        if catdir is None:
          catdir=catfile
        else:
          if not os.path.exists(catdir):
            print catdir+'does not exist!'
            return
          catdir=catdir+ext

        # Read in columns from file(s)
        if 'cols1' not in locals():
          cols1=[table.get(x,x) for x in cols]

        cols2,catcols,filenames,filenums=CatalogMethods.get_cat_cols(catdir,cols1,table,cutfunc,tiles,maxrows=maxrows,maxiter=maxiter,exiter=exiter,hdu=hdu)
        if cols1 is None:
          cols=cols2
        for i,x in enumerate(catcols):
          if isinstance(x[0], basestring)|(p is None):
            if len(np.shape(x))>2:
              continue
            elif len(np.shape(x))==2:
              for j in range(np.shape(x)[1]):
                setattr(self,cols[i]+'_'+str(j+1),x[:,j].copy())
            else:
              setattr(self,cols[i],x.copy())
          else:
            if len(np.shape(x))>1:
              for j in range(len(np.shape(x))):
                setattr(self,cols[i]+'_'+str(j),self.add_shared_array(len(filenames),x[:,j],p))
            else:
              setattr(self,cols[i],self.add_shared_array(len(filenames),x,p))
        self.filename=filenames
        self.filenum=filenums

      else:
        raise CatValError('Please specify the source of files: catfile, catdir, or goldfile/i3file/ngfile')

      #Generate id column if no unique id specified
      if 'coadd' not in cols:
        self.coadd=self.add_shared_array(len(filenames),np.arange(len(filenames)),p)

      if cattype=='ng':
        self.e1=self.mcal_g_1
        self.mcal_g_1=None
        self.e2=self.mcal_g_2
        self.mcal_g_2=None
        self.psf1=self.psfrec_g_1
        self.psfrec_g_1=None
        self.psf2=self.psfrec_g_2
        self.psfrec_g_2=None
        self.coadd=self.id
        self.id=None

      #Generate derived quantities
      if cattype in ['i3','ng']:
        if ('e1' in cols)&('e2' in cols):
          self.pos=self.add_shared_array(len(filenames),0.5*np.arctan2(self.e2,self.e1)+np.pi/2.,p)
          self.e=self.add_shared_array(len(filenames),np.sqrt(self.e1**2.+self.e2**2.),p)
        if ('m1' in cols):
          self.m2=self.m1
        else:
          self.m1=None
          self.m2=None
        if ('c1' not in cols):
          self.c1=None
          self.c2=None
        if ('w' not in cols):
          self.w=None
        if ('psf1' in cols)&('psf2' in cols):
          self.psfpos=self.add_shared_array(len(filenames),0.5*np.arctan2(self.psf2,self.psf1)+np.pi/2.,p)
          self.dpsf=self.add_shared_array(len(filenames),self.psf1-self.psf2,p)
          self.psfe=self.add_shared_array(len(filenames),np.sqrt(self.psf1**2.+self.psf2**2.),p)
        if ('hsmpsf1' in cols)&('hsmpsf2' in cols):
          self.hsmpsfpos=self.add_shared_array(len(filenames),0.5*np.arctan2(self.hsmpsf2,self.hsmpsf1)+np.pi/2.,p)
          self.hsmdpsf=self.add_shared_array(len(filenames),self.hsmpsf1-self.hsmpsf2,p)
          self.hsmpsfe=self.add_shared_array(len(filenames),np.sqrt(self.hsmpsf1**2.+self.hsmpsf2**2.),p)
        if ('psf1_exp' in cols)&('psf2_exp' in cols):
          self.psfpos=self.add_shared_array(len(filenames),0.5*np.arctan2(self.psf2_exp,self.psf1_exp)+np.pi/2.,p)
          self.dpsf=self.add_shared_array(len(filenames),self.psf1_exp-self.psf2_exp,p)
        if 'fluxfrac' in cols:
          self.invfluxfrac=self.add_shared_array(len(filenames),1.001-self.fluxfrac,p)
        if 'like' in cols:
          self.nlike=-self.like
      if cattype=='i3':
        if ('dflux' in cols)&('bflux' in cols)&(~('bfrac' in cols)):
          self.bfrac=np.zeros(len(self.coadd))
          self.bfrac[self.dflux==0]=1
          self.bfrac=self.add_shared_array(len(filenames),self.bfrac,p)
      if cattype=='i3epoch':
        if 'ccd' in cols:
          self.ccd-=1        
      if ('g' in cols)&('r' in cols):
        self.gr=self.add_shared_array(len(filenames),self.g-self.r,p)
      if ('r' in cols)&('i' in cols):
        self.ri=self.add_shared_array(len(filenames),self.r-self.i,p)
      if ('i' in cols)&('z' in cols):
        self.iz=self.add_shared_array(len(filenames),self.i-self.z,p)

      #Make footprint contiguous across ra=0
      if ('ra' in cols):
        ra=self.ra
        ra[self.ra>180]=self.ra[self.ra>180]-360
        self.ra=ra

      #Read random file for galaxy catalog
      if (cattype=='gal')&(ranfile is not None):
        tmp=fio.FITS(ranfile)[-1].read()
        try:
          self.ran_ra=tmp['ra']
          self.ran_dec=tmp['dec']
          self.ran_zp=tmp['z']
        except ValueError:
          self.ran_ra=tmp['RA']
          self.ran_dec=tmp['DEC']
          self.ran_zp=tmp['Z']
        ra=self.ran_ra
        ra[self.ran_ra>180]=self.ran_ra[self.ran_ra>180]-360
        self.ran_ra=ra
        # self.ran_ra=self.ran_ra[ra>60]
        # self.ran_dec=self.ran_dec[ra>60]

      #Build jackknife regions
      if jkbuild:
        X=np.vstack((self.ra,self.dec)).T
        km0 = km.kmeans_sample(X, config.cfg.get('num_reg',100), maxiter=100, tol=1.0e-5)
        self.regs=km0.labels
        #Number of jackknife regions
        self.num_reg=config.cfg.get('num_reg',100)
      else:
        self.num_reg=0
        self.regs=np.ones(len(self.coadd))


      # if jkload:
      #   self.num_reg,self.regs=jackknife_methods.load_jk_regs(self.ra,self.dec,64,jkdir)
      # else:
      #   self.num_reg=0
      #   self.regs=np.ones(len(self.coadd))


    # Default information assigned to a catalog object
    self.name=name
    self.cat=cattype
    self.release=release
    #Number of bins in linear split functions (see e.g., sys_split module).
    self.lbins=config.cfg.get('lbins',10)
    #Number of bins to split signal in for systematics null tests.
    self.sbins=config.cfg.get('sbins',2)
    #Binslop for use in Treecorr.
    self.slop=config.cfg.get('slop',1.)
    #Number of separation bins in Treecorr.
    self.tbins=config.cfg.get('tbins',10)
    #Number of bandpower bins 
    self.cbins=config.cfg.get('cbins',500)
    #Separation [min,max] for Treecorr.
    self.sep=config.cfg.get('sep',[1.,400.])
    self.calc_err=False
    #Number of simulation patches for covariances.
    self.num_patch=config.cfg.get('num_patch',100)
    #Whether to use weighting and bias/sensitivity corrections in calculations.
    if cattype=='gal':
      self.bs=False
      self.wt=config.cfg.get('wt',False)
    else:
      self.bs=config.cfg.get('bs',False)
      self.wt=config.cfg.get('wt',False)
    #Whether to reweight n(z) when comparing subsets of data for null tests.
    self.pzrw=config.cfg.get('pzrw',False) 

    return

  def add_shared_array(self,length,array0,p):

    if p is not None:
      if array0.dtype.kind=='i':
        shared_array_base=multiprocessing.RawArray(ctypes.c_ulong, length)
        array=np.frombuffer(shared_array_base,dtype=int)
      else:
        shared_array_base=multiprocessing.RawArray(ctypes.c_double, length)
        array=np.frombuffer(shared_array_base)
      array[:]=array0
    else:
      array=array0
    return array

class PZStore(object):
  """
  A flexible class that reads and stores photo-z catalog information. An example is given in the main testsuite.py.

  Use:

  :name:      str - A label for identifying the catalog in output. Multiple catalog classes can be defined, but each should have a unique name.
  :setup:     bool - Read in data and setup class structure? If false, only basic structure is created. Used for custom catalogs to feed into functions that expect a PZStore object.
  :pztype:    str - This identifies which photo-z code is stored in the class. In SV was used for dealing with multiple formats. Hopefully deprecated in Y1 analysis. For now identifies h5 table name.
  :filetype:  bool - Type of file to be read. dict - standard dict file for passing n(z) for spec validation, h5 - h5 file of pdfs, fits - fits file of pdfs. Non-dict file support not fully migrated from SV code yet - can use setup=False to store pdf information in class manually, though.
  :file:      str - File to be read in.
  """

  # This is messy currently due to randomness in catalog formats for photo-z information...

  def __init__(self,name,setup=True,pztype='skynet',filetype=None,file=None):

    if setup:

      if file is None:
        raise CatValError('Need a source file.')

      if filetype=='dict':
        d=CatalogMethods.load_spec_test_file(file)
        self.tomo=len(d['phot'])
        self.boot=len(d['phot'][0])-1
        self.bins=len(d['phot'][0][0])
        self.bin=d['binning']
        self.binlow=d['binning']-(d['binning'][1]-d['binning'][0])/2.
        self.binhigh=d['binning']+(d['binning'][1]-d['binning'][0])/2.
        self.spec=np.zeros((self.tomo,self.bins))
        self.pz=np.zeros((self.tomo,self.bins))
        self.bootspec=np.zeros((self.tomo,self.boot,self.bins))
        self.bootpz=np.zeros((self.tomo,self.boot,self.bins))
        for i in xrange(self.tomo):
          self.spec[i,:]=d['spec'][i][0]
          self.pz[i,:]=d['phot'][i][0]
          for j in xrange(self.boot):
            self.bootspec[i,j,:]=d['spec'][i][j+1]
            self.bootpz[i,j,:]=d['phot'][i][j+1]
      elif filetype=='h5':
        self.pdftype='full'
        store=pd.HDFStore(file, mode='r')
        if hasattr(store,pztype):
          pz0=store[pztype]
        elif hasattr(store,'pdf'):
          pz0=store['pdf']
          print 'First h5 is pdf',pz0.columns
          if hasattr(store,'point_predictions'):
            pz1=store['point_predictions']
            print 'Second h5 is point_predictions',pz1.columns
          elif hasattr(store,'point_pred'):
            pz1=store['point_pred']
            print 'Second h5 is point_pred',pz1.columns
          else:
            print 'No second h5'
            pz1=pz0
        else:
          print pztype+' does not exist in dataframe'
          return

        self.coadd=CatalogMethods.find_col_h5('coadd_objects_id',pz0,pz1)
        zm0=CatalogMethods.find_col_h5('z_mean',pz0,pz1)
        if zm0 is not None:
          self.z_mean_full=zm0.astype('float32')
        elif file==config.pzdir+'BPZ_v1_probs_DEC15_trainvalid.hdf5':
          tmp=fio.FITS(config.pzdir+'DEC15_trainvalid_magauto_BPZv1_point.fits')[-1].read()
          self.z_mean_full=tmp['MEAN_Z'].astype('float32')
        else:
          zm0=CatalogMethods.find_col_h5('mean_z',pz0,pz1)
          self.z_mean_full=zm0.astype('float32')
        print 'mean',np.min(zm0),np.max(zm0),np.min(self.z_mean_full),np.max(self.z_mean_full)
        zm0=CatalogMethods.find_col_h5('z_peak',pz0,pz1)
        if zm0 is not None:
          self.z_peak_full=zm0.astype('float32')
        elif file==config.pzdir+'BPZ_v1_probs_DEC15_trainvalid.hdf5':
          tmp=fio.FITS(config.pzdir+'DEC15_trainvalid_magauto_BPZv1_point.fits')[-1].read()
          self.z_peak_full=tmp['Z_PEAK'].astype('float32')
        else:
          zm0=CatalogMethods.find_col_h5('mode_z',pz0,pz1)
          self.z_peak_full=zm0.astype('float32')
        print 'peak',np.min(zm0),np.max(zm0),np.min(self.z_peak_full),np.max(self.z_peak_full)
        zm0=CatalogMethods.find_col_h5('z_mc',pz0,pz1)
        if zm0 is not None:
          self.z_mc_full=zm0.astype('float32')
        elif file==config.pzdir+'BPZ_v1_probs_DEC15_trainvalid.hdf5':
          tmp=fio.FITS(config.pzdir+'DEC15_trainvalid_magauto_BPZv1_point.fits')[-1].read()
          self.z_mc_full=tmp['Z_MC'].astype('float32')
        else:
          zm0=CatalogMethods.find_col_h5('hwe_z',pz0,pz1)
          self.z_mc_full=zm0.astype('float32')
        print 'peak',np.min(zm0),np.max(zm0),np.min(self.z_mc_full),np.max(self.z_mc_full)
        self.spec_full=CatalogMethods.find_col_h5('z_spec',pz0,pz1)
        if file==config.pzdir+'BPZ_v1_probs_DEC15_trainvalid.hdf5':
          tmp=fio.FITS(config.pzdir+'DEC15_trainvalid_magauto_BPZv1_point.fits')[-1].read()
          self.spec_full=tmp['Z_SPEC'].astype('float32')
        self.bins=config.pz_binning.get(pztype,[0,0,0])[2]
        self.bin=np.linspace(config.pz_binning.get(pztype,[0,0,0])[0],config.pz_binning.get(pztype,[0,0,0])[1],config.pz_binning.get(pztype,[0,0,0])[2])
        self.binlow=self.bin-(self.bin[1]-self.bin[0])/2.
        self.binhigh=self.bin+(self.bin[1]-self.bin[0])/2.
        pz=pz0[np.sort([col for col in pz0.columns if 'pdf' in col])].values
        if len(pz)==0:
          pz=CatalogMethods.find_col_h5('z_mc',pz0,pz1)
        self.pz_full=pz.astype('float32')

        tmp=np.load(config.pzdir+'spec_jk_regs.npy')
        if len(self.coadd)==len(tmp):
          print 'using jk regions'
          m1,s1,m2,s2=CatalogMethods.sort(self.coadd,tmp[:,0])
          tmp=tmp[m2][s2]
          self.regs=tmp[:,1]
          self.num_reg=len(np.unique(self.regs))
        if hasattr(store,'pdf'):
          tmp=np.load(config.pzdir+'gold_weights_v1.npy')
          print 'loading weights'
          m1,s1,m2,s2=CatalogMethods.sort(self.coadd,tmp[:,0])
          tmp=tmp[m2][s2]
          self.w=tmp[:,1]
          self.coadd=self.coadd[m1]
          self.z_mean_full=self.z_mean_full[m1]
          self.z_peak_full=self.z_peak_full[m1]
          self.z_mc_full=self.z_mc_full[m1]
          self.spec_full=self.spec_full[m1]
          self.pz_full=self.pz_full[m1]
        else:
          self.w=np.ones(len(self.z_mean_full))
        self.wt=False

        store.close()
      elif filetype=='fits':
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

      self.pztype=pztype
      self.name=name

      #self.pz_full
      #self.z_mean_full
      #self.z_median_full
      #self.coadd

    else:

      self.pztype=pztype
      self.name=name

    return

class MockCatStore(object):
  """
  A flexible class that reads and stores mock catalog information. Used for covarianace calculations. Mock catalog module not yet migrated from SV code.
  """

  coadd=[]
  ra=[]
  dec=[]
  z=[]
  A00=[]
  A01=[]
  A10=[]
  A11=[]
  w=[]
  e1=[]
  e2=[]
  g1=[]
  g2=[]
  lbins=10
  cbins=5
  tbins=9
  sep=np.array([1,400])
  slop=0.15
  use_jk=False

  def __init__(self,setup=True,filenum=0,mocktype='sva1_gold'):

    if setup:
      for ifile,file in enumerate(glob.glob(config.mockdir+mocktype+'/*.fit')):
        if ifile==filenum:
          print 'simfile',ifile,file
          try:
            fits=fio.FITS(file)
            #tmparray=apt.Table.read(file)
          except IOError:
            print 'error loading fits file: ',file
            return
          tmparray=fits[-1].read()
          self.coadd=np.arange(len(tmparray['id']))
          self.ra=tmparray['ra']
          self.dec=tmparray['dec']
          self.z=tmparray['z']
          self.A00=tmparray['A00']
          self.A01=tmparray['A01']
          self.A10=tmparray['A10']
          self.A11=tmparray['A11']
          self.w=[]
          self.e1=[]
          self.e2=[]
          self.prop=[]
          self.g1=tmparray['g1']
          self.g2=tmparray['g2']
          break

class ColError(Exception):
  def __init__(self, value):
    self.value = value
  def __str__(self):
    return repr(self.value)


class CatValError(Exception):
  def __init__(self, value):
    self.value = value
  def __str__(self):
    return repr(self.value)


class CatalogMethods(object):

  @staticmethod
  def get_cat_cols(dir,cols,table,cuts,tiles=None,maxiter=999999,exiter=-1,hdu=-1,maxrows=1):
    """
    Work function for CatalogStore to parse and read in catalog informaiton from one or more fits files.
    """

    import fitsio as fio
    import re

    lenst=0
    # Loop over file(s) [in directory]
    for ifile,file in enumerate(glob.glob(dir)):
      if ifile>maxiter:
        break
      if (exiter>=0)&(exiter!=ifile):
        break
      # Skip any tiles not in tile list - needs to be generalised to work for other keywords/patterns
      if hasattr(tiles, '__len__'):
        m=re.search('.*/(DES\d\d\d\d[+-]\d\d\d\d).*',file)
        if m:
          if m.group(1) not in tiles:
            continue

      # File format may not be readable
      try:
        fits=fio.FITS(file)
      except IOError:
        print 'error loading fits file: ',file

      # Not reading in full file
      if cols is None:
        cols=fits[hdu].get_colnames()
        if lenst==0:
          cutcols=cuts['col']
      else:
        if lenst==0:
          cutcols=[table.get(x,x) for x in cuts['col']]

      # Verify that the columns requested exist in the file
      colex,colist=CatalogMethods.col_exists(cols,fits[hdu].get_colnames())
      if colex<1:
        for i,x in enumerate(cols):
          cols[i]=x.lower()
        colex,colist=CatalogMethods.col_exists(cols,fits[hdu].get_colnames())
        if colex<1:
          raise ColError('columns '+colist+' do not exist in file: '+file)

      colex,colist=CatalogMethods.col_exists(cutcols,fits[hdu].get_colnames())
      if colex<1:
        cutcols=[table.get(x,None).lower() for x in cuts['col']]
        colex,colist=CatalogMethods.col_exists(cutcols,fits[hdu].get_colnames())
        if colex<1:
          raise ColError('cut columns '+colist+' do not exist in file: '+file)

      # Dump the columns needed for masking into memory if everything is there
      try:
        tmparray=fits[hdu].read(columns=cutcols)
      except IOError:
        print 'error loading fits file: ',file

      # Generate the selection mask based on the passed cut function
      mask=np.array([])
      for icut,cut in enumerate(cuts): 
        mask=CatalogMethods.cuts_on_col(mask,tmparray,cutcols[icut],cut['min'],cut['eq'],cut['max'])

      # Dump the requested columns into memory if everything is there
      try:
        tmparray=fits[hdu].read(columns=cols)
      except IOError:
        print 'error loading fits file: ',file

      # If first file, generate storage array to speed up reading in of data
      if lenst==0:
        array=np.empty((maxrows), dtype=tmparray.dtype.descr)
        filenames=np.empty((maxrows), dtype='S'+str(len(file.split('/')[-1].split('.')[0])))
        filenums = np.empty((maxrows), dtype=int)

      # If exceeded max number of rows reqested, end iteration and return catalog
      if lenst+np.sum(mask)>maxrows:
        fits.close()
        return [array[col][:lenst] for i,col in enumerate(cols)],filenames[:lenst]
        
      array[lenst:lenst+np.sum(mask)]=tmparray[mask]
      filenames[lenst:lenst+np.sum(mask)]=np.repeat(file.split('/')[-1].split('.')[0],np.sum(mask))
      filenums[lenst:lenst+np.sum(mask)]=np.ones(np.sum(mask))*ifile

      lenst+=np.sum(mask)
      print ifile,np.sum(mask),lenst,file

      fits.close()

    return cols,[array[col][:lenst] for i,col in enumerate(cols)],filenames[:lenst],filenums[:lenst]


  @staticmethod
  def col_exists(cols,colnames):
    """
    Check whether columns exist and return list of missing columns for get_cat_cols().
    """

    colist=''
    exists=np.in1d(cols,colnames)
    for i,val in enumerate(exists):
      if not val:
        colist+=' '+cols[i]

    return np.sum(exists)/len(cols),colist

  @staticmethod
  def cuts_on_col(mask,array,col,valmin,valeq,valmax):
    """
    Build mask for cutting catalogs as columns are read in to get_cat_cols().
    """    

    if mask.size==0:
      mask=np.ones((len(array[col])), dtype=bool)

    if (valmin==noval) & (valmax==noval):
      if valeq==noval:
        print 'warning, no range or equality set in cut on column '+col
      else:
        mask=mask & (array[col]==valeq)
    elif (valmin!=noval) & (valmax!=noval):
      if valeq!=noval:
        print 'cannot have both equality and range cut on column '+col
      else:
        mask=mask & (valmin<array[col]) & (array[col]<valmax)
    elif (valmin!=noval):
      mask=mask & (valmin<array[col])
    else:
      mask=mask & (array[col]<valmax)
    return mask

  @staticmethod
  def col_view(array, cols):

    dtype2 = np.dtype({name:array.dtype.fields[name] for name in cols})
    return np.ndarray(array.shape, dtype2, array, 0, array.strides)

  @staticmethod
  def add_cut(cuts,col,min,eq,max):
    """
    Helper function to format catalog cuts.
    """    
    
    if cuts.size==0:
      cuts=np.zeros((1), dtype=[('col',np.str,20),('min',np.float64),('eq',np.float64),('max',np.float64)])
      cuts[0]['col']=col
      cuts[0]['min']=min
      cuts[0]['eq']=eq
      cuts[0]['max']=max
    else:
      cuts0=np.zeros((1), dtype=[('col',np.str,20),('min',np.float64),('eq',np.float64),('max',np.float64)])
      cuts0[0]['col']=col
      cuts0[0]['min']=min
      cuts0[0]['eq']=eq
      cuts0[0]['max']=max
      cuts=np.append(cuts,cuts0,axis=0)

    return cuts

  @staticmethod
  def load_spec_test_file(file):
    """
    Reads pickled photo-z dict.
    """    

    f=open(file, 'r')
    d=pickle.load(f)
    f.close()

    return d

  @staticmethod
  def sort(a1,a2):
    """
    Sorts and matches two arrays of unique object ids (in DES this is coadd_objects_id). This function is too slow for DES Y1+ or beyond size catalogs. See sort2().

    len(a1[mask1])==len(a2[mask2])
    (a1[mask1])[sort1]==a2[mask2]
    a1[mask1]==(a2[mask2])[sort2]

    """

    mask1=np.in1d(a1,a2,assume_unique=False)
    mask2=np.in1d(a2,a1,assume_unique=False)
    print len(mask1),len(a1),len(mask2),len(a2)
    sort1=np.argsort(a1[mask1])[np.argsort(np.argsort(a2[mask2]))]
    sort2=np.argsort(a2[mask2])[np.argsort(np.argsort(a1[mask1]))]

    return mask1,sort1,mask2,sort2


  @staticmethod
  def sort2(x,y):
    """
    Sorts and matches two arrays of unique object ids (in DES this is coadd_objects_id).
    """
    
    xsort = np.argsort(x)
    ysort = np.argsort(y)
    i_xy  = np.intersect1d(x, y, assume_unique=True)
    i_x   = xsort[x[xsort].searchsorted(i_xy)]
    i_y   = ysort[y[ysort].searchsorted(i_xy)]
    
    return i_x, i_y

  @staticmethod
  def sort2n(x,y):
    """
    Sorts and matches two arrays of object ids where x is unique and y is not (in DES this is coadd_objects_id).
    Slower than sort2().
    """
    
    xsort = np.argsort(x)
    ysort = np.argsort(y)
    i_yx  = np.sort(y[np.in1d(y,x,assume_unique=False)])
    i_x   = xsort[x[xsort].searchsorted(i_yx)]    
    i_y   = ysort[y[ysort].searchsorted(i_yx)]
    
    return i_x, i_y

  @staticmethod
  def get_new_nbcw(cat,file,w=True,prune=False):
    """
    Convenience function to match and add new nbc values from a fits file to a CatalogeStore object. Takes a CatalogStore object to modify.
    """

    fits=fio.FITS(file)
    tmp=fits[-1].read()

    m1,s1,m2,s2=CatalogMethods.sort(cat.coadd,tmp['coadd_objects_id'])
    if prune:
      CatalogMethods.match_cat(cat,m1)
    elif np.sum(m1)<len(cat.coadd):
      print 'missing ids in file'
      return

    tmp=tmp[m2]
    tmp=tmp[s2]

    cat.c1=tmp['c1']
    cat.c2=tmp['c2']
    cat.m=tmp['m']
    if w:
      cat.w=tmp['w']

    return

  @staticmethod
  def final_null_cuts():
    """
    Masking functions for use in CatalogStore initialisation. 

    Use:

    Each entry of CatalogMethods.add_cut(array,col,a,b,c) adds to array a structured definition of the mask to apply for a given column in the catalog, col. a,b,c are limiting values. If be is set, value in column must be equal to b. Otherwise it must be greater than a and/or less than c.
    """

    cuts=CatalogMethods.add_cut(np.array([]),'coadd',0,noval,noval)

    return cuts

  @staticmethod
  def final_null_cuts_id():
    """
    Masking functions for use in CatalogStore initialisation. 

    Use:

    Each entry of CatalogMethods.add_cut(array,col,a,b,c) adds to array a structured definition of the mask to apply for a given column in the catalog, col. a,b,c are limiting values. If be is set, value in column must be equal to b. Otherwise it must be greater than a and/or less than c.
    """

    cuts=CatalogMethods.add_cut(np.array([]),'id',0,noval,noval)

    return cuts

  @staticmethod
  def final_null_cuts_ra():
    """
    Masking functions for use in CatalogStore initialisation. 

    Use:

    Each entry of CatalogMethods.add_cut(array,col,a,b,c) adds to array a structured definition of the mask to apply for a given column in the catalog, col. a,b,c are limiting values. If be is set, value in column must be equal to b. Otherwise it must be greater than a and/or less than c.
    """

    cuts=CatalogMethods.add_cut(np.array([]),'ra',-99999,noval,noval)

    return cuts

  @staticmethod
  def final_null_cuts_ra_flag():
    """
    Masking functions for use in CatalogStore initialisation. 

    Use:

    Each entry of CatalogMethods.add_cut(array,col,a,b,c) adds to array a structured definition of the mask to apply for a given column in the catalog, col. a,b,c are limiting values. If be is set, value in column must be equal to b. Otherwise it must be greater than a and/or less than c.
    """

    cuts=CatalogMethods.add_cut(np.array([]),'ra',-99999,noval,noval)
    cuts=CatalogMethods.add_cut(np.array([]),'flag',noval,noval,1)

    return cuts

  @staticmethod
  def i3_cuts():
    """
    Masking functions for use in CatalogStore initialisation. 

    Use:

    Each entry of CatalogMethods.add_cut(array,col,a,b,c) adds to array a structured definition of the mask to apply for a given column in the catalog, col. a,b,c are limiting values. If be is set, value in column must be equal to b. Otherwise it must be greater than a and/or less than c.
    """

    cuts=CatalogMethods.add_cut(np.array([]),'info',noval,0,noval)
    cuts=CatalogMethods.add_cut(cuts,'psf1',-99.,noval,noval)
    cuts=CatalogMethods.add_cut(cuts,'psf2',-99.,noval,noval)
    cuts=CatalogMethods.add_cut(cuts,'psffwhm',-99.,noval,noval)
    cuts=CatalogMethods.add_cut(cuts,'rgp',1.13,noval,3.)
    cuts=CatalogMethods.add_cut(cuts,'snr',12.,noval,200.)

    return cuts

  @staticmethod
  def redmagic():
    """
    Masking functions for use in CatalogStore initialisation. 

    Use:

    Each entry of CatalogMethods.add_cut(array,col,a,b,c) adds to array a structured definition of the mask to apply for a given column in the catalog, col. a,b,c are limiting values. If be is set, value in column must be equal to b. Otherwise it must be greater than a and/or less than c.
    """

    cuts=CatalogMethods.add_cut(np.array([]),'coadd',0,noval,noval)

    return cuts

  @staticmethod
  def get_cat_colnames(cat):
    """
    Takes a CatalogStore object. Finds all array names of numeric arrays of length the array of unique object ids - cat.coadd. 
    """
    cols=[]
    for x in dir(cat):
      if x=='coadd':
        continue
      obj = getattr(cat,x)
      if isinstance(obj,np.ndarray):
        if obj.dtype.type is not np.string_:
          if len(obj)==len(cat.coadd):
            cols.append(x)

    return cols

  @staticmethod
  def match_cat(cat,mask):
    """
    Takes a CatalogStore object to modify. Masks all arrays of length the array of unique object ids - cat.coadd. Useful for selecting parts of catalog to speed up other function calls over the arrays and for matching two catalogs sorted with sort2(). 
    """

    for x in dir(cat):
      if x=='coadd':
        continue
      obj = getattr(cat,x)
      if isinstance(obj,np.ndarray):
        if len(obj)==len(cat.coadd):
          setattr(cat,x,obj[mask])

    cat.coadd=cat.coadd[mask]

    return

  @staticmethod
  def check_mask(array,mask,p=None):
    """
    Convenience function to return true array for mask if mask=None is supplied in other functions.
    """

    if mask is None:
      return np.ones(len(array)).astype(bool)
    else:
      return mask


  @staticmethod
  def find_col_h5(col,h5a,h5b):
    """
    Necessary to parse multiple photo-z file formats.
    """

    if col in h5a.columns:
      print 'using '+col+' from first h5'
      return h5a[col].values
    if col.upper() in h5a.columns:
      print 'using '+col+' from first h5'
      return h5a[col.upper()].values
    if col in h5b.columns:
      print 'using '+col+' from second h5'
      return h5b[col].values
    if col.upper() in h5b.columns:
      print 'using '+col+' from second h5'
      return h5b[col.upper()].values

    return None


  @staticmethod
  def info_flag(cat):
    """
    Takes properly constructed im3shape CatalogStore object and generates info_flag values. The definitions here are deprecated.
    """

    import healpy as hp
    gdmask=hp.read_map(config.golddir+'y1a1_gold_1.0.2_wide_footprint_4096.fit.gz')
    badmask=hp.read_map(config.golddir+'y1a1_gold_1.0.2_wide_badmask_4096.fit.gz')

    pix=hp.ang2pix(4096, np.pi/2.-np.radians(i3.dec),np.radians(i3.ra), nest=False)
    i3.gold_mask=(gdmask[pix] >=1)
    i3.gold_flag=badmask[pix]

    info_cuts =[
        'i3.gold_mask==False',
        'i3.gold_flag>0',
        'i3.modest!=1',
        'i3.maskfrac>.75', #0.75
        'i3.evals>10000',
        'i3.flagr==1',
        'i3.flagr==2',
        'i3.fluxfrac<.75', #.7
        'i3.snr<10.', #10
        'i3.snr>10000.', #10000
        'i3.rgp<1.1', #1.1
        'i3.rgp>3.5', #3.5
        'i3.rad>5', #10
        'i3.rad<.1', 
        'np.sqrt(i3.ra_off**2.+i3.dec_off**2.)>1', #1
        'i3.chi2pix<.5', #.8
        'i3.chi2pix>1.5', #2
        'i3.resmin<-0.2',#-2
        'i3.resmax>0.2',#2
        'i3.psffwhm>7',
        'i3.psffwhm<0',
        'i3.error!=0'
        # model image?
    ]

    i3.info = np.zeros(len(i3.coadd), dtype=np.int64)
    for i,cut in enumerate(info_cuts):
      mask=eval(cut).astype(int)
      print i,cut,np.sum(mask)
      j=1<<i
      flags=mask*j
      i3.info|=flags


    return

  @staticmethod
  def nbc_sv(cat):
    """
    Constructs nbc values for im3shape based on SV greatDES simulation results. Takes CatalogStore object.
    """

    def basis_m(snr,rgp):
      snr1=snr/100.
      rgp1=(rgp-1.)/10.

      func=np.zeros((len(snr1),18))
      func[:,0]=1/snr1**2*1/rgp1**2
      func[:,1]=1/snr1**3*1/rgp1**3
      func[:,2]=1/snr1**3*1/rgp1**2
      func[:,3]=1/snr1**2*1/rgp1**3
      func[:,4]=1/snr1**4*1/rgp1**3
      func[:,5]=1/snr1**4*1/rgp1**4
      func[:,6]=1/snr1**3*1/rgp1**4
      func[:,7]=1/snr1**2.5*1/rgp1**2.5
      func[:,8]=1/snr1**2.5*1/rgp1**3
      func[:,9]=1/snr1**3*1/rgp1**2.5
      func[:,10]=1/snr1**1.5*1/rgp1**1.5
      func[:,11]=1/snr1**2.5*1/rgp1**1.5
      func[:,12]=1/snr1**1.5*1/rgp1**2.5
      func[:,13]=1/snr1**1.5*1/rgp1**2
      func[:,14]=1/snr1**2*1/rgp1**1.5
      func[:,15]=1/snr1**1.25*1/rgp1**1.75
      func[:,16]=1/snr1**1.75*1/rgp1**1.25
      func[:,17]=1/snr1**4*1/rgp1**4
      return func

    def basis_a(snr,rgp):
      snr1=snr/100.
      rgp1=(rgp-1.)/10.

      func=np.zeros((len(snr1),18))
      func[:,0] =1/snr1**2*1/rgp1**2
      func[:,1] =1/snr1**3*1/rgp1**3
      func[:,2] =1/snr1**3*1/rgp1**2
      func[:,3] =1/snr1**2*1/rgp1**3
      func[:,4] =1/snr1**4*1/rgp1**3
      func[:,5] =1/snr1**4*1/rgp1**4
      func[:,6] =1/snr1**3*1/rgp1**4
      func[:,7] =1/snr1**2.5*1/rgp1**2.5
      func[:,8] =1/snr1**2.5*1/rgp1**3
      func[:,9] =1/snr1**3* 1/rgp1**2.5
      func[:,10]=1/snr1**1.5*1/rgp1**1.5
      func[:,11]=1/snr1**2.5*1/rgp1**1.5
      func[:,12]=1/snr1**1.5*1/rgp1**2.5
      func[:,13]=1/snr1**3*1/rgp1**5
      func[:,14]=1/snr1**5*1/rgp1**3
      func[:,15]=1/snr1**5*1/rgp1**5
      func[:,16]=1/snr1**5*1/rgp1**4
      func[:,17]=1/snr1**4*1/rgp1**5
      return func

    wm=np.array([-1.05e-03,1.47e-06,8.10e-05,6.73e-06,0.00e+00,0.00e+00,0.00e+00,6.56e-05,-6.16e-06,-2.09e-05,-7.63e-03,-1.37e-03,-1.08e-04,1.63e-03,5.37e-03,1.63e-04,2.28e-03,-2.73e-11])
    wa=np.array([-1.67283612e-04, 1.09715332e-06, 5.95801408e-05, 6.39015150e-07, 2.97121531e-08, -3.60228146e-10, 4.73608639e-09, 4.05443791e-05, -3.52379986e-06, -1.95627195e-05, 8.97549111e-04, -3.23420375e-04, -1.91942923e-06, -6.57971727e-12, -1.41012000e-09, 1.61504257e-15, 2.36381064e-11, -1.76498862e-12])

    b=basis_m(cat.snr,cat.rgp)
    cat.m = np.dot(b, wm)
    b=basis_a(cat.snr,cat.rgp)
    a = np.dot(b, wa)
    cat.c1=a*cat.psf1
    cat.c2=a*cat.psf2

    return

  @staticmethod
  def download_cat_desdm(query,name='gold',table='table',dir='/share/des/sv/',order=True,num=1000000,start=0):

    from astropy.table import Table
    from desdb import Connection

    conn = Connection()

    if order:
      sorder='order by coadd_objects_id'
    else:
      sorder=''

    for tile in range(140):
      if tile<start:
        continue
      print tile
      if num==0:
        q = 'select '+query+' from '+table
      else:
        q = 'select '+query+' from ( select /*+ FIRST_ROWS(n) */ A.*, ROWNUM rnum from ( select * from '+table+' '+sorder+') A where ROWNUM < '+str((tile+1.)*num)+' ) where rnum  >= '+str(tile*num)
      print q
      data=conn.quick(q, array=True)
      params = data[0].keys()
      tables = {}
      for p in params:
        arr = [(d[p] if (d[p] is not None) else np.nan) for d in data ]
        arr = np.array(arr)
        tables[p] = arr

      t = Table(tables)
      t.write(dir+name+'_'+str(tile)+'.fits.gz')
      if num==0:
        break

    return

  @staticmethod
  def select_random_pts(nran,hpmap,rannside=262144,masknside=4096):
    """
    This function does the work on each processor for create_random_cat(). Options are passed from that function.
    """

    import healpy as hp
    import numpy.random as rand

    ranmap=hp.nside2npix(rannside)
    print ranmap
    hpmin=np.min(hpmap)
    hpmax=np.max(hpmap)

    tmp0=hp.nside2npix(rannside)//hp.nside2npix(masknside)

    ran=[]
    while len(ran)<nran:
      print nran,len(ran)

      tmp=rand.randint(hpmin*tmp0,high=hpmax*tmp0,size=nran)
      mask=np.in1d(tmp//tmp0,hpmap,assume_unique=False)
      ran=np.append(ran,tmp[mask])

    ran=ran[:nran]
    dec,ra=hp.pix2ang(rannside,ran.astype(int),nest=True)
    dec=90.-dec*180./np.pi
    ra=ra*180./np.pi

    return ra,dec,ran

  @staticmethod
  def create_random_cat(nran,maskpix,label='',rannside=262144,masknside=4096):
    """
    This will create a uniform (currently, will update as needed) random catalog from a mask defined via healpixels (maskpix). Input maskpix should be in nest form. label prepends a label to the output file. masknside is the nside of the mask, rannside is the pixelisation of the random distribution - defaults to about 2/3 arcsecond area pixels.

    This will produce a fits file with 'ra','dec' columns that contains nran*100*MPI.size() randoms.
    """

    from mpi4py import MPI
    import time

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    t1=time.time()
    for i in xrange(1):

      ra,dec,ran=CatalogMethods.select_random_pts(nran,maskpix,rannside=rannside,masknside=masknside)
      print 'after',i,rank,time.time()-t1

      x=np.empty((len(ra)*size))
      y=np.empty((len(dec)*size))

      comm.Allgather([ra, MPI.DOUBLE],[x, MPI.DOUBLE])
      comm.Allgather([dec, MPI.DOUBLE],[y, MPI.DOUBLE])

      if rank == 0:
        print 'end',i,rank
        if i!=0:
          x0=np.load(label+'ra.npy')
          y0=np.load(label+'dec.npy')
          x=np.hstack((x,x0))
          y=np.hstack((y,y0))
        np.save(label+'ra.npy',x)
        np.save(label+'dec.npy',y)

    if rank == 0:
      CatalogMethods.create_random_cat_finalise(label=label)

    return ran

  @staticmethod
  def create_random_cat_finalise(label=''):
    """
    This function removes duplicate randoms from the results of create_random_cat() and writes a fits file with the random catalog.
    """

    def unique(a):
      order = np.lexsort(a.T)
      a = a[order]
      diff = np.diff(a, axis=0)
      ui = np.ones(len(a), 'bool')
      ui[1:] = (diff != 0).any(axis=1) 
      return a[ui],ui

    a=np.vstack((np.load(label+'ra.npy'),np.load(label+'dec.npy'))).T
    u,i=unique(a)
    a=a[i]

    ran=np.empty(len(a), dtype=[('coadd_objects_id','f8')]+[('ra','f8')]+[('dec','f8')])
    ran['coadd_objects_id']=np.arange(len(a))
    ran['ra']=a[:,0].T
    ran['dec']=a[:,1].T

    os.remove(label+'ra.npy')
    os.remove(label+'dec.npy')

    fio.write(label+'random.fits.gz',ran,clobber=True)
    
    return

  @staticmethod
  def save_cat(cat):

    for x in dir(cat):
      obj = getattr(cat,x)
      if isinstance(obj,np.ndarray):
        if len(obj)==len(cat.coadd):
          fio.write(x+'.fits.gz',obj,clobber=True)

    return

  @staticmethod
  def load_cat(cat):

    for ifile,file in enumerate(glob.glob('/home/troxel/destest/*fits.gz')):
      fits=fio.FITS(file)
      setattr(cat,file[21:-8],fits[-1].read())

    return

  @staticmethod
  def footprint_area(cat,ngal=1,mask=None,nside=4096,nest=True,label=''):
    """
    Calculates footprint area of catalog.
    """    
    import healpy as hp
    import matplotlib
    matplotlib.use ('agg')
    import matplotlib.pyplot as plt
    plt.style.use('/home/troxel/SVA1/SVA1StyleSheet.mplstyle')
    from matplotlib.colors import LogNorm
    import pylab

    mask=CatalogMethods.check_mask(cat.coadd,mask)

    if not hasattr(cat, 'pix'):
      cat.pix=CatalogMethods.radec_to_hpix(cat.ra,cat.dec,nside=nside,nest=True)
    area=hp.nside2pixarea(nside)*(180./np.pi)**2
    print 'pixel area (arcmin)', area*60**2
    mask1=np.bincount(cat.pix[mask])>ngal
    print 'footprint area (degree)', np.sum(mask1)*area

    pix=np.arange(len(mask1))[mask1]
    print pix
    tmp=np.zeros((12*nside**2), dtype=[('hpix','int')])
    tmp['hpix'][pix.astype(int)]=1
    print tmp['hpix'][pix.astype(int)]
    fio.write('footprint_hpix'+label+'.fits.gz',tmp,clobber=True)

    tmp2=np.zeros(12*nside**2)
    tmp2[tmp.astype(int)]=1
    hp.cartview(tmp2,nest=True)
    plt.savefig('footprint_hpix'+label+'.png')
    plt.close()

    return 

  @staticmethod
  def radec_to_hpix(ra,dec,nside=4096,nest=True):
    """
    Returns healpix pixel array for input ra,dec.
    """  
    import healpy as hp

    return hp.ang2pix(nside, np.pi/2.-np.radians(dec),np.radians(ra), nest=nest)

  @staticmethod
  def remove_duplicates(cat):
    """
    Removes duplicate unique id entries in catalogstore object.
    """  
    a=np.argsort(cat.coadd)
    mask=np.diff(cat.coadd[a])
    mask=np.where(mask!=0)[0]
    mask=a[mask]
    CatalogMethods.match_cat(cat,mask)

    return

  @staticmethod
  def merge_red_shape(
    rmd=config.redmagicdirnersc+'y1a1_gold_1.0.2b-full_redmapper_v6.4.11_redmagic_highdens_0.5-10.fit',
    rml=config.redmagicdirnersc+'y1a1_gold_1.0.2b-full_redmapper_v6.4.11_redmagic_highlum_1.0-04.fit',
    rpm=config.redmapperdirnersc+'y1a1_gold_1.0.2b-full_run_redmapper_v6.4.11_lgt5_desformat_catalog_members.fit',
    spec0='/global/cscratch1/sd/troxel/spec_cat_0.fits.gz',
    shape='/home/troxel/y1a1-im3shape-r-1-1-1.fits',
    isdir=True):
    """
    Merges redma*, spec, and shape catalogs.
    """  

    def read_cat(file0,spec):

      print 'loading '+file0

      tmp=fio.FITS(file0)[-1].read()
      store=np.ones(tmp.shape, dtype=tmp.dtype.descr + [('e1','f8')]+[('e2','f8')]+[('m','f8')]+[('c1','f8')]+[('c2','f8')]+[('weight','f8')])
      for name in tmp.dtype.names:
        store[name]=tmp[name]

      store['e1']=-9999*store['e1']
      store['e2']=-9999*store['e2']
      store['c1']=-9999*store['c1']
      store['c2']=-9999*store['c2']
      store['m']=-9999*store['m']
      store['weight']=-9999*store['weight']

      try:
        x,y=CatalogMethods.sort2(store['COADD_OBJECTS_ID'],spec['coadd_objects_id'])
      except:
        x,y=CatalogMethods.sort2n(store['ID'],spec['coadd_objects_id']) 
      store['ZSPEC'][x]=spec['z_spec'][y]

      return store

    def store_shape(store,tmp2):

      try:
        x,y=CatalogMethods.sort2(store['COADD_OBJECTS_ID'],tmp2['coadd_objects_id'])
      except:
        x,y=CatalogMethods.sort2n(store['ID'],tmp2['coadd_objects_id']) 
      store['e1'][x]=tmp2['e1'][y]
      store['e2'][x]=tmp2['e2'][y]
      store['c1'][x]=tmp2['c1'][y]
      store['c2'][x]=tmp2['c2'][y]
      store['m'][x]=tmp2['m'][y]
      store['weight'][x]=tmp2['weight'][y]

      return

    spec=fio.FITS(spec0)[-1].read()

    store_rmd=read_cat(rmd,spec)
    store_rml=read_cat(rml,spec)
    store_rpm=read_cat(rpm,spec)

    if isdir:

      for ifile,file0 in enumerate(glob.glob(shape+'*')):
        print ifile,file0
        tmp2=fio.FITS(file0)[-1].read(columns=['coadd_objects_id','e1','e2','mean_psf_e1_sky','mean_psf_e2_sky','mean_psf_fwhm','mean_rgpp_rp','snr','m','c1','c2','weight','info_flag'])
        mask=(tmp2['info_flag']==0)&(tmp2['mean_rgpp_rp']>1.13)&(tmp2['snr']>12)&(tmp2['snr']<200)&(tmp2['mean_rgpp_rp']<3)&(~(np.isnan(tmp2['mean_psf_e1_sky'])|np.isnan(tmp2['mean_psf_e2_sky'])|np.isnan(tmp2['snr'])|np.isnan(tmp2['mean_psf_fwhm'])))
        tmp2=tmp2[mask]

        store_shape(store_rmd,tmp2)
        store_shape(store_rml,tmp2)
        store_shape(store_rpm,tmp2)

    else:

      tmp2=fio.FITS(shape)[-1].read(columns=['coadd_objects_id','e1','e2','mean_psf_e1_sky','mean_psf_e2_sky','mean_psf_fwhm','mean_rgpp_rp','snr','m','c1','c2','weight','info_flag'])
      mask=(tmp2['info_flag']==0)&(tmp2['mean_rgpp_rp']>1.13)&(tmp2['snr']>12)&(tmp2['snr']<200)&(tmp2['mean_rgpp_rp']<3)&(~(np.isnan(tmp2['mean_psf_e1_sky'])|np.isnan(tmp2['mean_psf_e2_sky'])|np.isnan(tmp2['snr'])|np.isnan(tmp2['mean_psf_fwhm'])))
      tmp2=tmp2[mask]

      store_shape(store_rmd,tmp2)
      store_shape(store_rml,tmp2)
      store_shape(store_rpm,tmp2)

    fio.write(config.redmagicdirnersc+'y1a1_gold_1.0.2b-full_redmapper_v6.4.11_redmagic_highdens_0.5-10_e.fit',store_rmd,clobber=True)
    fio.write(config.redmagicdirnersc+'y1a1_gold_1.0.2b-full_redmapper_v6.4.11_redmagic_highlum_1.0-04_e.fit',store_rml,clobber=True)
    fio.write(config.redmapperdirnersc+'y1a1_gold_1.0.2b-full_run_redmapper_v6.4.11_lgt5_desformat_catalog_members_e.fit',store_rpm,clobber=True)

    return
