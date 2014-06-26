"""
data_handler.py: defines the classes that serve data to the various Stile systematics tests.
"""
import os
import glob

class DataHandler:
    temp_dir = None # Can (should?) be overwritten by other DataHandlers
    def __init__(self):
        raise NotImplementedError()
    
    def bin(self,data,bin_list):
        """
        Apply a series of SingleBin objects in a bin_list to `data`, an image or array.
        """
        for bin in bin_list:
            data = data[bin(data)]
        return data

    def listData(self,pair_or_single,epoch,extent,data_format,required_fields=None):
        raise NotImplementedError()
    
    def getData(self,ident,pair_or_single="single",bin_list=None,format=None,epoch=None,extent=None,
                force=False):
        """
        Return some data matching the given kwargs.  This can be a numpy array, a tuple
        (file_name, field_schema) for a file already existing on the filesystem, or a list of 
        either of those things (but NOT BOTH!).
        
        If getData ever returns a (list of) (file_name, field_schema) tuple(s), then calling
        dh.getData(file_name,force=True) should return the contents of that file, even if the 
        "ident"s are not normally file names.
        """ 
        raise NotImplementedError()

    def getOutputPath(self,extension='.dat',multi_file=False,*args):
        """
        Return a path to an output file given a list of strings that should appear in the output
        filename, taking care not to clobber other files (unless requested).
        @param args       A list of strings to appear in the file name 
        @param extension  The file extension to be used
        @param multi_file Whether multiple files with the same args will be created within a single
                          run of Stile. This appends a number to the file name; if clobbering is 
                          allowed, this argument also prevents Stile from writing over outputs from 
                          the same systematics test during the same run.
        @returns A path to an output file meeting the input specifications.
        """ 
        #TODO: clobbering
        sys_test = '_'.join(args)
        base_path = os.path.join(self.output_path,sys_test)
        if multi_file:
            nfiles = glob.glob(base_path+'*'+extension)
            return os.path.join(self.output_path,base_path+'-'+str(nfiles)+extension)
        else:
            return os.path.join(self.output_path,base_path+extension)

class CachedDataHandler(DataHandler):
    """
    A child class of the base DataHandler which implements caches for catalog and/or image data.
    Useful for any case where the data is read into Stile and binned. The number of items in the
    caches is controlled by max_image_cache (images) and max_catalog_cache (tables).  Single-
    process runs of the main Stile drivers need only max_image_cache and max_catalog_cache==1, the
    default, since the `ident` is the top_level loop.  If multiprocessing, max_image_cache and
    max_catalog_cache should be >= the number of processors used to avoid excessive file reads.
    Running CachedDataHandler.clear_cache(ident) at the end of the top-level loops will ensure
    that no data is read twice; otherwise, some data may be read twice, if separate processors are
    taking disparate amounts of time to run.
    
    Data is cached such that the least recently used data is discarded first, if data must be
    discarded to fit within max_cache.  Data should be cached before any binning takes place.  The
    method `initCache` should be called at the end of the __init__ block for child classes of
    CachedDataHandler.
    
    The data is stored in a dict whose keys are the data idents.  If saveCache() is called with an
    ident which is already in the cache, the data in the cache is overwritten.
    """
    def initCache(self,max_image_cache=1,max_catalog_cache=1)
        """
        Set up the image and catalog caches.
        """
        self._image_cache = {}
        self._catalog_cache = {}
        self._image_cache_recent = []
        self._catalog_cache_recent = []
        self._max_image_cache = max_image_cache
        self._max_catalog_cache = max_catalog_cache
    def saveCache(self,ident,data,data_format):
        """
        Save the `data` with the associated `ident` for use in the future in the cache
        corresponding to `data_format`.  Remove an item from the cache to make room, if
        necessary.  If the associated `ident` is already in the cache, the corresponding data is
        overwritten.
        """
        if data_format=='image':
            cache = self._image_cache
            max_cache = self._max_image_cache
            recent = self._image_cache_recent
        else:
            cache = self._catalog_cache
            max_cache = self._max_catalog_cache
            recent = self._catalog_cache_recent
        if max_cache>0:
            keys = cache.keys()
            if len(keys)>max_cache:
                raise RuntimeError('%s cache is too large'%data_format)
            if ident in cache:
                cache[ident] = data
                recent.delete(ident)
                recent.append(ident)
            else:
                if len(keys)==max_cache:
                    del cache[recent[0]]
                    recent = recent[1:]
                cache[ident] = data
                recent.append(ident)

    def getCache(self,ident,data_format):
        """
        Retrieve the data corresponding to `ident` from the cache corresponding to `data_format`.
        """
        if data_format=='image':
            cache = self._image_cache
            recent = self._image_cache_recent
        else:
            cache = self._catalog_cache
            recent = self._catalog_cache_recent
        if ident not in recent:
            raise RuntimeError(('Ident %s is in the %s cache, but cannot be found in the list of'+
                                'recent items.')%(ident,data_format))
        recent.delete(ident)
        recent.append(ident)
        return cache[ident]
        
    def inCache(self,ident,data_format):
        """
        Returns True if `ident` is in the cache corresponding to `data_format`.
        """
        if data_format=="image":
            return (ident in self._image_cache)
        else:
            return (ident in self._catalog_cache)
            
    def clearCache(self,ident,data_format=None):
        """
        Remove the data corresponding to `ident` from one or both caches.  If `data_format`==None,
        both caches will be cleared; otherwise only the `data_format` cache will be cleared.
        """
        if data_format=='image' or not data_format:
            if ident in self._image_cache:
                del self._image_cache[ident]
                self._image_cache_recent.delete(ident)
        if data_format=='catalog' or not data_format:
            if ident in self._catalog_cache:
                del self._catalog_cache[ident]
                self._catalog_cache_recent.delete(ident)
        
class HSCDataHandler(CachedDataHandler):
    def __init__(self, hsc_args):
        try:
            import lsst.daf.persistence as dafPersist
        except:
            raise ImportError("Cannot import lsst.daf.persistence which is required for "+
                              "HSCDataHandler.")
        dataDir = hsc_args.dataDir
        self.butler = dafPersist.Butler(dataDir)
        self.output_path = '.'
        max_cache = hsc_args.get('max_cache',1)
        max_image_cache = hsc_args.get('max_image_cache',max_cache)
        max_catalog_cache = hsc_args.get('max_catalog_cache',max_cache)
        self.initCache(max_image_cache,max_catalog_cache)

    def _fixCovMomentsD(self,arr,name):
        pass
    def _fixFlag(self,arr,name):
        pass
    def _fixArrayD(self,arr,name):
        pass
    def _fixPointD(self,arr,name):
        pass
    def _fixCovPointD(self,arr,name):
        pass
    def _fixCoord(self,arr,name):
        pass
    def _fixPSF(self,arr,name):
        pass
    
    def _modifyArray(self,data,required_fields):
        posskeys_fix = [('coord',self._fixCoord)] # and others
        posskeys_ok = [('psf.flux','psf.flux')] # and others
        cols = {}
        dtypes = []
        for key,stile_key in posskeys_ok:
            if key in data.schema:
                cols[stile_key] = append(data.get(key))
                dtypes.append((stile_key,cols[-1].dtype))
        for key,func in posskeys_fix:
            if key in data.schema:
                objs = func([data_point.get(key) for data_point in data],key)
                for col, dtype in objs:
                    cols[dtype[0]] = col
                    dtypes.append(dtype)
        data = numpy.zeros(len(data),dtype=dtypes)
        for key in data.dtype.names:
            data[key] = cols[key]
        return data

    def _makePSFImage(self,psf):
        """
        Turn the PSF object into an image
        """
        raise NotImplementedError()

    def _matchReqs(self, ident, required_fields):
        """
        Make sure the catalog contains the required fields, and retrieve or measure them if not.
        """
        if not required_fields:
            return True
        else:
            raise NotImplementedError()

    def _getImageData(self,ident,bin_list=None,data_format='image',epoch=None,extent=None,
                           force=False,required_fields=None):
        """
        Get and process any image data from the butler.
        """
        if data_format!='image':
            raise ValueError('_getImageData requires "image" for the data_format')
        if self.inCache(ident,'image'):
            data = self.getCache(ident,'image')
        else:
            if epoch=='multi':
                epoch_key = 'deepCoadd_'
            else:
                epoch_key = ''
            if 'psf' in required_fields:
                data = self.butler.get(ident,epoch_key+'psf')
                data = self._makePSFImage(data)
            else:
                data = self.butler.get(ident,epoch_key+'calexp')
            self.saveCache(ident,data,'image')
        return self.bin(data,bin_list)

    def _getCatalogData(self,ident,bin_list=None,data_format='catalog',epoch=None,extent=None,
                             force=False,required_fields=None):
        """
        Get and process any catalog data from the butler.  Make sure it contains the right fields
        for required_fields, or retrieve or measure them from other places if not.
        """
        if data_format!='catalog':
            raise ValueError('_getImageData requires "catalog" for the data_format')
        if self.inCache(ident,'catalog'):
            data = self.getCache(ident,'catalog')
            if not self._matchReqs(ident,required_fields):
                data = self._updateReqs(data,ident,required_fields) # Retrieve, add necessary fields
                self.saveCache(ident,data,'catalog') # Save the extended data back to the cache
        else:
            if epoch=='multi':
                epoch_key = 'deepCoadd_'
            else:
                epoch_key = ''
            data = self.butler.get(ident,epoch_key+'src')
            if any(['psf' in rq for rq in required_fields and rq not in data]):
                data['psf'] = self.butler.get(ident,epoch_key+'psf')
            data = self._modifyArray(data,required_fields)
            self.saveCache(ident,data,'catalog')
        return self.bin(data,bin_list)

    def listData(self,pair_or_single,epoch,extent,data_format,required_fields=None):
        raise NotImplementedError()

    def getData(self,ident,pair_or_single=None,epoch=None,extent=None,data_format=None,
                random=False,bin_list=None,force=False,required_fields=None):
        if random:
            raise NotImplementedError()
        if force:
            return self.bin(ident,bin_list)
        elif format=='image':
            return self._getImageData(ident,bin_list,format,epoch,extent,force,required_fields)
        else:
            return self._getTableData(ident,bin_list,format,epoch,extent,force,required_fields)
