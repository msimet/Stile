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

class HSCDataHandler():
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
