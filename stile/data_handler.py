"""
data_handler.py: defines the classes that serve data to the various Stile tests.
"""
import os
import glob

class DataHandler:
    temp_dir = '.' # Can (should?) be overwritten by other DataHandlers
    def __init__(self):
        raise NotImplementedError()
    
    def listData(self,format=None,epoch=None,extent=None,fields=None):
        raise NotImplementedError()
    
    def getData(self,ident,bin_list=None,format=None,epoch=None,extent=None,force=False):
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
                          the same test during the same run.
        @returns A path to an output file meeting the input specifications.
        """ 
        #TODO: clobbering
        test = '_'.join(args)
        base_path = os.path.join(self.output_path,test)
        if multi_file:
            nfiles = glob.glob(base_path+'*'+extension)
            return os.path.join(self.output_path,base_path+'-'+str(nfiles)+extension)
        else:
            return os.path.join(self.output_path,base_path+extension)

