"""
data_handler.py: defines the classes that serve data to the various Stile systematics tests in the
default drivers.
"""
import os
import glob

class DataHandler:
    """
    A class which contains information about the data set Stile is to be run on.  This is used for
    the default drivers, not necessarily the pipeline-specific drivers (such as HSC/LSST).

    The class needs to be able to do two things:
      - List some data given a set of requirements (DataHandler.listData()).  The requirements
        generally follow the form:
         -- object_types: a list of strings such as "star", "PSF star", "galaxy" or "galaxy random"
              describing the objects that are needed for the tests.
         -- epoch: whether this is a single/summary catalog, or a multiepoch time series. (Coadded
              catalogs with no per-epoch information count as a single/summary catalog!)
         -- extent: "CCD", "field", "patch" or "tract".  This can be ignored if you don't mind
              running some extra inappropriate tests!  "CCD" should be a CCD-type dataset, "field" a
              single pointing/field-of-view, "patch" an intermediate-size area, and "tract" a large
              area.  (These terms have specific meanings in the LSST pipeline, but are used for
              convenience here.)
         -- data_format: "image" or "catalog."  Right now no image-level tests are implemented but
              we request this kwarg for future development.
      - Take each element of the data list from DataHandler.listData() and retrieve it for use
        (DataHandler.getData()), optionally with bins defined.  (Bins can also be defined on a test-
        by-test basis, depending on which format makes the most sense for your data setup.)

      Additionally, the class can define a .getOutputPath() function to place the data in a more
      complex system than the default (all in one directory with long output path names).
      """
    def __init__(self):
        raise NotImplementedError()

    def listData(self, object_types, epoch, extent, data_format, required_fields=None):
        raise NotImplementedError()

    def getData(self, ident, object_types=None, epoch=None, extent=None, data_format=None,
                bin_list=None):
        """
        Return some data matching the `ident` for the given kwargs.  This can be a numpy array, a
        tuple (file_name, field_schema) for a file already existing on the filesystem, or a list of
        either of those things.

        If it's a tuple (file_name, field_schema), the assumption is that it can be read by a simple
        FITS or ASCII reader.  The format will be determined from the file extension.
        """
        raise NotImplementedError()

    def getOutputPath(self, extension='.dat', multi_file=False, *args):
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
        #TODO: no-clobbering case
        sys_test_string = '_'.join(args)
        if multi_file:
            nfiles = glob.glob(os.path.join(self.output_path, sys_test_string)+'*'+extension)
            return os.path.join(self.output_path, sys_test_string+'_'+str(nfiles)+extension)
        else:
            return os.path.join(self.output_path, sys_test_string+extension)
