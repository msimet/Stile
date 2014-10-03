import numpy
import tempfile
import os
import helper
import unittest
try:
    import stile
except:
    import sys
    sys.path.append('..')
    import stile

class TestFileIO(unittest.TestCase):
    def setUp(self):
        # Test data tables.
        self.table1 = numpy.array(
            [(5.389e-02, 5.443e-02, 2.206e-02, -4.259e-02, 2.578e-02, 1.820e+02, 1.820e+02),
             (6.260e-02, 6.205e-02, 3.738e-03, 2.995e-02, 2.079e-02, 2.800e+02, 2.800e+02),
             (7.271e-02, 7.241e-02, 1.857e-02, -4.192e-02, 1.755e-02, 3.930e+02, 3.930e+02), 
             (8.446e-02, 8.489e-02, -9.208e-03, -6.122e-03, 1.556e-02, 5.000e+02, 5.000e+02), 
             (9.811e-02, 9.817e-02, 1.986e-02, 9.554e-03, 1.447e-02, 5.780e+02, 5.780e+02), 
             (1.140e-01, 1.140e-01, 7.698e-03, 4.370e-03, 1.068e-02, 1.061e+03, 1.061e+03), 
             (1.324e-01, 1.342e-01, 9.733e-03, -4.264e-03, 1.037e-02, 1.125e+03, 1.125e+03), 
             (1.538e-01, 1.520e-01, 2.400e-03, -7.149e-03, 9.048e-03, 1.478e+03, 1.478e+03), 
             (1.786e-01, 1.772e-01, 2.964e-03, -1.999e-03, 7.230e-03, 2.315e+03, 2.315e+03), 
             (2.075e-01, 2.076e-01, 1.652e-03, -2.616e-03, 6.580e-03, 2.795e+03, 2.795e+03), 
             (2.410e-01, 2.411e-01, -1.277e-02, 2.982e-04, 5.274e-03, 4.350e+03, 4.350e+03), 
             (2.799e-01, 2.855e-01, 1.326e-03, -1.140e-03, 4.651e-03, 5.593e+03, 5.593e+03), 
             (3.252e-01, 3.270e-01, -2.517e-03, 3.061e-04, 4.554e-03, 5.834e+03, 5.834e+03), 
             (3.777e-01, 3.752e-01, -3.427e-03, -5.551e-03, 3.459e-03, 1.011e+04, 1.011e+04), 
             (4.387e-01, 4.393e-01, 2.000e-04, -2.414e-03, 3.095e-03, 1.263e+04, 1.263e+04), 
             (5.096e-01, 4.973e-01, -3.784e-03, 5.500e-04, 3.025e-03, 1.322e+04, 1.322e+04), 
             (5.920e-01, 5.873e-01, 2.131e-03, 9.000e-05, 3.673e-03, 8.971e+03, 8.971e+03), 
             (6.877e-01, 6.877e-01, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00), 
             (7.988e-01, 7.988e-01, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00),
             (9.278e-01, 9.278e-01, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00, 0.000e+00)],
             dtype='d,d,d,d,d,d,d')

        self.table2_withstring = numpy.array([
            (1, 'first', 'aa', 1.0),
            (1, 'second', 'bb', 1.0),
            (2, 'third', 'cc', 2.0),
            (3, 'fourth', 'dd', 3.0),
            (5, 'fifth', 'ee', 5.0),
            (8, 'sixth', 'ff', 8.0),
            (13, 'seventh', 'gg', 13.0),
            (21, 'eighth', 'hh', 21.0),
            (34, 'ninth', 'ii', 34.0),
            (55, 'tenth', 'jj', 55.0)],dtype='l,S7,S2,d')

        # contents of test_data/[table.fits,image_and_table.fits,two_tables.fits (hdu 2)]
        self.fits_table = numpy.array([(1.5,'hello',2),(3,'goodbye',5)],
                                      dtype=[('q',float),('status','S7'),('final',int)])
        # contents of first extension of test_data/two_tables.fits                         
        self.fits_table_2 = numpy.array([(1.5,2),(3.7,4),(1050.2,5)],
                                        dtype=[('red',float),('blue',int)])
        # contents of test_data/[image_int.fits,image_and_table.fits]
        self.fits_int_image = numpy.array([[1,2],[3,4]])
        # contents of test_data/image_float.fits
        self.fits_float_image = numpy.array([[1.0,2.1],[3.4,1.6]])

    def test_ReadFITSImage(self):
        """Test the ability to read in a FITS image."""
        if stile.file_io.has_fits:
            # Test that it read in the contents correctly, basically
            numpy.testing.assert_array_equal(stile.ReadFITSImage('test_data/image_int.fits'),
                                             self.fits_int_image)
            numpy.testing.assert_array_equal(stile.ReadFITSImage('test_data/image_and_table.fits'),
                                             self.fits_int_image)
            numpy.testing.assert_array_equal(stile.ReadFITSImage('test_data/image_float.fits'),
                                             self.fits_float_image)
            self.assertRaises(IOError,stile.ReadFITSImage,'test_data/data_table.dat')
                                            
    def test_ReadFITSTable(self):
        """Test the ability to read in a FITS table."""
        if stile.file_io.has_fits:
            # First, test that it's reading in the files correctly.  (The helper.FormatSame 
            # thing is to deal with the fact that assert_equal [and assert_array_equal] don't think 
            # that numpy.ndarray and FITSrecs should be compared to each other, and also to deal 
            # with FITS vs Python byteorder things for machines where that matters.  
            result = stile.ReadFITSTable('test_data/table.fits')
            numpy.testing.assert_equal(*helper.FormatSame(result,self.fits_table))
            result = stile.ReadTable('test_data/table.fits')
            numpy.testing.assert_equal(*helper.FormatSame(result,self.fits_table))
            result = stile.ReadFITSTable('test_data/two_tables.fits')
            numpy.testing.assert_equal(*helper.FormatSame(result,self.fits_table_2))
            result = stile.ReadFITSTable('test_data/image_and_table.fits')
            numpy.testing.assert_equal(*helper.FormatSame(result,self.fits_table))
            result = stile.ReadFITSTable('test_data/two_tables.fits',hdu=2)
            numpy.testing.assert_equal(*helper.FormatSame(result,self.fits_table))
            # And make sure that ReadTable picks up that this is a FITS file
            result = stile.ReadTable('test_data/two_tables.fits',hdu=2)
            numpy.testing.assert_equal(*helper.FormatSame(result,self.fits_table))
            self.assertRaises(IOError,stile.ReadFITSImage,'test_data/data_table.dat')

    def test_ReadASCIITable(self):
        """Test the ability to read in an ASCII table."""
        # ReadASCIITable is a wrapper for numpy.genfromtxt() that turns things into formatted
        # arrays if necessary, so we don't really need to test much of the functionality--just make 
        # sure that both natively formatted arrays (table_with_string) and natively raw arrays 
        # (treecorr_output) are both returned as formatted arrays.
        results = stile.ReadASCIITable('test_data/TreeCorr_output.dat',comments='#')
        numpy.testing.assert_equal(results,self.table1)
        results = stile.ReadASCIITable('test_data/table_with_string.dat')
        numpy.testing.assert_equal(results,self.table2_withstring)
        results = stile.ReadTable('test_data/table_with_string.dat')
        numpy.testing.assert_equal(results,self.table2_withstring)
        
    def test_WriteASCIITable(self):
        """Test the ability to write an ASCII table."""
        # Must be done after test_read_ASCII_table() since it uses the read_ASCII_table function!
        handle, filename = tempfile.mkstemp()
        stile.file_io.WriteASCIITable(filename,self.table1)
        results = stile.ReadASCIITable(filename)
        # genfromtxt will cleverly figure out the last two columns are ints, which we want it to, 
        # but that causes problems here.
        try:
            numpy.testing.assert_equal(results,self.table1)
            raise RuntimeError('Arrays are equal, but should not have the same format')
        except AssertionError:
            pass
        numpy.testing.assert_equal(results.astype('f'),self.table1.astype('f')) 

        # check field reordering
        field_list = ['f3','f4','f6','f0','f2','f1','f5']
        stile.file_io.WriteASCIITable(filename,self.table1,fields=field_list)
        results = stile.ReadASCIITable(filename)
        numpy.testing.assert_equal(results.astype('f'),self.table1[field_list].astype('f'))
        os.close(handle)
        if not stile.file_io.has_fits:  
            # If there is a FITS handler, this table would be written as a FITS table, so we
            # shouldn't check it.  But otherwise these will be ASCII files, so test that 
            # WriteTable() is sending them through that function properly.
            handle, filename = tempfile.mkstemp()
            stile.file_io.WriteTable(filename,self.table1)
            results = stile.ReadASCIITable(filename)
            numpy.testing.assert_equal(self.table1.astype('f'),results.astype('f')) 
            os.close(handle)

    def test_WriteFITSTable(self):
        """Test the ability to write a FITS table."""
        if stile.file_io.has_fits:
            # First check that WriteFITSTable writes a FITS table of the right sort...
            handle, filename = tempfile.mkstemp()
            stile.WriteFITSTable(filename,self.fits_table)
            try:
                self.assertTrue(
                    stile.file_io.fits_handler.FITSDiff('test_data/table.fits','temp.fits'))
            except AttributeError: 
                # FITSDiff seems to not exist for one of my pyfits installations
                numpy.testing.assert_equal(stile.ReadFITSTable('test_data/table.fits'),
                                           stile.ReadFITSTable(filename))
            os.close(handle)
            
            # Now see if WriteTable properly sends things through WriteFITSTable
            handle, filename = tempfile.mkstemp()
            stile.WriteTable(filename,self.fits_table_2)
            numpy.testing.assert_equal(stile.ReadFITSTable('test_data/two_tables.fits'),
                                       stile.ReadFITSTable(filename))
            
            os.close(handle)
            handle, filename = tempfile.mkstemp()
            self.assertRaises(TypeError,stile.WriteFITSTable,filename,[])
            os.close(handle)
          
if __name__=='__main__':
    if not stile.file_io.has_fits:
        print "Skipping FITS tests (no FITS module found)"
    unittest.main()

