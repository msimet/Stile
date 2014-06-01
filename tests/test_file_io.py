import numpy
import time
try:
    import stile
except:
    import sys
    sys.path.append('..')
    import stile

# Test data tables.
table1 = numpy.array(
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

table2_withstring = numpy.array([
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

table3_singleline = numpy.array([(1.0,2.0,3,'hello')], dtype='d,d,l,S5')

def test_ReadFitsImage():
    if not stile.file_io.has_fits:
        print "No FITS handler found; skipping test of read_fits_image"

def test_ReadFitsTable():
    if not stile.file_io.has_fits:
        print "No FITS handler found; skipping test of read_fits_table"

def test_ReadAsciiTable():
    t0 = time.time()
    results = stile.ReadAsciiTable('test_data/corr2_output.dat',comment='#')
    numpy.testing.assert_equal(results,table1)
    results = stile.ReadAsciiTable('test_data/corr2_output.dat',start_line=3)
    numpy.testing.assert_equal(results,table1[1:]) # since first skipped line is a comment
    results = stile.ReadAsciiTable('test_data/table_with_string.dat')
    numpy.testing.assert_equal(results,table2_withstring)
    results = stile.ReadAsciiTable('test_data/table_with_string.dat',comment='s')
    numpy.testing.assert_equal(results,table2_withstring)
    numpy.testing.assert_raises(IndexError,stile.ReadAsciiTable,
                                'test_data/table_with_missing_field.dat')
    t1 = time.time()
    print "Time to test ASCII table read: ", 1000*(t1-t0), "ms"
    
def test_WriteAsciiTable():
    # Must be done after test_read_ascii_table() since it uses the read_ascii_table function!
    import tempfile
    import os
    t0 = time.time()
    handle, filename = tempfile.mkstemp()
    stile.file_io.WriteAsciiTable(filename,table1)
    results = stile.ReadAsciiTable(filename)
    numpy.testing.assert_equal(table1.astype('f'),results.astype('f')) 

    col_list = ['f3','f4','f6','f0','f2','f1','f5']
    stile.file_io.WriteAsciiTable(filename,table1,cols=col_list)
    results = stile.ReadAsciiTable(filename)
    numpy.testing.assert_equal(table1[col_list].astype('f'),results.astype('f'))
    os.close(handle)
    if os.path.isfile(filename):
        os.remove(filename)
    t1 = time.time()
    print "Time to test ASCII table write: ", 1000*(t1-t0), "ms"

if __name__=='__main__':
    test_ReadFitsImage()
    test_ReadFitsTable()
    test_ReadAsciiTable()
    test_WriteAsciiTable()

