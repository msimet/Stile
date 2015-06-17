import numpy
import sys
import unittest

try:
    import stile
except ImportError:
    sys.path.append('..')
    import stile


e1 = [0.421276226426, 0.248889539829, 0.148089741551, 0.566468893815, 0.770334448403,
      -1.27622668103, -0.278332559516, -0.0251835420899, -1.62818069301, 0.702161390942,
      0.129529824212, -1.27776942697, -0.710875839398, -1.1335793859, -1.04080707461,
      -1.65086191504, -1.40401920141, 0.7444639257, 0.433889348913, -0.41658808983, 0.0595854281922,
      -0.954522564497, 0.289112274197, -0.0269082232735, 0.524005376038, -0.884279805103,
      0.386267187801, 0.459108410812, -0.837961549229, -2.94522510662, 0.217256993842,
      1.03461460385, 0.747697610909, -1.04474109671, 0.148850586487, -0.25280413009, 0.439367536488,
      -0.654960854765, 0.720404748973, -0.294096536953, 0.743294479812, -0.876907911916,
      1.7968329893, 0.525003247233, 0.843964480056, -0.0439629812406, 0.734188694026,
      0.107428156418, -0.708738286387, 2.10618204709, 0.0375719635484, 1.28394004218,
      -0.0148659488638, 0.335083593734, 0.270957283902, -0.0216356036923, 0.102495479916,
      0.0788080513263, -0.152250345965, -1.9232014627, 1.76281941927, -0.77827391566,
      -0.327698245345, -2.8349486293, 0.486518651676, -0.174461225907, 1.49712076004,
      -0.0192066839702, -0.440034592757, 0.531829410488, 1.00373231941, 1.00209273045, 1.1443289089,
      0.400742577523, 0.339739232257, -1.44558832832, -0.694514979012, -0.745566597959,
      1.75860234203, -0.0884143619186, 1.71907390731, 1.60589446927, 0.30850571315, -2.55379214159,
      0.99217104105, 1.09011037649, 0.0380480602541, -0.724767050203, -0.0229718168813,
      -0.123116588353, 1.06084090472, -3.06698994438, 0.753312286232, -1.27553750378,
      -1.19888696958, -0.505577200354, -1.53081183926, 0.721787458056, -1.64629351295,
      2.78859472966, -0.628463116903, 1.83803080657, -1.22681682534, 1.53392950221, -0.146953706144,
      -0.223805341429, -0.934543873994, -2.47187956559, -0.172048621467, 0.626254208767,
      -0.320556115122, -0.152165788827, 0.176846339142, 0.799374722463, -0.528247969287,
      0.0175850586217, 2.29336191268, -1.81488630602, -0.091685410815, 1.94971486187]
mag = [14.8435156914, 14.1908463802, 14.3260978164, 14.7713128658, 14.2375565438, 14.7216183965,
       14.2066642651, 14.6890539665, 14.3540117115, 14.4674828168, 14.2725141398, 14.6878726633,
       14.8357036783, 14.8605710431, 14.2105982155, 14.6843866996, 14.1853052891, 14.7347528304,
       14.260101013, 14.0080025531, 15.5034102914, 15.8914987723, 15.3574767202, 15.2144006756,
       15.3464926008, 15.8491548213, 15.2385574084, 15.32234957, 15.7373386709, 15.3210672383,
       15.3820830471, 15.2247734595, 15.7135322218, 15.5777172989, 15.8207178049, 15.3497158703,
       15.733125029, 15.0961534562, 15.2008993803, 15.3465561104, 16.736433564, 16.9498795809,
       16.5998325489, 16.5615294886, 16.3587625526, 16.4944837469, 16.1678565896, 16.8446540809,
       16.7225771659, 16.2774487571, 16.3307575884, 16.6506168378, 16.2848323207, 16.9597441601,
       16.0599585425, 16.2993441047, 16.5644310218, 16.0404524774, 16.1004008827, 16.105415745,
       17.9515814785, 17.4868439292, 17.7695008586, 17.5840244446, 17.6474927952, 17.1217961689,
       17.1482286314, 17.4023940335, 17.7749584264, 17.6197726047, 17.538029214, 17.1410107284,
       17.2664306017, 17.0677492338, 17.2480659229, 17.8482180569, 17.9289409015, 17.056659366,
       17.8604921386, 17.4703316731, 18.1843444943, 18.5056441016, 18.1254228416, 18.5605439377,
       18.9192087749, 18.9835669315, 18.3701788909, 18.0059356899, 18.5994306164, 18.7364872086,
       18.847013152, 18.2390753809, 18.2296005197, 18.334774316, 18.5773209381, 18.9908561801,
       18.0341188447, 18.0163669231, 18.4686584294, 18.0542893363, 19.4487328726, 19.0490591931,
       19.8500530986, 19.1794731161, 19.562552228, 19.2382319793, 19.2138138099, 19.9348404502,
       19.784382543, 19.2725371274, 19.5729389869, 19.7636863178, 19.3471996974, 19.6713559119,
       19.7096265534, 19.9638882085, 19.1217978937, 19.1296452118, 19.0901038016, 19.3641559185]      
data = numpy.rec.fromarrays([numpy.array(mag), numpy.array(e1)], names=['mag','e1'])

means = numpy.array([-0.33386605345, -0.143461455482, 0.261209398388,
                     0.118940939595, -0.119718822072, 0.0503444521599])
medians = numpy.array([-0.1517580508039, 0.10421800734, 0.104961818167,
                       0.160266274143, 0.00753812168641, -0.149559747485])
rms = numpy.array([0.898951027918, 0.890891207925, 0.911144568394,
                   1.11059740575, 1.51276947794, 1.16757299242])
counts = [20, 20, 20, 20, 18, 22]          
                   
def e1_median(array):
    return numpy.median(array['e1'])

class TestSysTests(unittest.TestCase):
    def test_BinnedScatterPlotSysTest(self):
        """
        Test that the values passed to scatterPlot from BinnedScatterPlotSysTest.__call__ make sense
        """
        test_obj_1 = stile.BinnedScatterPlotSysTest() # Blank, to test kwarg calls
        test_obj_2 = stile.BinnedScatterPlotSysTest(x_field='mag', y_field='e1', binning = 6)
        test_obj_3 = stile.BinnedScatterPlotSysTest(x_field='mag', y_field='e1',
            binning=stile.BinStep(field='mag', low=14.0080025531, high=19.9638941644, n_bins=6))
        test_obj_4 = stile.BinnedScatterPlotSysTest(x_field='mag', y_field='e1', binning = 6,
            method='median')
        test_obj_5 = stile.BinnedScatterPlotSysTest(x_field='mag', y_field='e1', binning = 6,
            method='rms')
        test_obj_6 = stile.BinnedScatterPlotSysTest(x_field='mag', y_field='e1', binning = 6,
            method='count')
        test_obj_7 = stile.BinnedScatterPlotSysTest(x_field='mag', y_field='e1', binning = 6,
            method=e1_median)
         
        test_obj_1(data, x_field='mag', y_field='e1', binning=6)
        mean_obj_1 = test_obj_1.getData()
        numpy.testing.assert_almost_equal(mean_obj_1['mean of e1'], means)
        test_obj_2(data)
        mean_obj_2 = test_obj_2.getData()
        numpy.testing.assert_almost_equal(mean_obj_2['mean of e1'], means)
        test_obj_3(data)
        mean_obj_3 = test_obj_3.getData()
        numpy.testing.assert_almost_equal(mean_obj_3['mean of e1'], means)
        test_obj_1(data, x_field='mag', y_field='e1', binning=6, method='median')
        median_obj_1 = test_obj_1.getData()
        numpy.testing.assert_almost_equal(median_obj_1['median of e1'], medians)
        test_obj_4(data)
        median_obj_4 = test_obj_4.getData()
        numpy.testing.assert_almost_equal(median_obj_4['median of e1'], medians)
        test_obj_1(data, x_field='mag', y_field='e1', binning=6, method='rms')
        rms_obj_1 = test_obj_1.getData()
        numpy.testing.assert_almost_equal(rms_obj_1['rms of e1'], rms)
        test_obj_5(data)
        rms_obj_5 = test_obj_5.getData()
        numpy.testing.assert_almost_equal(rms_obj_5['rms of e1'], rms)

        test_obj_1(data, x_field='mag', y_field='e1', binning=6, method='count')
        count_obj_1 = test_obj_1.getData()
        numpy.testing.assert_equal(count_obj_1['count of e1'], counts)
        test_obj_6(data)
        count_obj_6 = test_obj_6.getData()
        numpy.testing.assert_equal(count_obj_6['count of e1'], counts)


        test_obj_1(data, x_field='mag', y_field='e1', binning=6, method=e1_median)
        median_obj_1 = test_obj_1.getData()
        numpy.testing.assert_almost_equal(median_obj_1['f(data)'], medians)
        test_obj_7(data)
        median_obj_7 = test_obj_7.getData()
        numpy.testing.assert_almost_equal(median_obj_7['f(data)'], medians)

if __name__=='__main__':
    unittest.main()
        