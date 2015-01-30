try:
    import stile
except ImportError:
    import sys
    sys.path.append('..')
    import stile
from stile import stile_utils
import unittest
import copy

class Set(object):
    def __init__(self, config, expected_files, format, object, not_found_format, not_found_object):
        self.config = config
        self.expected_files = expected_files
        self.format = format
        self.object = object
        self.not_found_format = not_found_format
        self.not_found_object = not_found_object

class TestDataHandler(unittest.TestCase):
    def setUp(self):
        # An empty "files" keyword will make the DataHandler fail, so do this and don't bother checking the outputs till the end.
        self.testConfigDataHandler = stile.ConfigDataHandler({'files': {'extent': 'CCD', 'epoch': 'single', 'data_format': 'catalog',
                                                                        'object_type': 'galaxy', 'name': 'sg1.dat'}})
        # dict0: simple
        self.dict0 = {'single': {
            'CCD': {
                'catalog': {
                    'galaxy': ['g1.dat','g2.dat','g3.dat'],
                    'star':   ['s1.dat','s2.dat','s3.dat']
            } } } }
        self.expected_files0 = {stile_utils.Format(epoch='single',extent='CCD',data_format='catalog').str: {
                                'galaxy': [{'name': 'g1.dat', 'group': '_stile_group_0'},
                                           {'name': 'g2.dat', 'group': '_stile_group_1'},
                                           {'name': 'g3.dat', 'group': '_stile_group_2'}],
                                'star':   [{'name': 's1.dat', 'group': '_stile_group_0'},
                                           {'name': 's2.dat', 'group': '_stile_group_1'},
                                           {'name': 's3.dat', 'group': '_stile_group_2'}]}}
        self.expected_groups0 = {'_stile_group_0': {
                'single-CCD-catalog': { 'star': 0, 'galaxy': 0}}, 
            '_stile_group_1': {
                'single-CCD-catalog': { 'star': 1, 'galaxy': 1}},
            '_stile_group_2': {
                'single-CCD-catalog': { 'star': 2, 'galaxy': 2}}}

        # dict1: the file lists are also dicts
        self.dict1 = {'single': {
            'CCD': {
                'catalog': {
                    'galaxy': [{'name': 'g1.dat','file_reader': 'ASCII'},
                               {'name': 'g2.dat','file_reader': 'ASCII'},
                               {'name': 'g3.dat','file_reader': 'ASCII'}],
                    'star':   [{'name': 's1.dat','file_reader': 'ASCII'},
                               {'name': 's2.dat','file_reader': 'ASCII'},
                               {'name': 's3.dat','file_reader': 'ASCII'}]
            } } } } 
        self.expected_files1 = {stile_utils.Format(epoch='single',extent='CCD',data_format='catalog').str: {
                                'galaxy': [{'name': 'g1.dat', 'file_reader': 'ASCII', 'group': '_stile_group_0'},
                                           {'name': 'g2.dat', 'file_reader': 'ASCII', 'group': '_stile_group_1'},
                                           {'name': 'g3.dat', 'file_reader': 'ASCII', 'group': '_stile_group_2'}],
                                'star':   [{'name': 's1.dat', 'file_reader': 'ASCII', 'group': '_stile_group_0'},
                                           {'name': 's2.dat', 'file_reader': 'ASCII', 'group': '_stile_group_1'},
                                           {'name': 's3.dat', 'file_reader': 'ASCII', 'group': '_stile_group_2'}]}}
        self.expected_groups1 = self.expected_groups0
        
        # dict2: the file lists are mixed
        self.dict2 = {'single': {
            'CCD': {
                'catalog': {
                    'galaxy': [{'name': 'g1.dat','file_reader': 'ASCII'},'g2.dat','g3.dat'],
                    'star':   ['s1.dat',{'name': 's2.dat','file_reader': 'ASCII'},
                               {'name': 's3.dat','file_reader': 'ASCII'}]
            } } } }
        self.expected_files2 = {stile_utils.Format(epoch='single',extent='CCD',data_format='catalog').str: {
                                'galaxy': [{'name': 'g1.dat', 'file_reader': 'ASCII', 'group': '_stile_group_0'},
                                           {'name': 'g2.dat', 'group': '_stile_group_1'},
                                           {'name': 'g3.dat', 'group': '_stile_group_2'}],
                                'star':   [{'name': 's1.dat', 'group': '_stile_group_0'},
                                           {'name': 's2.dat', 'file_reader': 'ASCII', 'group': '_stile_group_1'},
                                           {'name': 's3.dat', 'file_reader': 'ASCII', 'group': '_stile_group_2'}]}}
        self.expected_groups2 = self.expected_groups0
        
        # dict3: multiple levels
        self.dict3 = {'single': {
            'CCD': {
                'catalog': {
                    'galaxy': [{'name': 'g1.dat','file_reader': 'ASCII'},'g2.dat','g3.dat'],
                    'star':   ['s1.dat',{'name': 's2.dat','file_reader': 'ASCII'},
                               {'name': 's3.dat','file_reader': 'ASCII'}]
                    },
                'image': {
                    'galaxy': [{'name': 'g1.dat','file_reader': 'ASCII'},'g2.dat','g3.dat'],
                    'star':   ['s1.dat',{'name': 's2.dat','file_reader': 'ASCII'},
                               {'name': 's3.dat','file_reader': 'ASCII'}]
                    } },
            'field': {
                'catalog': {
                    'galaxy': [{'name': 'g1.dat','file_reader': 'ASCII'},'g2.dat','g3.dat'],
                    'star':   ['s1.dat',{'name': 's2.dat','file_reader': 'ASCII'},
                               {'name': 's3.dat','file_reader': 'ASCII'}]
            } } } }
        self.expected_files3 = {stile_utils.Format(epoch='single',extent='CCD',data_format='image').str: {
                                'galaxy': [{'name': 'g1.dat', 'file_reader': 'ASCII', 'group': '_stile_group_0'},
                                           {'name': 'g2.dat', 'group': '_stile_group_1'},
                                           {'name': 'g3.dat', 'group': '_stile_group_2'}],
                                'star':   [{'name': 's1.dat', 'group': '_stile_group_0'},
                                           {'name': 's2.dat', 'file_reader': 'ASCII', 'group': '_stile_group_1'},
                                           {'name': 's3.dat', 'file_reader': 'ASCII', 'group': '_stile_group_2'}]},
                            stile_utils.Format(epoch='single',extent='CCD',data_format='catalog').str: {
                                'galaxy': [{'name': 'g1.dat', 'file_reader': 'ASCII', 'group': '_stile_group_3'},
                                           {'name': 'g2.dat', 'group': '_stile_group_4'},
                                           {'name': 'g3.dat', 'group': '_stile_group_5'}],
                                'star':   [{'name': 's1.dat', 'group': '_stile_group_3'},
                                           {'name': 's2.dat', 'file_reader': 'ASCII', 'group': '_stile_group_4'},
                                           {'name': 's3.dat', 'file_reader': 'ASCII', 'group': '_stile_group_5'}]},
                            stile_utils.Format(epoch='single',extent='field',data_format='catalog').str: {
                                'galaxy': [{'name': 'g1.dat', 'file_reader': 'ASCII', 'group': '_stile_group_6'},
                                           {'name': 'g2.dat', 'group': '_stile_group_7'},
                                           {'name': 'g3.dat', 'group': '_stile_group_8'}],
                                'star':   [{'name': 's1.dat', 'group': '_stile_group_6'},
                                           {'name': 's2.dat', 'file_reader': 'ASCII', 'group': '_stile_group_7'},
                                           {'name': 's3.dat', 'file_reader': 'ASCII', 'group': '_stile_group_8'}]}}
        self.expected_groups3 = {'_stile_group_0': {
                'single-CCD-image': { 'star': 0, 'galaxy': 0}}, 
            '_stile_group_1': {
                'single-CCD-image': { 'star': 1, 'galaxy': 1}},
            '_stile_group_2': {
                'single-CCD-image': { 'star': 2, 'galaxy': 2}},
            '_stile_group_3': {
                'single-CCD-catalog': { 'star': 0, 'galaxy': 0}},
            '_stile_group_4': {
                'single-CCD-catalog': { 'star': 1, 'galaxy': 1}},
            '_stile_group_5': {
                'single-CCD-catalog': { 'star': 2, 'galaxy': 2}},
            '_stile_group_6': {
                'single-field-catalog': { 'star': 0, 'galaxy': 0}},
            '_stile_group_7': {
                'single-field-catalog': { 'star': 1, 'galaxy': 1}},
            '_stile_group_8': {
                'single-field-catalog': { 'star': 2, 'galaxy': 2}}}
        
            
        # dict4: multiple levels, different order
        self.dict4 = {'single': {
            'CCD': {
                'catalog': {
                    'galaxy': [{'name': 'g1.dat','file_reader': 'ASCII'},'g2.dat','g3.dat'],
                    'star':   ['s1.dat',{'name': 's2.dat','file_reader': 'ASCII'},
                               {'name': 's3.dat','file_reader': 'ASCII'}]
                },
                'image': {
                    'galaxy': [{'name': 'g1.dat','file_reader': 'ASCII'},'g2.dat','g3.dat'],
                    'star':   ['s1.dat',{'name': 's2.dat','file_reader': 'ASCII'},
                               {'name': 's3.dat','file_reader': 'ASCII'}]
                } },
            'field': {
                'galaxy': {
                    'catalog': [{'name': 'g1.dat','file_reader': 'ASCII'},'g2.dat','g3.dat'], },
                'star': {
                    'catalog': ['s1.dat',{'name': 's2.dat','file_reader': 'ASCII'},
                              {'name': 's3.dat','file_reader': 'ASCII'}]}
            } } }
        self.expected_files4 = self.expected_files3
        self.expected_groups4 = self.expected_groups3
        
        # dict5: multiple levels, add some non-format-related kwargs at different levels
        # [note: wildcard means ['field'] should be empty since the files don't exist!]
        self.dict5 = {'single': {
            'CCD': {
                'catalog': {
                    'galaxy': [{'name': 'g1.dat','file_reader': 'ASCII'},'g2.dat','g3.dat'],
                    'star':   ['s1.dat',{'name': 's2.dat','file_reader': 'ASCII'},
                               {'name': 's3.dat','file_reader': 'ASCII'}]
                    },
                'image': {
                    'group': False,
                    'galaxy': [{'name': 'g1.dat','file_reader': 'ASCII'},'g2.dat','g3.dat'],
                    'star':   ['s1.dat',{'name': 's2.dat','file_reader': 'ASCII'},
                               {'name': 's3.dat','file_reader': 'ASCII'}]
                    },
                },
            'field': {
                'wildcard': True,
                'catalog': {
                    'galaxy': [{'name': 'g1.dat','file_reader': 'ASCII'},'g2.dat','g3.dat'],
                    'star':   ['s1.dat',{'name': 's2.dat','file_reader': 'ASCII'},
                               {'name': 's3.dat','file_reader': 'ASCII'}]
                    },
                'image': {
                    'galaxy': [{'name': 'g1.dat','file_reader': 'ASCII'},'g2.dat','g3.dat'],
                    'star':   ['s1.dat',{'name': 's2.dat','file_reader': 'ASCII'},
                               {'name': 's3.dat','file_reader': 'ASCII'}]
                    }
                }
            } }
        # parseFilse removes Boolean groups that _parseFileHelper doesn't, so we need 2 "expected files" here
        self.expected_files_helper5 = {stile_utils.Format(epoch='single',extent='CCD',data_format='catalog').str: {
                                'galaxy': [{'name': 'g1.dat', 'file_reader': 'ASCII', 'group': '_stile_group_0'},
                                           {'name': 'g2.dat', 'group': '_stile_group_1'},
                                           {'name': 'g3.dat', 'group': '_stile_group_2'}],
                                'star':   [{'name': 's1.dat', 'group': '_stile_group_0'},
                                           {'name': 's2.dat', 'file_reader': 'ASCII', 'group': '_stile_group_1'},
                                           {'name': 's3.dat', 'file_reader': 'ASCII', 'group': '_stile_group_2'}]},
                            stile_utils.Format(epoch='single',extent='CCD',data_format='image').str: {
                                'galaxy': [{'name': 'g1.dat', 'file_reader': 'ASCII', 'group': False},
                                           {'name': 'g2.dat', 'group': False},
                                           {'name': 'g3.dat', 'group': False}],
                                'star':   [{'name': 's1.dat', 'group': False},
                                           {'name': 's2.dat', 'file_reader': 'ASCII', 'group': False},
                                           {'name': 's3.dat', 'file_reader': 'ASCII', 'group': False}]}}
        self.expected_files5 = {stile_utils.Format(epoch='single',extent='CCD',data_format='catalog').str: {
                                'galaxy': [{'name': 'g1.dat', 'file_reader': 'ASCII', 'group': '_stile_group_0'},
                                           {'name': 'g2.dat', 'group': '_stile_group_1'},
                                           {'name': 'g3.dat', 'group': '_stile_group_2'}],
                                'star':   [{'name': 's1.dat', 'group': '_stile_group_0'},
                                           {'name': 's2.dat', 'file_reader': 'ASCII', 'group': '_stile_group_1'},
                                           {'name': 's3.dat', 'file_reader': 'ASCII', 'group': '_stile_group_2'}]},
                            stile_utils.Format(epoch='single',extent='CCD',data_format='image').str: {
                                'galaxy': [{'name': 'g1.dat', 'file_reader': 'ASCII'},
                                           {'name': 'g2.dat'},
                                           {'name': 'g3.dat'}],
                                'star':   [{'name': 's1.dat'},
                                           {'name': 's2.dat', 'file_reader': 'ASCII'},
                                           {'name': 's3.dat', 'file_reader': 'ASCII'}]}}
        self.expected_groups5 = {'_stile_group_0': {
                'single-CCD-catalog': { 'star': 0, 'galaxy': 0}}, 
            '_stile_group_1': {
                'single-CCD-catalog': { 'star': 1, 'galaxy': 1}},
            '_stile_group_2': {
                'single-CCD-catalog': { 'star': 2, 'galaxy': 2}}}
        
        # dict6: disrupt automatic grouping
        self.dict6 = {'single': {
            'CCD': {
                'catalog': {
                    'galaxy': [{'name': 'g1.dat','file_reader': 'ASCII'},'g2.dat','g3.dat'],
                    'star':   ['s1.dat',{'name': 's2.dat','file_reader': 'ASCII'},
                               {'name': 's3.dat','file_reader': 'ASCII', 'group': False}]
            } } } }
        self.expected_file_helper6 = {stile_utils.Format(epoch='single',extent='CCD',data_format='catalog').str: {
                                'galaxy': [{'name': 'g1.dat', 'file_reader': 'ASCII'},
                                           {'name': 'g2.dat'},
                                           {'name': 'g3.dat'}],
                                'star':   [{'name': 's1.dat'},
                                           {'name': 's2.dat', 'file_reader': 'ASCII'},
                                           {'name': 's3.dat', 'file_reader': 'ASCII', 'group': False}]}}
        self.expected_files6 = {stile_utils.Format(epoch='single',extent='CCD',data_format='catalog').str: {
                                'galaxy': [{'name': 'g1.dat', 'file_reader': 'ASCII'},
                                           {'name': 'g2.dat'},
                                           {'name': 'g3.dat'}],
                                'star':   [{'name': 's1.dat'},
                                           {'name': 's2.dat', 'file_reader': 'ASCII'},
                                           {'name': 's3.dat', 'file_reader': 'ASCII'}]}}
        self.expected_groups6 = {}
        
        # dict7: force grouping
        self.dict7 = {'single': {
            'CCD': {
                'catalog': {
                    'galaxy': [{'name': 'g1.dat','group': 2},{'name': 'g2.dat','group': 1},{'name': 'g3.dat','group': 3}],
                    'star':   [{'name': 's1.dat','group': 3},{'name': 's2.dat','group': 2},{'name': 's3.dat','group': 1}]
            } } } }
        self.expected_files7 = {stile_utils.Format(epoch='single',extent='CCD',data_format='catalog').str: {
                                'galaxy': [{'name': 'g1.dat', 'group': 2},
                                           {'name': 'g2.dat', 'group': 1},
                                           {'name': 'g3.dat', 'group': 3}],
                                'star':   [{'name': 's1.dat', 'group': 3},
                                           {'name': 's2.dat', 'group': 2},
                                           {'name': 's3.dat', 'group': 1}]}}
        self.expected_groups7 = {1: {
                'single-CCD-catalog': { 'star': 2, 'galaxy': 1}}, 
            2: {
                'single-CCD-catalog': { 'star': 1, 'galaxy': 0}},
            3: {
                'single-CCD-catalog': { 'star': 0, 'galaxy': 2}}}

        # dict8: disrupt automatic grouping for one file, retain it for the rest
        self.dict8 = {'single': {
            'CCD': {
                'catalog': {
                    'galaxy': [{'name': 'g1.dat','file_reader': 'ASCII'},'g3.dat'],
                    'star':   ['s1.dat',{'name': 's2.dat','file_reader': 'ASCII', 'group': False},
                               {'name': 's3.dat','file_reader': 'ASCII'}]
            } } } }
        self.expected_file_helper8 = {stile_utils.Format(epoch='single',extent='CCD',data_format='catalog').str: {
                                'galaxy': [{'name': 'g1.dat', 'file_reader': 'ASCII', 'group': '_stile_group_0'},
                                           {'name': 'g3.dat', 'group': '_stile_group_1'}],
                                'star':   [{'name': 's1.dat', 'group': '_stile_group_0'},
                                           {'name': 's2.dat', 'file_reader': 'ASCII', 'group': False},
                                           {'name': 's3.dat', 'file_reader': 'ASCII', 'group': '_stile_group_1'}]}}
        self.expected_files8 = {stile_utils.Format(epoch='single',extent='CCD',data_format='catalog').str: {
                                'galaxy': [{'name': 'g1.dat', 'file_reader': 'ASCII', 'group': '_stile_group_0'},
                                           {'name': 'g3.dat', 'group': '_stile_group_1'}],
                                'star':   [{'name': 's1.dat', 'group': '_stile_group_0'},
                                           {'name': 's2.dat', 'file_reader': 'ASCII'},
                                           {'name': 's3.dat', 'file_reader': 'ASCII', 'group': '_stile_group_1'}]}}
        self.expected_groups8 = {'_stile_group_0': {
                'single-CCD-catalog': { 'star': 0, 'galaxy': 0}}, 
            '_stile_group_1': {
                'single-CCD-catalog': { 'star': 2, 'galaxy': 1}}}


        # dict9: the dreaded multiepoch
        self.dict9 = {'single': {
                'CCD': {
                    'catalog': {
                        'galaxy': [{'name': ['g1-0.dat','g1-1.dat'],'file_reader': 'ASCII'},['g2-0.dat','g2-1.dat'],['g3-0.dat','g3-1.dat']],
                    } } },
            'multiepoch': {
                'CCD': {
                    'catalog': {
                        'galaxy': [{'name': ['g1-0.dat','g1-1.dat'],'file_reader': 'ASCII'},['g2-0.dat','g2-1.dat'],['g3-0.dat','g3-1.dat']],
            } } } }
        self.expected_files9 = {stile_utils.Format(epoch='single',extent='CCD',data_format='catalog').str: {
                                'galaxy': [{'name': 'g1-0.dat', 'file_reader': 'ASCII'},
                                           {'name': 'g1-1.dat', 'file_reader': 'ASCII'},
                                           {'name': 'g2-0.dat'},
                                           {'name': 'g2-1.dat'},
                                           {'name': 'g3-0.dat'},
                                           {'name': 'g3-1.dat'}]},
                            stile_utils.Format(epoch='multiepoch',extent='CCD',data_format='catalog').str: {
                                'galaxy': [{'name': ['g1-0.dat','g1-1.dat'], 'file_reader': 'ASCII'},
                                           {'name': ['g2-0.dat','g2-1.dat']},
                                           {'name': ['g3-0.dat','g3-1.dat']}]
                           } }
        self.expected_groups9 = {}

        # list0: list-form
        self.list0 = [
                {'name': 'sg1.dat', 'epoch': 'single', 'extent': 'field', 'data_format': 'catalog', 'object_type': 'galaxy', 'flag_col': 'is_galaxy'},
                {'name': 'sg1.dat', 'epoch': 'single', 'extent': 'field', 'data_format': 'catalog', 'object_type': 'star', 'flag_col': 'is_star'}
            ]
        self.expected_files_list0 = {stile_utils.Format(epoch='single',extent='field',data_format='catalog').str: {
                                'galaxy': [{'name': 'sg1.dat', 'flag_col': 'is_galaxy'}],
                                'star':   [{'name': 'sg1.dat', 'flag_col': 'is_star'}]
                            }}
        self.expected_groups_list0 = {}

        # sys tests 0: list form, complete description
        self.sys_tests_0 = [{'epoch': 'single', 'extent': 'CCD', 'data_format': 'catalog', 'name': 'CorrelationFunctionSysTest', 'type': 'GalaxyShear'},
            {'epoch': 'single', 'extent': 'CCD', 'data_format': 'catalog', 'name': 'CorrelationFunctionSysTest', 'type': 'BrightStarShear'},
            {'epoch': 'single', 'extent': 'CCD', 'data_format': 'catalog', 'name': 'ScatterPlotSysTest', 'type': 'StarVsPSFG1'},
            {'epoch': 'single', 'extent': 'CCD', 'data_format': 'catalog', 'name': 'StatSysTest', 'field': 'g1', 'object': 'galaxy'}]
        # sys tests 1: list form, incomplete description
        self.sys_tests_1 = [{'epoch': 'single', 'data_format': 'catalog', 'name': 'CorrelationFunctionSysTest', 'type': 'GalaxyShear'},
            {'extent': 'CCD', 'data_format': 'catalog', 'name': 'CorrelationFunctionSysTest', 'type': 'BrightStarShear'},
            {'extent': 'CCD', 'data_format': 'catalog', 'name': 'CorrelationFunctionSysTest', 'type': 'StarXStarShear'},
            {'data_format': 'catalog', 'name': 'ScatterPlotSysTest', 'type': 'StarVsPSFG1'},
            {'data_format': 'image', 'name': 'StatSysTest', 'field': 'g1', 'object': 'galaxy'}] # This last one doesn't really make sense, but as a test
        # sys tests 2: nested form, complete description
        self.sys_tests_2 = {'single': {
            'CCD': {
                'catalog': [{'name': 'CorrelationFunctionSysTest', 'type': 'GalaxyShear'},
                            {'name': 'CorrelationFunctionSysTest', 'type': 'BrightStarShear'},
                            {'name': 'ScatterPlotSysTest', 'type': 'StarVsPSFG1'},
                            {'name': 'StatSysTest', 'field': 'g1', 'object': 'galaxy'}] } } }
        # sys tests 3: nested form, incomplete description.
        # Note that due to processing requirements we can't do these all as one dict
        self.sys_tests_3a = {
            'catalog': [{'name': 'ScatterPlotSysTest', 'type': 'StarVsPSFG1'}], 
            'image': [{'name': 'StatSysTest', 'field': 'g1', 'object': 'galaxy'}]}
        self.sys_tests_3b = {
            'CCD': {
                'catalog': [{'name': 'CorrelationFunctionSysTest', 'type': 'BrightStarShear'},
                            {'name': 'CorrelationFunctionSysTest', 'type': 'StarXStarShear'},]}}
        self.sys_tests_3c = {
            'single': [{'data_format': 'catalog', 'name': 'CorrelationFunctionSysTest', 'type': 'GalaxyShear'}]}
        # sys tests 4: nested form, complete description, more complex
        self.sys_tests_4 = {'single': {
            'CCD': {
                'catalog': [{'name': 'CorrelationFunctionSysTest', 'type': 'GalaxyShear'},
                            {'name': 'CorrelationFunctionSysTest', 'type': 'BrightStarShear'}],
                'image':   [{'name': 'ScatterPlotSysTest', 'type': 'StarVsPSFG1'},
                            {'name': 'StatSysTest', 'field': 'g1', 'object': 'galaxy'}] } } }
        # sys tests 5: nested form, with multiepoch since that processing is slightly different
        self.sys_tests_5 = {'multiepoch': {
            'CCD': {
                'catalog': [{'name': 'CorrelationFunctionSysTest', 'type': 'GalaxyShear'},
                            {'name': 'CorrelationFunctionSysTest', 'type': 'BrightStarShear'}],
                'image':   [{'name': 'ScatterPlotSysTest', 'type': 'StarVsPSFG1'},
                            {'name': 'StatSysTest', 'field': 'g1', 'object': 'galaxy'}] } } }
        
            
    def test_parseFileHelper(self):
        # First make sure the file list is built correctly
        results, n = self.testConfigDataHandler._parseFileHelper(copy.deepcopy(self.dict0))
        self.assertEqual(results,self.expected_files0)
        # Next, make sure the group list is built correctly
        results, groups = self.testConfigDataHandler.parseFiles({'file': copy.deepcopy(self.dict0)})
        self.assertEqual(results,self.expected_files0)
        self.assertEqual(groups,self.expected_groups0)
        # Also check: does file querying work?
        # This test may need to be made more flexible later--it relies in the internal order of the dict, which may be different between
        # different Python installations.
        results = self.testConfigDataHandler.queryFile('s1.dat')
        self.assertEqual(results,'1 - format: single-CCD-catalog, object type: star, group: _stile_group_0')
        # Dummy check: are these functions removing duplicates properly?  ('group' was turned into a list here...)
        expected_results = {stile_utils.Format(epoch='single',extent='CCD',data_format='catalog').str: {
                                'galaxy': [{'name': 'g1.dat', 'group': ['_stile_group_0']},
                                           {'name': 'g2.dat', 'group': ['_stile_group_1']},
                                           {'name': 'g3.dat', 'group': ['_stile_group_2']}],
                                'star':   [{'name': 's1.dat', 'group': ['_stile_group_0']},
                                           {'name': 's2.dat', 'group': ['_stile_group_1']},
                                           {'name': 's3.dat', 'group': ['_stile_group_2']}]}}
        results,groups = self.testConfigDataHandler.parseFiles({'file': copy.deepcopy(self.dict0), 'file2': copy.deepcopy(self.dict0)})
        self.assertEqual(results,expected_results)
        self.assertEqual(groups,self.expected_groups0)
        # Repeat (skipping the dummy check) for dict1,2,3 etc...
        results, n = self.testConfigDataHandler._parseFileHelper(copy.deepcopy(self.dict1))
        self.assertEqual(results,self.expected_files1)
        results, groups = self.testConfigDataHandler.parseFiles({'file0':copy.deepcopy(self.dict1)})
        self.assertEqual(results,self.expected_files1)
        self.assertEqual(groups,self.expected_groups1)
        results, n = self.testConfigDataHandler._parseFileHelper(copy.deepcopy(self.dict2))
        self.assertEqual(results,self.expected_files2)
        results, groups = self.testConfigDataHandler.parseFiles({'filelist':copy.deepcopy(self.dict2)})
        self.assertEqual(results,self.expected_files2)
        self.assertEqual(groups,self.expected_groups2)
        results, n = self.testConfigDataHandler._parseFileHelper(copy.deepcopy(self.dict3))
        self.assertEqual(results,self.expected_files3)
        results, groups = self.testConfigDataHandler.parseFiles({'file99':copy.deepcopy(self.dict3)})
        self.assertEqual(results,self.expected_files3)
        self.assertEqual(groups,self.expected_groups3)
        # try queryFile with multiple files
        results = self.testConfigDataHandler.queryFile('s1.dat')
        self.assertEqual(results,'1 - format: single-CCD-image, object type: star, group: _stile_group_0\n2 - format: single-CCD-catalog, object type: star, group: _stile_group_3\n3 - format: single-field-catalog, object type: star, group: _stile_group_6')

        results, n = self.testConfigDataHandler._parseFileHelper(copy.deepcopy(self.dict4))
        self.assertEqual(results,self.expected_files4) 
        results, groups = self.testConfigDataHandler.parseFiles({'file':copy.deepcopy(self.dict4)})
        self.assertEqual(results,self.expected_files4)
        self.assertEqual(groups,self.expected_groups4)
        results, n = self.testConfigDataHandler._parseFileHelper(copy.deepcopy(self.dict5))
        self.assertEqual(results,self.expected_files_helper5)
        # parseFiles removes boolean groups
        results, groups = self.testConfigDataHandler.parseFiles({'file':copy.deepcopy(self.dict5)})
        self.assertEqual(results,self.expected_files5)
        self.assertEqual(groups,self.expected_groups5)
        results, n = self.testConfigDataHandler._parseFileHelper(copy.deepcopy(self.dict6))
        self.assertEqual(results,self.expected_file_helper6)
        results, groups = self.testConfigDataHandler.parseFiles({'file':copy.deepcopy(self.dict6)})
        self.assertEqual(results,self.expected_files6)
        self.assertEqual(groups,self.expected_groups6)
        results, n = self.testConfigDataHandler._parseFileHelper(copy.deepcopy(self.dict7))
        self.assertEqual(results,self.expected_files7)
        results, groups = self.testConfigDataHandler.parseFiles({'file':copy.deepcopy(self.dict7)})
        self.assertEqual(results,self.expected_files7)
        self.assertEqual(groups,self.expected_groups7)
        results, n = self.testConfigDataHandler._parseFileHelper(copy.deepcopy(self.dict8))
        self.assertEqual(results,self.expected_file_helper8)
        results, groups = self.testConfigDataHandler.parseFiles({'file':copy.deepcopy(self.dict8)})
        self.assertEqual(results,self.expected_files8)
        self.assertEqual(groups,self.expected_groups8)
        results, n = self.testConfigDataHandler._parseFileHelper(copy.deepcopy(self.dict9))
        self.assertEqual(results,self.expected_files9)
        results, groups = self.testConfigDataHandler.parseFiles({'file':copy.deepcopy(self.dict9)})
        self.assertEqual(results,self.expected_files9)
        self.assertEqual(groups,self.expected_groups9)
        # list0: list-form
        results, groups = self.testConfigDataHandler.parseFiles({'file':self.list0})
        self.assertEqual(results,self.expected_files_list0)
        self.assertEqual(groups,self.expected_groups_list0)
        # Now, just make sure that if you send multiple dicts through it combines them correctly
        results, groups = self.testConfigDataHandler.parseFiles({'file_0':copy.deepcopy(self.dict0),'file_6':copy.deepcopy(self.dict6)})
        expected_results = {stile_utils.Format(epoch='single',extent='CCD',data_format='catalog').str: {
                                'galaxy': [{'name': 'g1.dat', 'group': '_stile_group_0'},
                                           {'name': 'g2.dat', 'group': ['_stile_group_1']},
                                           {'name': 'g3.dat', 'group': ['_stile_group_2']},
                                           {'name': 'g1.dat', 'file_reader': 'ASCII'}],
                                'star':   [{'name': 's1.dat', 'group': ['_stile_group_0']},
                                           {'name': 's2.dat', 'group': '_stile_group_1'},
                                           {'name': 's3.dat', 'group': '_stile_group_2'},
                                           {'name': 's2.dat', 'file_reader': 'ASCII'},
                                           {'name': 's3.dat', 'file_reader': 'ASCII'}]}}
        expected_groups = {'_stile_group_0': {
                'single-CCD-catalog': { 'star': 0, 'galaxy': 0}}, 
            '_stile_group_1': {
                'single-CCD-catalog': { 'star': 1, 'galaxy': 1}},
            '_stile_group_2': {
                'single-CCD-catalog': { 'star': 2, 'galaxy': 2}}}
        self.assertEqual(results,expected_results)
        self.assertEqual(groups,expected_groups)
        # Finally, check that queryFile still works with multiple same file names in the same format & object type
        results = self.testConfigDataHandler.queryFile('g1.dat')
        self.assertEqual(results,'1 - format: single-CCD-catalog, object type: galaxy, group: _stile_group_0\n2 - format: single-CCD-catalog, object type: galaxy, file_reader: ASCII')
        # And check that it handles extra keys correctly as well
        results, groups = self.testConfigDataHandler.parseFiles({'file_0':copy.deepcopy(self.dict0),'file_6':copy.deepcopy(self.dict6),'file_reader':'ASCII'})
        expected_results = {stile_utils.Format(epoch='single',extent='CCD',data_format='catalog').str: {
                                'galaxy': [{'name': 'g1.dat', 'file_reader': 'ASCII', 'group': ['_stile_group_0']},
                                           {'name': 'g2.dat', 'file_reader': 'ASCII', 'group': ['_stile_group_1']},
                                           {'name': 'g3.dat', 'file_reader': 'ASCII', 'group': ['_stile_group_2']}],
                                'star':   [{'name': 's1.dat', 'file_reader': 'ASCII', 'group': ['_stile_group_0']},
                                           {'name': 's2.dat', 'file_reader': 'ASCII', 'group': ['_stile_group_1']},
                                           {'name': 's3.dat', 'file_reader': 'ASCII', 'group': ['_stile_group_2']}]}}
        self.assertEqual(results,expected_results)
        self.assertEqual(groups,expected_groups)
        results, groups = self.testConfigDataHandler.parseFiles({'file_0':copy.deepcopy(self.dict0),'file_6':copy.deepcopy(self.dict6),'file_reader':{'extent': 'CCD', 'name': 'ASCII'}})
        self.assertEqual(results,expected_results)
        self.assertEqual(groups,expected_groups)
        results, groups = self.testConfigDataHandler.parseFiles({'file_0':copy.deepcopy(self.dict0),'file_6':copy.deepcopy(self.dict6),'file_reader':{'star':'ASCII'}})
        expected_results = {stile_utils.Format(epoch='single',extent='CCD',data_format='catalog').str: {
                                'galaxy': [{'name': 'g1.dat', 'group': '_stile_group_0'},
                                           {'name': 'g2.dat', 'group': ['_stile_group_1']},
                                           {'name': 'g3.dat', 'group': ['_stile_group_2']},
                                           {'name': 'g1.dat', 'file_reader': 'ASCII'}],
                                'star':   [{'name': 's1.dat', 'file_reader': 'ASCII', 'group': ['_stile_group_0']},
                                           {'name': 's2.dat', 'file_reader': 'ASCII', 'group': ['_stile_group_1']},
                                           {'name': 's3.dat', 'file_reader': 'ASCII', 'group': ['_stile_group_2']}]}}
        self.assertEqual(results,expected_results)
        self.assertEqual(groups,expected_groups)
        results, groups = self.testConfigDataHandler.parseFiles({'file_0':copy.deepcopy(self.dict0),'file_6':copy.deepcopy(self.dict6),'file_reader':{'object_type': 'star', 'name': 'ASCII'}})
        self.assertEqual(results,expected_results)
        self.assertEqual(groups,expected_groups)

    def test_lists(self):
        """
        Test the ConfigDataHandler methods listFileTypes(), listObjects(), and listData(), which allow us to query the available data contained
        within the ConfigDataHandler.
        """
        # Set up a list of tuples: first item is the string to be appended to error messages; second is a Set object (defined above).
        # The Set object contains:
        # - A ConfigDataHandler object
        # - The expected self.files argument of the ConfigDataHandler
        # - A format key expected to be found in self.files
        # - An object key expected to be found in self.files[format]
        # - A format key NOT expected to be found in self.files
        # - An obejct key NOT expected to be found in self.files[format]
        test_sets = []
        test_sets.append(("(dict0)", Set(stile.ConfigDataHandler({'files': self.dict0}), self.expected_files0, 'single-CCD-catalog', 'galaxy', 'multiepoch-CCD-catalog', 'galaxy lens')))
        test_sets.append(("(dict1)", Set(stile.ConfigDataHandler({'files': self.dict1}), self.expected_files1, 'single-CCD-catalog', 'star', 'single-field-catalog', 'star bright')))
        test_sets.append(("(dict2)", Set(stile.ConfigDataHandler({'files': self.dict2}), self.expected_files2, 'single-CCD-catalog', 'galaxy', 'single-CCD-image', 'galaxy random')))
        test_sets.append(("(dict3)", Set(stile.ConfigDataHandler({'files': self.dict3}), self.expected_files3, 'single-CCD-image', 'galaxy', 'single-field-image', 'galaxy random')))
        test_sets.append(("(dict4 field)", Set(stile.ConfigDataHandler({'files': self.dict4}), self.expected_files4, 'single-field-catalog', 'star', 'multiepoch-CCD-catalog', 'galaxy random')))
        test_sets.append(("(dict4 CCD)", Set(test_sets[-1][1].config, self.expected_files4, 'single-CCD-catalog', 'star', 'single-field-image', 'galaxy lens')))
        test_sets.append(("(dict5)", Set(stile.ConfigDataHandler({'files': self.dict5}), self.expected_files5, 'single-CCD-image', 'galaxy', 'single-field-catalog', 'galaxy random')))
        # 6,7,8 are basically duplicates of earlier tests, just with different groupings, already tested
        test_sets.append(("(dict9 single)", Set(stile.ConfigDataHandler({'files': self.dict9}), self.expected_files9, 'single-CCD-catalog', 'galaxy', 'single-CCD-image', 'star')))
        test_sets.append(("(dict9 multiepoch)", Set(test_sets[-1][1].config, self.expected_files9, 'multiepoch-CCD-catalog', 'galaxy', 'multiepoch-CCD-image', 'star')))
        test_sets.append(("(list0)", Set(stile.ConfigDataHandler({'files': self.list0}), self.expected_files_list0, 'single-field-catalog', 'star', 'single-CCD-image', 'galaxy random')))

        # Now, loop through those defined test sets, checking that we get the expected formats and objects, and NOT the formats and objects we
        # don't expect
        for name, test_set in test_sets:
            results = test_set.config.listFileTypes()
            self.assertEqual(set(results), set(test_set.expected_files.keys()), msg='Failed to retrieve proper formats with listFileTypes '+name)
            results = test_set.config.listObjects(test_set.format)
            self.assertEqual(set(results), set(test_set.expected_files[test_set.format].keys()), msg='Failed to retrieve proper lists of objects with listObjects '+name)
            results = test_set.config.listData(test_set.object, test_set.format)
            self.assertEqual(results, test_set.expected_files[test_set.format][test_set.object], msg='Failed to retrieve proper list of files with listData '+name)
            results = test_set.config.listObjects(test_set.not_found_format)
            self.assertEqual(results, [], msg='Found a non-empty set for incorrect format in listObjects '+name)
            results = test_set.config.listData(test_set.object, test_set.not_found_format)
            self.assertEqual(results, [], msg='Found a non-empty set for incorrect format in listData '+name)
            results = test_set.config.listData(test_set.not_found_object, test_set.format)
            self.assertEqual(results, [], msg='Found a non-empty set for incorrect object in listData '+name)

    def test_systests(self):
        # We will use self.dict0 (simple), self.dict3 (more complicated), self.dict9 (only 1 object type per level) as our base file dicts for the sys_test cases.
        config_0 = stile.ConfigDataHandler({'files': self.dict0})
        config_3 = stile.ConfigDataHandler({'files': self.dict3})
        config_9 = stile.ConfigDataHandler({'files': self.dict9})
        
        results = config_0.parseSysTests({'sys_tests': copy.deepcopy(self.sys_tests_0)})
        expected_results = {'single-CCD-catalog': 
            [{'name': 'CorrelationFunctionSysTest', 'type': 'GalaxyShear'},
             {'name': 'CorrelationFunctionSysTest', 'type': 'BrightStarShear'},
             {'name': 'ScatterPlotSysTest', 'type': 'StarVsPSFG1'},
             {'name': 'StatSysTest', 'field': 'g1', 'object': 'galaxy'}]}
        self.assertEqual(results, expected_results)
        results = config_9.parseSysTests({'sys_tests': copy.deepcopy(self.sys_tests_0)})
        expected_results['multiepoch-CCD-catalog'] = []
        self.assertEqual(results, expected_results)
        del expected_results['multiepoch-CCD-catalog']
        results = config_3.parseSysTests({'sys_tests': copy.deepcopy(self.sys_tests_0)})
        expected_results['single-CCD-image'] = []
        expected_results['single-field-catalog'] = []
        self.assertEqual(results, expected_results)
        
        results = config_0.parseSysTests({'sys_tests': copy.deepcopy(self.sys_tests_1)})
        expected_results = {'single-CCD-catalog': 
            [{'name': 'CorrelationFunctionSysTest', 'type': 'GalaxyShear'},
             {'name': 'CorrelationFunctionSysTest', 'type': 'BrightStarShear'},
             {'name': 'CorrelationFunctionSysTest', 'type': 'StarXStarShear'},
             {'name': 'ScatterPlotSysTest', 'type': 'StarVsPSFG1'}]} # The last item of sys_tests_1 is an 'image' so shouldn't appear
        self.assertEqual(results, expected_results)
        results = config_9.parseSysTests({'sys_tests': copy.deepcopy(self.sys_tests_1)})
        expected_results['multiepoch-CCD-catalog'] = [{'name': 'CorrelationFunctionSysTest', 'type': 'BrightStarShear'},
            {'name': 'CorrelationFunctionSysTest', 'type': 'StarXStarShear'},
            {'name': 'ScatterPlotSysTest', 'type': 'StarVsPSFG1'}]
        self.assertEqual(results, expected_results)
        results = config_3.parseSysTests({'sys_tests': copy.deepcopy(self.sys_tests_1)})
        del expected_results['multiepoch-CCD-catalog']
        expected_results['single-CCD-image'] = [{'name': 'StatSysTest', 'field': 'g1', 'object': 'galaxy'}]
        expected_results['single-field-catalog'] = [{'name': 'CorrelationFunctionSysTest', 'type': 'GalaxyShear'},
            {'name': 'ScatterPlotSysTest', 'type': 'StarVsPSFG1'}]
        self.assertEqual(results, expected_results)

        results = config_0.parseSysTests({'sys_test_2': copy.deepcopy(self.sys_tests_2)})
        expected_results = {'single-CCD-catalog': 
            [{'name': 'CorrelationFunctionSysTest', 'type': 'GalaxyShear'},
             {'name': 'CorrelationFunctionSysTest', 'type': 'BrightStarShear'},
             {'name': 'ScatterPlotSysTest', 'type': 'StarVsPSFG1'},
             {'name': 'StatSysTest', 'field': 'g1', 'object': 'galaxy'}]}
        self.assertEqual(results, expected_results)
        results = config_9.parseSysTests({'sys_test_2': copy.deepcopy(self.sys_tests_2)})
        expected_results['multiepoch-CCD-catalog'] = []
        self.assertEqual(results, expected_results)
        del expected_results['multiepoch-CCD-catalog']
        results = config_3.parseSysTests({'sys_test_2': copy.deepcopy(self.sys_tests_2)})
        expected_results['single-CCD-image'] = []
        expected_results['single-field-catalog'] = []
        self.assertEqual(results, expected_results)
        
        results = config_0.parseSysTests({'sys_test_1': copy.deepcopy(self.sys_tests_3a), 'sys_test_2': copy.deepcopy(self.sys_tests_3b), 'sys_test_3': copy.deepcopy(self.sys_tests_3c)})
        expected_results = {'single-CCD-catalog': 
            [{'name': 'ScatterPlotSysTest', 'type': 'StarVsPSFG1'},
             {'name': 'CorrelationFunctionSysTest', 'type': 'BrightStarShear'},
             {'name': 'CorrelationFunctionSysTest', 'type': 'StarXStarShear'},
             {'name': 'CorrelationFunctionSysTest', 'type': 'GalaxyShear'}]}
        self.assertEqual(results, expected_results)
        results = config_9.parseSysTests({'sys_test_1': copy.deepcopy(self.sys_tests_3a), 'sys_test_2': copy.deepcopy(self.sys_tests_3b), 'sys_test_3': copy.deepcopy(self.sys_tests_3c)})
        expected_results['multiepoch-CCD-catalog'] = [{'name': 'ScatterPlotSysTest', 'type': 'StarVsPSFG1'},
            {'name': 'CorrelationFunctionSysTest', 'type': 'BrightStarShear'},
            {'name': 'CorrelationFunctionSysTest', 'type': 'StarXStarShear'}] 
        self.assertEqual(results, expected_results)
        del expected_results['multiepoch-CCD-catalog']
        results = config_3.parseSysTests({'sys_test_1': copy.deepcopy(self.sys_tests_3a), 'sys_test_2': copy.deepcopy(self.sys_tests_3b), 'sys_test_3': copy.deepcopy(self.sys_tests_3c)})
        expected_results['single-CCD-image'] = [{'name': 'StatSysTest', 'field': 'g1', 'object': 'galaxy'}]
        expected_results['single-field-catalog'] = [{'name': 'ScatterPlotSysTest', 'type': 'StarVsPSFG1'},
            {'name': 'CorrelationFunctionSysTest', 'type': 'GalaxyShear'}]
        self.assertEqual(results, expected_results)
        
        results = config_0.parseSysTests({'sys_test': copy.deepcopy(self.sys_tests_4)})
        expected_results = {'single-CCD-catalog': 
            [{'name': 'CorrelationFunctionSysTest', 'type': 'GalaxyShear'},
             {'name': 'CorrelationFunctionSysTest', 'type': 'BrightStarShear'}]}
        self.assertEqual(results, expected_results)
        results = config_9.parseSysTests({'sys_test': copy.deepcopy(self.sys_tests_4)})
        expected_results['multiepoch-CCD-catalog'] = []
        self.assertEqual(results, expected_results)
        del expected_results['multiepoch-CCD-catalog']
        results = config_3.parseSysTests({'sys_test': copy.deepcopy(self.sys_tests_4)})
        expected_results['single-CCD-image'] = [{'name': 'ScatterPlotSysTest', 'type': 'StarVsPSFG1'},
                            {'name': 'StatSysTest', 'field': 'g1', 'object': 'galaxy'}]
        expected_results['single-field-catalog'] = []
        self.assertEqual(results, expected_results)        
        
        results = config_0.parseSysTests({'sys_test': copy.deepcopy(self.sys_tests_5)})
        expected_results = {'single-CCD-catalog': []}
        self.assertEqual(results, expected_results)
        results = config_9.parseSysTests({'sys_test': copy.deepcopy(self.sys_tests_5)})
        expected_results['multiepoch-CCD-catalog'] = [{'name': 'CorrelationFunctionSysTest', 'type': 'GalaxyShear'},
                            {'name': 'CorrelationFunctionSysTest', 'type': 'BrightStarShear'}]
        self.assertEqual(results, expected_results)
        del expected_results['multiepoch-CCD-catalog']
        results = config_3.parseSysTests({'sys_test': copy.deepcopy(self.sys_tests_5)})
        expected_results['single-CCD-image'] = []
        expected_results['single-field-catalog'] = []
        self.assertEqual(results, expected_results)        
        
        
if __name__=='__main__':
    unittest.main()
