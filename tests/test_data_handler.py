try:
    import stile
except ImportError:
    import sys
    sys.path.append('..')
    import stile
from stile import stile_utils
import unittest
import copy
import numpy

class Set(object):
    """
    Container class to hold information about format-keyed file dicts.
    """
    def __init__(self, config, expected_files, format, object, not_found_format, not_found_object):
        self.config = config
        self.expected_files = expected_files
        self.format = format
        self.object = object
        self.not_found_format = not_found_format
        self.not_found_object = not_found_object

def CompareTest(t1, t2):
    same_bool = (t1['bin_list']==t2['bin_list'] and t1['extra_args']==t2['extra_args'] and
                 type(t1['sys_test'])==type(t2['sys_test']))
    if same_bool and isinstance(t1['sys_test'], stile.StatSysTest):
        same_bool &= t1['sys_test'].field==t2['sys_test'].field and t1['sys_test'].objects_list==t2['sys_test'].objects_list
    return same_bool
        
class TestDataHandler(unittest.TestCase):
    def setUp(self):
        # An empty "files" keyword will make the DataHandler fail, so do this and don't bother
        # checking the outputs till the end.
        self.testConfigDataHandler = stile.ConfigDataHandler({})
        # dict0: simple
        self.dict0 = {'single': {
            'CCD': {
                'catalog': {
                    'galaxy': ['g1.dat', 'g2.dat', 'g3.dat'],
                    'star':   ['s1.dat', 's2.dat', 's3.dat']
            } } } }
        self.expected_files0 = {'single-CCD-catalog': {
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
                    'galaxy': [{'name': 'g1.dat', 'file_reader': 'ASCII'},
                               {'name': 'g2.dat', 'file_reader': 'ASCII'},
                               {'name': 'g3.dat', 'file_reader': 'ASCII'}],
                    'star':   [{'name': 's1.dat', 'file_reader': 'ASCII'},
                               {'name': 's2.dat', 'file_reader': 'ASCII'},
                               {'name': 's3.dat', 'file_reader': 'ASCII'}]
            } } } }
        self.expected_files1 = {'single-CCD-catalog': {
                                'galaxy': [{'name': 'g1.dat', 'file_reader': 'ASCII',
                                            'group': '_stile_group_0'},
                                           {'name': 'g2.dat', 'file_reader': 'ASCII',
                                            'group': '_stile_group_1'},
                                           {'name': 'g3.dat', 'file_reader': 'ASCII',
                                            'group': '_stile_group_2'}],
                                'star':   [{'name': 's1.dat', 'file_reader': 'ASCII',
                                            'group': '_stile_group_0'},
                                           {'name': 's2.dat', 'file_reader': 'ASCII',
                                            'group': '_stile_group_1'},
                                           {'name': 's3.dat', 'file_reader': 'ASCII',
                                            'group': '_stile_group_2'}]}}
        self.expected_groups1 = self.expected_groups0

        # dict2: the file lists are mixed
        self.dict2 = {'single': {
            'CCD': {
                'catalog': {
                    'galaxy': [{'name': 'g1.dat', 'file_reader': 'ASCII'}, 'g2.dat', 'g3.dat'],
                    'star':   ['s1.dat', {'name': 's2.dat', 'file_reader': 'ASCII'},
                               {'name': 's3.dat', 'file_reader': 'ASCII'}]
            } } } }
        self.expected_files2 = {'single-CCD-catalog': {
                                'galaxy': [{'name': 'g1.dat', 'file_reader': 'ASCII',
                                            'group': '_stile_group_0'},
                                           {'name': 'g2.dat', 'group': '_stile_group_1'},
                                           {'name': 'g3.dat', 'group': '_stile_group_2'}],
                                'star':   [{'name': 's1.dat', 'group': '_stile_group_0'},
                                           {'name': 's2.dat', 'file_reader': 'ASCII',
                                            'group': '_stile_group_1'},
                                           {'name': 's3.dat', 'file_reader': 'ASCII',
                                            'group': '_stile_group_2'}]}}
        self.expected_groups2 = self.expected_groups0

        # dict3: multiple levels
        self.dict3 = {'single': {
            'CCD': {
                'catalog': {
                    'galaxy': [{'name': 'g1.dat', 'file_reader': 'ASCII'}, 'g2.dat', 'g3.dat'],
                    'star':   ['s1.dat', {'name': 's2.dat', 'file_reader': 'ASCII'},
                               {'name': 's3.dat', 'file_reader': 'ASCII'}]
                    },
                'image': {
                    'galaxy': [{'name': 'g1.dat', 'file_reader': 'ASCII'}, 'g2.dat', 'g3.dat'],
                    'star':   ['s1.dat', {'name': 's2.dat', 'file_reader': 'ASCII'},
                               {'name': 's3.dat', 'file_reader': 'ASCII'}]
                    } },
            'field': {
                'catalog': {
                    'galaxy': [{'name': 'g1.dat', 'file_reader': 'ASCII'}, 'g2.dat', 'g3.dat'],
                    'star':   ['s1.dat', {'name': 's2.dat', 'file_reader': 'ASCII'},
                               {'name': 's3.dat', 'file_reader': 'ASCII'}]
            } } } }
        self.expected_files3 = {'single-CCD-image': {
                                'galaxy': [{'name': 'g1.dat', 'file_reader': 'ASCII',
                                            'group': '_stile_group_0'},
                                           {'name': 'g2.dat', 'group': '_stile_group_1'},
                                           {'name': 'g3.dat', 'group': '_stile_group_2'}],
                                'star':   [{'name': 's1.dat', 'group': '_stile_group_0'},
                                           {'name': 's2.dat', 'file_reader': 'ASCII',
                                            'group': '_stile_group_1'},
                                           {'name': 's3.dat', 'file_reader': 'ASCII',
                                            'group': '_stile_group_2'}]},
                            'single-CCD-catalog': {
                                'galaxy': [{'name': 'g1.dat', 'file_reader': 'ASCII',
                                            'group': '_stile_group_3'},
                                           {'name': 'g2.dat', 'group': '_stile_group_4'},
                                           {'name': 'g3.dat', 'group': '_stile_group_5'}],
                                'star':   [{'name': 's1.dat', 'group': '_stile_group_3'},
                                           {'name': 's2.dat', 'file_reader': 'ASCII',
                                            'group': '_stile_group_4'},
                                           {'name': 's3.dat', 'file_reader': 'ASCII',
                                            'group': '_stile_group_5'}]},
                            'single-field-catalog': {
                                'galaxy': [{'name': 'g1.dat', 'file_reader': 'ASCII',
                                            'group': '_stile_group_6'},
                                           {'name': 'g2.dat', 'group': '_stile_group_7'},
                                           {'name': 'g3.dat', 'group': '_stile_group_8'}],
                                'star':   [{'name': 's1.dat', 'group': '_stile_group_6'},
                                           {'name': 's2.dat', 'file_reader': 'ASCII',
                                            'group': '_stile_group_7'},
                                           {'name': 's3.dat', 'file_reader': 'ASCII',
                                            'group': '_stile_group_8'}]}}
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
                    'galaxy': [{'name': 'g1.dat', 'file_reader': 'ASCII'}, 'g2.dat', 'g3.dat'],
                    'star':   ['s1.dat', {'name': 's2.dat', 'file_reader': 'ASCII'},
                               {'name': 's3.dat', 'file_reader': 'ASCII'}]
                },
                'image': {
                    'galaxy': [{'name': 'g1.dat', 'file_reader': 'ASCII'}, 'g2.dat', 'g3.dat'],
                    'star':   ['s1.dat', {'name': 's2.dat', 'file_reader': 'ASCII'},
                               {'name': 's3.dat', 'file_reader': 'ASCII'}]
                } },
            'field': {
                'galaxy': {
                    'catalog': [{'name': 'g1.dat', 'file_reader': 'ASCII'}, 'g2.dat', 'g3.dat'], },
                'star': {
                    'catalog': ['s1.dat', {'name': 's2.dat', 'file_reader': 'ASCII'},
                              {'name': 's3.dat', 'file_reader': 'ASCII'}]}
            } } }
        self.expected_files4 = self.expected_files3
        self.expected_groups4 = self.expected_groups3

        # dict5: multiple levels, add some non-format-related kwargs at different levels
        # [note: wildcard means ['field'] should be empty since the files don't exist!]
        self.dict5 = {'single': {
            'CCD': {
                'catalog': {
                    'galaxy': [{'name': 'g1.dat', 'file_reader': 'ASCII'}, 'g2.dat', 'g3.dat'],
                    'star':   ['s1.dat', {'name': 's2.dat', 'file_reader': 'ASCII'},
                               {'name': 's3.dat', 'file_reader': 'ASCII'}]
                    },
                'image': {
                    'group': False,
                    'galaxy': [{'name': 'g1.dat', 'file_reader': 'ASCII'}, 'g2.dat', 'g3.dat'],
                    'star':   ['s1.dat', {'name': 's2.dat', 'file_reader': 'ASCII'},
                               {'name': 's3.dat', 'file_reader': 'ASCII'}]
                    },
                },
            'field': {
                'wildcard': True,
                'catalog': {
                    'galaxy': [{'name': 'g1.dat', 'file_reader': 'ASCII'}, 'g2.dat', 'g3.dat'],
                    'star':   ['s1.dat', {'name': 's2.dat', 'file_reader': 'ASCII'},
                               {'name': 's3.dat', 'file_reader': 'ASCII'}]
                    },
                'image': {
                    'galaxy': [{'name': 'g1.dat', 'file_reader': 'ASCII'}, 'g2.dat', 'g3.dat'],
                    'star':   ['s1.dat', {'name': 's2.dat', 'file_reader': 'ASCII'},
                               {'name': 's3.dat', 'file_reader': 'ASCII'}]
                    }
                }
            } }
        # parseFilse removes Boolean groups that _parseFileHelper doesn't, so we need 2 "expected
        # files" here
        self.expected_files_helper5 = {'single-CCD-catalog': {
                                'galaxy': [{'name': 'g1.dat', 'file_reader': 'ASCII',
                                            'group': '_stile_group_0'},
                                           {'name': 'g2.dat', 'group': '_stile_group_1'},
                                           {'name': 'g3.dat', 'group': '_stile_group_2'}],
                                'star':   [{'name': 's1.dat', 'group': '_stile_group_0'},
                                           {'name': 's2.dat', 'file_reader': 'ASCII',
                                            'group': '_stile_group_1'},
                                           {'name': 's3.dat', 'file_reader': 'ASCII',
                                            'group': '_stile_group_2'}]},
                            'single-CCD-image': {
                                'galaxy': [{'name': 'g1.dat', 'file_reader': 'ASCII',
                                            'group': False},
                                           {'name': 'g2.dat', 'group': False},
                                           {'name': 'g3.dat', 'group': False}],
                                'star':   [{'name': 's1.dat', 'group': False},
                                           {'name': 's2.dat', 'file_reader': 'ASCII',
                                            'group': False},
                                           {'name': 's3.dat', 'file_reader': 'ASCII',
                                            'group': False}]}}
        self.expected_files5 = {'single-CCD-catalog': {
                                'galaxy': [{'name': 'g1.dat', 'file_reader': 'ASCII',
                                            'group': '_stile_group_0'},
                                           {'name': 'g2.dat', 'group': '_stile_group_1'},
                                           {'name': 'g3.dat', 'group': '_stile_group_2'}],
                                'star':   [{'name': 's1.dat', 'group': '_stile_group_0'},
                                           {'name': 's2.dat', 'file_reader': 'ASCII',
                                            'group': '_stile_group_1'},
                                           {'name': 's3.dat', 'file_reader': 'ASCII',
                                            'group': '_stile_group_2'}]},
                            'single-CCD-image': {
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
                    'galaxy': [{'name': 'g1.dat', 'file_reader': 'ASCII'}, 'g2.dat', 'g3.dat'],
                    'star':   ['s1.dat', {'name': 's2.dat', 'file_reader': 'ASCII'},
                               {'name': 's3.dat', 'file_reader': 'ASCII', 'group': False}]
            } } } }
        self.expected_file_helper6 = {'single-CCD-catalog': {
                                'galaxy': [{'name': 'g1.dat', 'file_reader': 'ASCII'},
                                           {'name': 'g2.dat'},
                                           {'name': 'g3.dat'}],
                                'star':   [{'name': 's1.dat'},
                                           {'name': 's2.dat', 'file_reader': 'ASCII'},
                                           {'name': 's3.dat', 'file_reader': 'ASCII',
                                            'group': False}]}}
        self.expected_files6 = {'single-CCD-catalog': {
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
                    'galaxy': [{'name': 'g1.dat', 'group': 2}, {'name': 'g2.dat', 'group': 1},
                               {'name': 'g3.dat', 'group': 3}],
                    'star':   [{'name': 's1.dat', 'group': 3}, {'name': 's2.dat', 'group': 2},
                               {'name': 's3.dat', 'group': 1}]
            } } } }
        self.expected_files7 = {'single-CCD-catalog': {
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
                    'galaxy': [{'name': 'g1.dat', 'file_reader': 'ASCII'}, 'g3.dat'],
                    'star':   ['s1.dat', {'name': 's2.dat', 'file_reader': 'ASCII', 'group': False},
                               {'name': 's3.dat', 'file_reader': 'ASCII'}]
            } } } }
        self.expected_file_helper8 = {'single-CCD-catalog': {
                                'galaxy': [{'name': 'g1.dat', 'file_reader': 'ASCII',
                                            'group': '_stile_group_0'},
                                           {'name': 'g3.dat', 'group': '_stile_group_1'}],
                                'star':   [{'name': 's1.dat', 'group': '_stile_group_0'},
                                           {'name': 's2.dat', 'file_reader': 'ASCII',
                                            'group': False},
                                           {'name': 's3.dat', 'file_reader': 'ASCII',
                                            'group': '_stile_group_1'}]}}
        self.expected_files8 = {'single-CCD-catalog': {
                                'galaxy': [{'name': 'g1.dat', 'file_reader': 'ASCII',
                                            'group': '_stile_group_0'},
                                           {'name': 'g3.dat', 'group': '_stile_group_1'}],
                                'star':   [{'name': 's1.dat', 'group': '_stile_group_0'},
                                           {'name': 's2.dat', 'file_reader': 'ASCII'},
                                           {'name': 's3.dat', 'file_reader': 'ASCII',
                                            'group': '_stile_group_1'}]}}
        self.expected_groups8 = {'_stile_group_0': {
                'single-CCD-catalog': { 'star': 0, 'galaxy': 0}},
            '_stile_group_1': {
                'single-CCD-catalog': { 'star': 2, 'galaxy': 1}}}


        # dict9: the dreaded multiepoch
        self.dict9 = {'single': {
                'CCD': {
                    'catalog': {
                        'galaxy': [{'name': ['g1-0.dat', 'g1-1.dat'], 'file_reader': 'ASCII'},
                                   ['g2-0.dat', 'g2-1.dat'],
                                   ['g3-0.dat', 'g3-1.dat']],
                    } } },
            'multiepoch': {
                'CCD': {
                    'catalog': {
                        'galaxy': [{'name': ['g1-0.dat', 'g1-1.dat'], 'file_reader': 'ASCII'},
                                   ['g2-0.dat', 'g2-1.dat'],
                                   ['g3-0.dat', 'g3-1.dat']],
            } } } }
        self.expected_files9 = {'single-CCD-catalog': {
                                'galaxy': [{'name': 'g1-0.dat', 'file_reader': 'ASCII'},
                                           {'name': 'g1-1.dat', 'file_reader': 'ASCII'},
                                           {'name': 'g2-0.dat'},
                                           {'name': 'g2-1.dat'},
                                           {'name': 'g3-0.dat'},
                                           {'name': 'g3-1.dat'}]},
                            'multiepoch-CCD-catalog': {
                                'galaxy': [{'name': ['g1-0.dat', 'g1-1.dat'],
                                            'file_reader': 'ASCII'},
                                           {'name': ['g2-0.dat', 'g2-1.dat']},
                                           {'name': ['g3-0.dat', 'g3-1.dat']}]
                           } }
        self.expected_groups9 = {}

        # list0: list-form
        self.list0 = [
                {'name': 'sg1.dat', 'epoch': 'single', 'extent': 'field', 'data_format': 'catalog',
                 'object_type': 'galaxy', 'flag_col': 'is_galaxy'},
                {'name': 'sg1.dat', 'epoch': 'single', 'extent': 'field', 'data_format': 'catalog',
                 'object_type': 'star', 'flag_col': 'is_star'}
            ]
        self.expected_files_list0 = {'single-field-catalog': {
                                'galaxy': [{'name': 'sg1.dat', 'flag_col': 'is_galaxy'}],
                                'star':   [{'name': 'sg1.dat', 'flag_col': 'is_star'}]
                            }}
        self.expected_groups_list0 = {}

        # bins0: like dict0, but with binning defined for two of the files
        self.bins0 = {'single': {
            'CCD': {
                'catalog': {
                    'galaxy': [{'name': 'g1.dat', 
                                'bins': [{'name': 'Step', 'field': 'ra', 'n_bins': 2, 'low': 0, 
                                          'high': 2}]}, 
                               'g2.dat', 'g3.dat'],
                    'star':   [{'name': 's1.dat',
                                'bins': [{'name': 'List', 'field': 'ra', 'endpoints': [0,1,2]}]}, 
                               's2.dat', 's3.dat']
            } } } }
        self.expected_files_bins0 = {'single-CCD-catalog': {
                                'galaxy': [{'name': 'g1.dat', 'group': '_stile_group_0', 
                                            'bins': [{'name': 'Step', 'field': 'ra', 'n_bins': 2, 
                                            'low': 0, 'high': 2}]},
                                           {'name': 'g2.dat', 'group': '_stile_group_1'},
                                           {'name': 'g3.dat', 'group': '_stile_group_2'}],
                                'star':   [{'name': 's1.dat', 'group': '_stile_group_0', 
                                            'bins': [{'name': 'List', 'field': 'ra', 
                                            'endpoints': [0,1,2]}]}, 
                                           {'name': 's2.dat', 'group': '_stile_group_1'},
                                           {'name': 's3.dat', 'group': '_stile_group_2'}]}}
        self.expected_groups_bins0 = {'_stile_group_0': {
                'single-CCD-catalog': { 'star': 0, 'galaxy': 0}},
            '_stile_group_1': {
                'single-CCD-catalog': { 'star': 1, 'galaxy': 1}},
            '_stile_group_2': {
                'single-CCD-catalog': { 'star': 2, 'galaxy': 2}}}

        # bins1: test binning with multiepoch data sets
        self.bins1 = {'multiepoch': {
            'CCD': {
                'catalog': {
                    'galaxy': [{'name': ['g1.dat', 'g2.dat', 'g3.dat'],
                                'bins': [{'name': 'Step', 'field': 'ra', 'n_bins': 2, 'low': 0, 
                                          'high': 2}]}],
                    'star':   [{'name': ['s1.dat', 's2.dat'],
                                'bins': [{'name': 'List', 'field': 'ra', 'endpoints': [0,1,2]}]}]
            } } } }
        self.expected_files_bins1 = {'multiepoch-CCD-catalog': {
                                'galaxy': [{'name': ['g1.dat', 'g2.dat', 'g3.dat'], 'group': '_stile_group_0', 
                                            'bins': [{'name': 'Step', 'field': 'ra', 'n_bins': 2, 
                                            'low': 0, 'high': 2}]}],
                                'star':   [{'name': ['s1.dat', 's2.dat'], 'group': '_stile_group_0', 
                                            'bins': [{'name': 'List', 'field': 'ra', 
                                            'endpoints': [0,1,2]}]}]}}
        self.expected_groups_bins1 = {'_stile_group_0': {
                'multiepoch-CCD-catalog': { 'star': 0, 'galaxy': 0}}}

        # sys tests 0: list form, complete description
        self.sys_tests_0 = [{'epoch': 'single', 'extent': 'CCD', 'data_format': 'catalog',
                             'name': 'CorrelationFunction', 'type': 'GalaxyShear'},
            {'epoch': 'single', 'extent': 'CCD', 'data_format': 'catalog',
             'name': 'CorrelationFunction', 'type': 'BrightStarShear'},
            {'epoch': 'single', 'extent': 'CCD', 'data_format': 'catalog',
             'name': 'ScatterPlot', 'type': 'StarVsPSFG1'},
            {'epoch': 'single', 'extent': 'CCD', 'data_format': 'catalog',
             'name': 'Stat', 'field': 'g1', 'object_type': 'galaxy'}]
        # sys tests 1: list form, incomplete description
        self.sys_tests_1 = [{'epoch': 'single', 'data_format': 'catalog',
                             'name': 'CorrelationFunction', 'type': 'GalaxyShear'},
            {'extent': 'CCD', 'data_format': 'catalog', 'name': 'CorrelationFunction',
             'type': 'BrightStarShear'},
            {'extent': 'CCD', 'data_format': 'catalog', 'name': 'CorrelationFunction',
             'type': 'StarXStarShear'},
            {'data_format': 'catalog', 'name': 'ScatterPlot', 'type': 'StarVsPSFG1'},
            # This last one doesn't really make sense, but as a test
            {'data_format': 'image', 'name': 'Stat', 'field': 'g1', 'object_type': 'galaxy'}]
        # sys tests 2: nested form, complete description
        self.sys_tests_2 = {'single': {
            'CCD': {
                'catalog': [{'name': 'CorrelationFunction', 'type': 'GalaxyShear'},
                            {'name': 'CorrelationFunction', 'type': 'BrightStarShear'},
                            {'name': 'ScatterPlot', 'type': 'StarVsPSFG1'},
                            {'name': 'Stat', 'field': 'g1', 'object_type': 'galaxy'}] } } }
        # sys tests 3: nested form, incomplete description.
        # Note that due to processing requirements we can't do these all as one dict
        self.sys_tests_3a = {
            'catalog': [{'name': 'ScatterPlot', 'type': 'StarVsPSFG1'}],
            'image': [{'name': 'Stat', 'field': 'g1', 'object_type': 'galaxy'}]}
        self.sys_tests_3b = {
            'CCD': {
                'catalog': [{'name': 'CorrelationFunction', 'type': 'BrightStarShear'},
                            {'name': 'CorrelationFunction', 'type': 'StarXStarShear'}]}}
        self.sys_tests_3c = {
            'single': [{'data_format': 'catalog', 'name': 'CorrelationFunction',
                        'type': 'GalaxyShear'}]}
        # sys tests 4: nested form, complete description, more complex
        self.sys_tests_4 = {'single': {
            'CCD': {
                'catalog': [{'name': 'CorrelationFunction', 'type': 'GalaxyShear'},
                            {'name': 'CorrelationFunction', 'type': 'BrightStarShear'}],
                'image':   [{'name': 'ScatterPlot', 'type': 'StarVsPSFG1'},
                            {'name': 'Stat', 'field': 'g1', 'object_type': 'galaxy'}] } } }
        # sys tests 5: nested form, with multiepoch since that processing is slightly different
        self.sys_tests_5 = {'multiepoch': {
            'CCD': {
                'catalog': [{'name': 'CorrelationFunction', 'type': 'GalaxyShear'},
                            {'name': 'CorrelationFunction', 'type': 'BrightStarShear'}],
                'image':   [{'name': 'ScatterPlot', 'type': 'StarVsPSFG1'},
                            {'name': 'Stat', 'field': 'g1', 'object_type': 'galaxy'}] } } }


    def test_parseFileHelper(self):
        """
        Test the internals of the parseFiles method.
        """
        # First make sure the file list is built correctly
        results, n = self.testConfigDataHandler._parseFileHelper(copy.deepcopy(self.dict0))
        self.assertEqual(results, self.expected_files0)
        # Next, make sure the group list is built correctly
        results, groups = self.testConfigDataHandler.parseFiles({'file': copy.deepcopy(self.dict0)})
        self.assertEqual(results, self.expected_files0)
        self.assertEqual(groups, self.expected_groups0)
        # Also check: does file querying work?
        # This test may need to be made more flexible later--it relies in the internal order of the
        # dict, which may be different between different Python installations.
        results = self.testConfigDataHandler.queryFile('s1.dat')
        self.assertEqual(results,
                         '1 - format: single-CCD-catalog, object type: star, group: _stile_group_0')
        # Dummy check: are these functions removing duplicates properly?  ('group' was turned into a
        # list here...)
        expected_results = {'single-CCD-catalog': {
                                'galaxy': [{'name': 'g1.dat', 'group': ['_stile_group_0']},
                                           {'name': 'g2.dat', 'group': ['_stile_group_1']},
                                           {'name': 'g3.dat', 'group': ['_stile_group_2']}],
                                'star':   [{'name': 's1.dat', 'group': ['_stile_group_0']},
                                           {'name': 's2.dat', 'group': ['_stile_group_1']},
                                           {'name': 's3.dat', 'group': ['_stile_group_2']}]}}
        results, groups = self.testConfigDataHandler.parseFiles({'file': copy.deepcopy(self.dict0),
                                                                'file2': copy.deepcopy(self.dict0)})
        self.assertEqual(results, expected_results)
        self.assertEqual(groups, self.expected_groups0)
        
        # Quickly test that empty keys are properly skipped
        skip_dict = copy.deepcopy(self.dict0)
        skip_dict['single']['field'] = {}
        results, n = self.testConfigDataHandler._parseFileHelper(copy.deepcopy(self.dict0))
        self.assertEqual(results, self.expected_files0)
        
        # Repeat (skipping the dummy check) for dict1, 2, 3 etc...
        results, n = self.testConfigDataHandler._parseFileHelper(copy.deepcopy(self.dict1))
        self.assertEqual(results, self.expected_files1)
        results, groups = self.testConfigDataHandler.parseFiles({'file0':
            copy.deepcopy(self.dict1)})
        self.assertEqual(results, self.expected_files1)
        self.assertEqual(groups, self.expected_groups1)
        results, n = self.testConfigDataHandler._parseFileHelper(copy.deepcopy(self.dict2))
        self.assertEqual(results, self.expected_files2)
        results, groups = self.testConfigDataHandler.parseFiles({'filelist':
            copy.deepcopy(self.dict2)})
        self.assertEqual(results, self.expected_files2)
        self.assertEqual(groups, self.expected_groups2)
        results, n = self.testConfigDataHandler._parseFileHelper(copy.deepcopy(self.dict3))
        self.assertEqual(results, self.expected_files3)
        results, groups = self.testConfigDataHandler.parseFiles({'file99':copy.deepcopy(self.dict3),
            'notfile': self.dict2}) # make sure it ignores notfile
        self.assertEqual(results, self.expected_files3)
        self.assertEqual(groups, self.expected_groups3)
        # try queryFile with multiple files
        results = self.testConfigDataHandler.queryFile('s1.dat')
        self.assertEqual(results, '1 - format: single-CCD-image, object type: star, group: '+
                         '_stile_group_0\n2 - format: single-CCD-catalog, object type: star, '+
                         'group: _stile_group_3\n3 - format: single-field-catalog, object type: '+
                         'star, group: _stile_group_6')

        results, n = self.testConfigDataHandler._parseFileHelper(copy.deepcopy(self.dict4))
        self.assertEqual(results, self.expected_files4)
        results, groups = self.testConfigDataHandler.parseFiles({'file':copy.deepcopy(self.dict4)})
        self.assertEqual(results, self.expected_files4)
        self.assertEqual(groups, self.expected_groups4)
        results, n = self.testConfigDataHandler._parseFileHelper(copy.deepcopy(self.dict5))
        self.assertEqual(results, self.expected_files_helper5)
        # parseFiles removes boolean groups
        results, groups = self.testConfigDataHandler.parseFiles({'file':copy.deepcopy(self.dict5)})
        self.assertEqual(results, self.expected_files5)
        self.assertEqual(groups, self.expected_groups5)
        results, n = self.testConfigDataHandler._parseFileHelper(copy.deepcopy(self.dict6))
        self.assertEqual(results, self.expected_file_helper6)
        results, groups = self.testConfigDataHandler.parseFiles({'file':copy.deepcopy(self.dict6)})
        self.assertEqual(results, self.expected_files6)
        self.assertEqual(groups, self.expected_groups6)
        results, n = self.testConfigDataHandler._parseFileHelper(copy.deepcopy(self.dict7))
        self.assertEqual(results, self.expected_files7)
        results, groups = self.testConfigDataHandler.parseFiles({'file':copy.deepcopy(self.dict7)})
        self.assertEqual(results, self.expected_files7)
        self.assertEqual(groups, self.expected_groups7)
        results, n = self.testConfigDataHandler._parseFileHelper(copy.deepcopy(self.dict8))
        self.assertEqual(results, self.expected_file_helper8)
        results, groups = self.testConfigDataHandler.parseFiles({'file':copy.deepcopy(self.dict8)})
        self.assertEqual(results, self.expected_files8)
        self.assertEqual(groups, self.expected_groups8)
        results, n = self.testConfigDataHandler._parseFileHelper(copy.deepcopy(self.dict9))
        self.assertEqual(results, self.expected_files9)
        results, groups = self.testConfigDataHandler.parseFiles({'file':copy.deepcopy(self.dict9)})
        self.assertEqual(results, self.expected_files9)
        self.assertEqual(groups, self.expected_groups9)
        # list0: list-form
        results, groups = self.testConfigDataHandler.parseFiles({'file':self.list0})
        self.assertEqual(results, self.expected_files_list0)
        self.assertEqual(groups, self.expected_groups_list0)
        # bins0: with bins
        results, groups = self.testConfigDataHandler.parseFiles({'file':self.bins0})
        self.assertEqual(results, self.expected_files_bins0)
        self.assertEqual(groups, self.expected_groups_bins0)
        # bins1: multiepoch with bins
        results, groups = self.testConfigDataHandler.parseFiles({'file':self.bins1})
        self.assertEqual(results, self.expected_files_bins1)
        self.assertEqual(groups, self.expected_groups_bins1)
        # Now, just make sure that if you send multiple dicts through it combines them correctly
        results, groups = self.testConfigDataHandler.parseFiles({'file_0':copy.deepcopy(self.dict0),
                                                                'file_6':copy.deepcopy(self.dict6)})
        expected_results = {'single-CCD-catalog': {
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
        self.assertEqual(results, expected_results)
        self.assertEqual(groups, expected_groups)
        # Finally, check that queryFile still works with multiple same file names in the same format
        # & object type
        results = self.testConfigDataHandler.queryFile('g1.dat')
        self.assertEqual(results, '1 - format: single-CCD-catalog, object type: galaxy, group: '+
            '_stile_group_0\n2 - format: single-CCD-catalog, object type: galaxy, file_reader: '+
            'ASCII')
        # And check that it handles extra keys correctly as well
        results, groups = self.testConfigDataHandler.parseFiles({'file_0':copy.deepcopy(self.dict0),
            'file_6':copy.deepcopy(self.dict6), 'file_reader':'ASCII'})
        expected_results = {'single-CCD-catalog': {
                                'galaxy': [{'name': 'g1.dat', 'file_reader': 'ASCII',
                                            'group': ['_stile_group_0']},
                                           {'name': 'g2.dat', 'file_reader': 'ASCII',
                                            'group': ['_stile_group_1']},
                                           {'name': 'g3.dat', 'file_reader': 'ASCII',
                                            'group': ['_stile_group_2']}],
                                'star':   [{'name': 's1.dat', 'file_reader': 'ASCII',
                                            'group': ['_stile_group_0']},
                                           {'name': 's2.dat', 'file_reader': 'ASCII',
                                            'group': ['_stile_group_1']},
                                           {'name': 's3.dat', 'file_reader': 'ASCII',
                                            'group': ['_stile_group_2']}]}}
        self.assertEqual(results, expected_results)
        self.assertEqual(groups, expected_groups)
        results, groups = self.testConfigDataHandler.parseFiles({'file_0':copy.deepcopy(self.dict0),
            'file_6':copy.deepcopy(self.dict6), 'file_reader':{'extent': 'CCD', 'name': 'ASCII'}})
        self.assertEqual(results, expected_results)
        self.assertEqual(groups, expected_groups)
        results, groups = self.testConfigDataHandler.parseFiles({'file_0':copy.deepcopy(self.dict0),
            'file_6':copy.deepcopy(self.dict6), 'file_reader':{'star':'ASCII'}})
        expected_results = {'single-CCD-catalog': {
                                'galaxy': [{'name': 'g1.dat', 'group': '_stile_group_0'},
                                           {'name': 'g2.dat', 'group': ['_stile_group_1']},
                                           {'name': 'g3.dat', 'group': ['_stile_group_2']},
                                           {'name': 'g1.dat', 'file_reader': 'ASCII'}],
                                'star':   [{'name': 's1.dat', 'file_reader': 'ASCII',
                                            'group': ['_stile_group_0']},
                                           {'name': 's2.dat', 'file_reader': 'ASCII',
                                            'group': ['_stile_group_1']},
                                           {'name': 's3.dat', 'file_reader': 'ASCII',
                                            'group': ['_stile_group_2']}]}}
        self.assertEqual(results, expected_results)
        self.assertEqual(groups, expected_groups)
        results, groups = self.testConfigDataHandler.parseFiles({'file_0':copy.deepcopy(self.dict0),
            'file_6':copy.deepcopy(self.dict6),
            'file_reader':{'object_type': 'star', 'name': 'ASCII'}})
        self.assertEqual(results, expected_results)
        self.assertEqual(groups, expected_groups)

    def test_lists(self):
        """
        Test the ConfigDataHandler methods listFileTypes(), listObjects(), and listData(), which
        allow us to query the available data contained within the ConfigDataHandler.
        """
        # Set up a list of tuples: first item is the string to be appended to error messages; second
        # is a Set object (defined above).
        # The Set object contains:
        # - A ConfigDataHandler object
        # - The expected self.files argument of the ConfigDataHandler
        # - A format key expected to be found in self.files
        # - An object key expected to be found in self.files[format]
        # - A format key NOT expected to be found in self.files
        # - An obejct key NOT expected to be found in self.files[format]
        test_sets = []
        test_sets.append(("(dict0)", Set(stile.ConfigDataHandler({'files': self.dict0}),
                                         self.expected_files0, 'single-CCD-catalog', 'galaxy',
                                         'multiepoch-CCD-catalog', 'galaxy lens')))
        test_sets.append(("(dict1)", Set(stile.ConfigDataHandler({'files': self.dict1}),
                                         self.expected_files1, 'single-CCD-catalog', 'star',
                                         'single-field-catalog', 'star bright')))
        test_sets.append(("(dict2)", Set(stile.ConfigDataHandler({'files': self.dict2}),
                                         self.expected_files2, 'single-CCD-catalog', 'galaxy',
                                         'single-CCD-image', 'galaxy random')))
        test_sets.append(("(dict3)", Set(stile.ConfigDataHandler({'files': self.dict3}),
                                         self.expected_files3, 'single-CCD-image', 'galaxy',
                                         'single-field-image', 'galaxy random')))
        test_sets.append(("(dict4 field)", Set(stile.ConfigDataHandler({'files': self.dict4}),
                                               self.expected_files4, 'single-field-catalog', 'star',
                                               'multiepoch-CCD-catalog', 'galaxy random')))
        test_sets.append(("(dict4 CCD)", Set(test_sets[-1][1].config, self.expected_files4,
                                             'single-CCD-catalog', 'star', 'single-field-image',
                                             'galaxy lens')))
        test_sets.append(("(dict5)", Set(stile.ConfigDataHandler({'files': self.dict5}),
                                         self.expected_files5, 'single-CCD-image', 'galaxy',
                                         'single-field-catalog', 'galaxy random')))
        # 6, 7, 8 are duplicates of earlier tests with different groupings, so already tested
        test_sets.append(("(dict9 single)", Set(stile.ConfigDataHandler({'files': self.dict9}),
                                                self.expected_files9, 'single-CCD-catalog',
                                                'galaxy', 'single-CCD-image', 'star')))
        test_sets.append(("(dict9 multiepoch)", Set(test_sets[-1][1].config, self.expected_files9,
                                                    'multiepoch-CCD-catalog', 'galaxy',
                                                    'multiepoch-CCD-image', 'star')))
        test_sets.append(("(list0)", Set(stile.ConfigDataHandler({'files': self.list0}),
                                         self.expected_files_list0, 'single-field-catalog', 'star',
                                         'single-CCD-image', 'galaxy random')))

        # Now, loop through those defined test sets, checking that we get the expected formats and 
        # objects, and NOT the formats and objects we don't expect
        for name, test_set in test_sets:
            results = test_set.config.listFileTypes()
            self.assertEqual(set(results), set(test_set.expected_files.keys()),
                msg='Failed to retrieve proper formats with listFileTypes '+name)
            results = test_set.config.listObjects(test_set.format)
            self.assertEqual(set(results), set(test_set.expected_files[test_set.format].keys()),
                msg='Failed to retrieve proper lists of objects with listObjects '+name)
            results = test_set.config.listData(test_set.object, test_set.format)
            self.assertEqual(results, test_set.expected_files[test_set.format][test_set.object],
                msg='Failed to retrieve proper list of files with listData '+name)
            results = test_set.config.listObjects(test_set.not_found_format)
            self.assertEqual(results, [],
                msg='Found a non-empty set for incorrect format in listObjects '+name)
            results = test_set.config.listData(test_set.object, test_set.not_found_format)
            self.assertEqual(results, [],
                msg='Found a non-empty set for incorrect format in listData '+name)
            results = test_set.config.listData(test_set.not_found_object, test_set.format)
            self.assertEqual(results, [],
                msg='Found a non-empty set for incorrect object in listData '+name)
        # And a quick test of multiobject "obj_type" lists
        test_set = test_sets[0][1]
        results = test_set.config.listData(['galaxy', 'star'], test_set.format)
        expected_results = zip(test_set.expected_files[test_set.format]['galaxy'],
                               test_set.expected_files[test_set.format]['star'])
        # because listData gives lists, not tuples as zip does
        expected_results = [list(e) for e in expected_results]
        # Now, because the group names are hashed and might not be in the same order, we can't test
        # for pure equality...
        # Technically, these will both pass if there are duplicate elements in 'results' replacing
        # elements which should be there, but it's hard to think of a bug that would do that
        self.assertEqual(len(results), len(expected_results))
        self.assertTrue(all([r in expected_results for r in results]))

        config = stile.ConfigDataHandler({'file': self.dict7})
        results = config.listData(['galaxy', 'star'], 'single-CCD-catalog')
        expected_results = [[{'name': 'g2.dat', 'group': 1}, {'name': 's3.dat', 'group': 1}],
                            [{'name': 'g1.dat', 'group': 2}, {'name': 's2.dat', 'group': 2}],
                            [{'name': 'g3.dat', 'group': 3}, {'name': 's1.dat', 'group': 3}]]
        self.assertEqual(len(results), len(expected_results))
        self.assertTrue(all([r in expected_results for r in results]))

        config = stile.ConfigDataHandler({'file': self.dict8})
        results = config.listData(['galaxy', 'star'], 'single-CCD-catalog')
        expected_results = [[{'name': 'g1.dat', 'file_reader': 'ASCII', 'group': '_stile_group_0'},
                             {'name': 's1.dat', 'group': '_stile_group_0'}],
                            [{'name': 'g3.dat', 'group': '_stile_group_1'},
                             {'name': 's3.dat', 'file_reader': 'ASCII', 'group': '_stile_group_1'}]]
        self.assertEqual(len(results), len(expected_results))
        self.assertTrue(all([r in expected_results for r in results]))
        
        # Make sure that a properly zipped file list is returned if binning is defined
        config = stile.ConfigDataHandler({'file': self.bins0})
        results = config.listData(['galaxy', 'star'], 'single-CCD-catalog')
        expected_results = [[{'name': 'g1.dat', 'group': '_stile_group_0', 
                              'bin_list': [stile.binning.SingleBin('ra',0,1,'name')],
                              'bins': [{'name': 'Step', 'field': 'ra', 'n_bins': 2, 'low': 0, 
                                          'high': 2}]},
                             {'name': 's1.dat', 'group': '_stile_group_0', 
                              'bin_list': [stile.binning.SingleBin('ra',0,1,'name')],
                              'bins': [{'name': 'List', 'field': 'ra', 'endpoints': [0,1,2]}]}],
                            [{'name': 'g1.dat', 'group': '_stile_group_0', 
                              'bin_list': [stile.binning.SingleBin('ra',0,1,'name')],
                              'bins': [{'name': 'Step', 'field': 'ra', 'n_bins': 2, 'low': 0, 
                                          'high': 2}]},
                             {'name': 's1.dat', 'group': '_stile_group_0', 
                              'bin_list': [stile.binning.SingleBin('ra',1,2,'name')],
                              'bins': [{'name': 'List', 'field': 'ra', 'endpoints': [0,1,2]}]}],
                            [{'name': 'g1.dat', 'group': '_stile_group_0', 
                              'bin_list': [stile.binning.SingleBin('ra',1,2,'name')],
                              'bins': [{'name': 'Step', 'field': 'ra', 'n_bins': 2, 'low': 0, 
                                          'high': 2}]},
                             {'name': 's1.dat', 'group': '_stile_group_0', 
                              'bin_list': [stile.binning.SingleBin('ra',0,1,'name')],
                              'bins': [{'name': 'List', 'field': 'ra', 'endpoints': [0,1,2]}]}],
                            [{'name': 'g1.dat', 'group': '_stile_group_0', 
                              'bin_list': [stile.binning.SingleBin('ra',1,2,'name')], 
                              'bins': [{'name': 'Step', 'field': 'ra', 'n_bins': 2, 'low': 0, 
                                          'high': 2}]}, 
                             {'name': 's1.dat', 'group': '_stile_group_0', 
                              'bin_list': [stile.binning.SingleBin('ra',1,2,'name')],
                              'bins': [{'name': 'List', 'field': 'ra', 'endpoints': [0,1,2]}]}],
                            [{'name': 'g2.dat', 'group': '_stile_group_1'},
                             {'name': 's2.dat', 'group': '_stile_group_1'}],
                            [{'name': 'g3.dat', 'group': '_stile_group_2'},
                             {'name': 's3.dat', 'group': '_stile_group_2'}]]
        self.assertEqual(len(results), len(expected_results))
        from test_binning import compare_single_bin
        for r, er in zip(results, expected_results):
            for i in range(2):
                self.assertTrue(('bin_list' in r[i] and 'bin_list' in er[i]) or 
                                ('bin_list' not in r[i] and 'bin_list' not in er[i]))
                if 'bin_list' in r[i]:
                    self.assertEqual(len(r[i]['bin_list']), len(er[i]['bin_list']))
                    self.assertTrue(all([compare_single_bin(rb,erb) 
                                        for rb, erb in zip(r[i]['bin_list'], er[i]['bin_list'])]))
                    b_l = r[i]['bin_list']
                    del r[i]['bin_list']
                    del er[i]['bin_list']
                    self.assertEqual(r[i], er[i])
                    # Not sure why we have to do this, but it seems to be necessary to pass tests
                    r[i]['bin_list'] = b_l                      
                    
    def test_systests(self):
        """
        Test the internals of the parseSysTests method.
        """
        # We will use self.dict0 (simple), self.dict3 (more complicated), self.dict9 (only 1 object
        # type per level) as our base file dicts for the sys_test cases.
        config_0 = stile.ConfigDataHandler({'files': self.dict0})
        config_3 = stile.ConfigDataHandler({'files': self.dict3})
        config_9 = stile.ConfigDataHandler({'files': self.dict9})

        results = config_0.parseSysTestsDict({'sys_tests': copy.deepcopy(self.sys_tests_0)})
        expected_results = {'single-CCD-catalog':
            [{'name': 'CorrelationFunction', 'type': 'GalaxyShear'},
             {'name': 'CorrelationFunction', 'type': 'BrightStarShear'},
             {'name': 'ScatterPlot', 'type': 'StarVsPSFG1'},
             {'name': 'Stat', 'field': 'g1', 'object_type': 'galaxy'}]}
        self.assertEqual(results, expected_results)
        results = config_0.parseSysTests({'sys_tests': copy.deepcopy(self.sys_tests_0)})
        expected_results_obj = {'single-CCD-catalog': 
            [{'sys_test': stile.GalaxyShearSysTest(), 'bin_list': [], 'extra_args': {}}, 
             {'sys_test': stile.BrightStarShearSysTest(), 'bin_list': [], 'extra_args': {}}, 
             {'sys_test': stile.ScatterPlotStarVsPSFG1SysTest(), 'bin_list': [], 'extra_args': {}}, 
             {'sys_test': stile.StatSysTest(field='g1'), 'bin_list': [], 'extra_args': {}}]}
        expected_results_obj['single-CCD-catalog'][3]['sys_test'].objects_list = ['galaxy']
        self.assertEqual(results.keys(), expected_results_obj.keys())
        self.assertTrue(all([CompareTest(r, e) for format in results for r,e in zip(results[format], expected_results_obj[format])]))
        results = config_9.parseSysTestsDict({'sys_tests': copy.deepcopy(self.sys_tests_0)})
        expected_results['multiepoch-CCD-catalog'] = []
        expected_results_obj['multiepoch-CCD-catalog'] = []
        self.assertEqual(results, expected_results)
        results = config_9.parseSysTests({'sys_tests': copy.deepcopy(self.sys_tests_0)})
        self.assertEqual(results.keys(), expected_results_obj.keys())
        self.assertTrue(all([CompareTest(r, e) for format in results for r,e in zip(results[format], expected_results_obj[format])]))
        del expected_results['multiepoch-CCD-catalog']
        del expected_results_obj['multiepoch-CCD-catalog']
        results = config_3.parseSysTestsDict({'sys_tests': copy.deepcopy(self.sys_tests_0)})
        expected_results['single-CCD-image'] = []
        expected_results['single-field-catalog'] = []
        self.assertEqual(results, expected_results)
        expected_results_obj['single-CCD-image'] = []
        expected_results_obj['single-field-catalog'] = []
        results = config_3.parseSysTests({'sys_tests': copy.deepcopy(self.sys_tests_0)})
        self.assertEqual(results.keys(), expected_results_obj.keys())
        self.assertTrue(all([CompareTest(r, e) for format in results for r,e in zip(results[format], expected_results_obj[format])]))
        
        results = config_0.parseSysTestsDict({'sys_tests': copy.deepcopy(self.sys_tests_1)})
        expected_results = {'single-CCD-catalog':
            [{'name': 'CorrelationFunction', 'type': 'GalaxyShear'},
             {'name': 'CorrelationFunction', 'type': 'BrightStarShear'},
             {'name': 'CorrelationFunction', 'type': 'StarXStarShear'},
             {'name': 'ScatterPlot', 'type': 'StarVsPSFG1'}]}
             # The last item of sys_tests_1 is an 'image' so shouldn't appear
        self.assertEqual(results, expected_results)
        expected_results_obj = {'single-CCD-catalog':
            [{'sys_test': stile.GalaxyShearSysTest(), 'bin_list': [], 'extra_args': {}}, 
             {'sys_test': stile.BrightStarShearSysTest(), 'bin_list': [], 'extra_args': {}}, 
             {'sys_test': stile.StarXStarShearSysTest(), 'bin_list': [], 'extra_args': {}}, 
             {'sys_test': stile.ScatterPlotStarVsPSFG1SysTest(), 'bin_list': [], 'extra_args': {}}]}
        results = config_0.parseSysTests({'sys_tests': copy.deepcopy(self.sys_tests_1)})
        self.assertEqual(results.keys(), expected_results_obj.keys())
        self.assertTrue(all([CompareTest(r, e) for format in results for r,e in zip(results[format], expected_results_obj[format])]))
        results = config_9.parseSysTestsDict({'sys_tests': copy.deepcopy(self.sys_tests_1)})
        expected_results['multiepoch-CCD-catalog'] = [{'name': 'CorrelationFunction',
                                                       'type': 'BrightStarShear'},
            {'name': 'CorrelationFunction', 'type': 'StarXStarShear'},
            {'name': 'ScatterPlot', 'type': 'StarVsPSFG1'}]
        self.assertEqual(results, expected_results)
        expected_results_obj['multiepoch-CCD-catalog'] = [
            {'sys_test': stile.BrightStarShearSysTest(), 'bin_list': [], 'extra_args': {}}, 
            {'sys_test': stile.StarXStarShearSysTest(), 'bin_list': [], 'extra_args': {}}, 
            {'sys_test': stile.ScatterPlotStarVsPSFG1SysTest(), 'bin_list': [], 'extra_args': {}}]
        results = config_9.parseSysTests({'sys_tests': copy.deepcopy(self.sys_tests_1)})
        self.assertEqual(results.keys(), expected_results_obj.keys())
        self.assertTrue(all([CompareTest(r, e) for format in results for r,e in zip(results[format], expected_results_obj[format])]))
        del expected_results['multiepoch-CCD-catalog']
        expected_results['single-CCD-image'] = [{'name': 'Stat', 'field': 'g1',
                                                 'object_type': 'galaxy'}]
        expected_results['single-field-catalog'] = [{'name': 'CorrelationFunction',
                                                     'type': 'GalaxyShear'},
            {'name': 'ScatterPlot', 'type': 'StarVsPSFG1'}]
        results = config_3.parseSysTestsDict({'sys_tests': copy.deepcopy(self.sys_tests_1)})
        self.assertEqual(results, expected_results)
        del expected_results_obj['multiepoch-CCD-catalog']
        expected_results_obj['single-CCD-image'] = [
            {'sys_test': stile.StatSysTest(field='g1'), 'bin_list': [], 'extra_args': {}}]
        expected_results_obj['single-CCD-image'][0]['sys_test'].objects_list = ['galaxy']
        expected_results_obj['single-field-catalog'] = [
            {'sys_test': stile.GalaxyShearSysTest(), 'bin_list': [], 'extra_args': {}},
            {'sys_test': stile.ScatterPlotStarVsPSFG1SysTest(), 'bin_list': [], 'extra_args': {}}]
        results = config_3.parseSysTests({'sys_tests': copy.deepcopy(self.sys_tests_1)})
        self.assertEqual(results.keys(), expected_results_obj.keys())
        self.assertTrue(all([CompareTest(r, e) for format in results for r,e in zip(results[format], expected_results_obj[format])]))

        results = config_0.parseSysTestsDict({'sys_test_2': copy.deepcopy(self.sys_tests_2)})
        expected_results = {'single-CCD-catalog':
            [{'name': 'CorrelationFunction', 'type': 'GalaxyShear'},
             {'name': 'CorrelationFunction', 'type': 'BrightStarShear'},
             {'name': 'ScatterPlot', 'type': 'StarVsPSFG1'},
             {'name': 'Stat', 'field': 'g1', 'object_type': 'galaxy'}]}
        self.assertEqual(results, expected_results)
        expected_results_obj = {'single-CCD-catalog':
            [{'sys_test': stile.GalaxyShearSysTest(), 'bin_list': [], 'extra_args': {}}, 
             {'sys_test': stile.BrightStarShearSysTest(), 'bin_list': [], 'extra_args': {}}, 
             {'sys_test': stile.ScatterPlotStarVsPSFG1SysTest(), 'bin_list': [], 'extra_args': {}}, 
             {'sys_test': stile.StatSysTest(field='g1'), 'bin_list': [], 'extra_args': {}}]}
        expected_results_obj['single-CCD-catalog'][3]['sys_test'].objects_list = ['galaxy']
        results = config_0.parseSysTests({'sys_tests': copy.deepcopy(self.sys_tests_2)})
        self.assertEqual(results.keys(), expected_results_obj.keys())
        self.assertTrue(all([CompareTest(r, e) for format in results for r,e in zip(results[format], expected_results_obj[format])]))
        results = config_9.parseSysTestsDict({'sys_test_2': copy.deepcopy(self.sys_tests_2)})
        expected_results['multiepoch-CCD-catalog'] = []
        self.assertEqual(results, expected_results)
        results = config_9.parseSysTests({'sys_tests': copy.deepcopy(self.sys_tests_2)})
        expected_results_obj['multiepoch-CCD-catalog'] = []
        self.assertEqual(results.keys(), expected_results_obj.keys())
        self.assertTrue(all([CompareTest(r, e) for format in results for r,e in zip(results[format], expected_results_obj[format])]))
        del expected_results['multiepoch-CCD-catalog']
        results = config_3.parseSysTestsDict({'sys_test_2': copy.deepcopy(self.sys_tests_2)})
        expected_results['single-CCD-image'] = []
        expected_results['single-field-catalog'] = []
        self.assertEqual(results, expected_results)
        del expected_results_obj['multiepoch-CCD-catalog']
        expected_results_obj['single-CCD-image'] = []
        expected_results_obj['single-field-catalog'] = []
        results = config_3.parseSysTests({'sys_tests': copy.deepcopy(self.sys_tests_2)})
        self.assertEqual(results.keys(), expected_results_obj.keys())
        self.assertTrue(all([CompareTest(r, e) for format in results for r,e in zip(results[format], expected_results_obj[format])]))

        results = config_0.parseSysTestsDict({'sys_test_1': copy.deepcopy(self.sys_tests_3a),
            'sys_test_2': copy.deepcopy(self.sys_tests_3b),
            'sys_test_3': copy.deepcopy(self.sys_tests_3c)})
        expected_results = {'single-CCD-catalog':
            [{'name': 'ScatterPlot', 'type': 'StarVsPSFG1'},
             {'name': 'CorrelationFunction', 'type': 'BrightStarShear'},
             {'name': 'CorrelationFunction', 'type': 'StarXStarShear'},
             {'name': 'CorrelationFunction', 'type': 'GalaxyShear'}]}
        self.assertEqual(results, expected_results)
        expected_results_obj = {'single-CCD-catalog':
            [{'sys_test': stile.ScatterPlotStarVsPSFG1SysTest(), 'bin_list': [], 'extra_args': {}}, 
             {'sys_test': stile.BrightStarShearSysTest(), 'bin_list': [], 'extra_args': {}}, 
             {'sys_test': stile.StarXStarShearSysTest(), 'bin_list': [], 'extra_args': {}}, 
             {'sys_test': stile.GalaxyShearSysTest(), 'bin_list': [], 'extra_args': {}}]} 
        results = config_0.parseSysTests({'sys_test_1': copy.deepcopy(self.sys_tests_3a),
            'sys_test_2': copy.deepcopy(self.sys_tests_3b),
            'sys_test_3': copy.deepcopy(self.sys_tests_3c)})
        self.assertEqual(results.keys(), expected_results_obj.keys())
        self.assertTrue(all([CompareTest(r, e) for format in results for r,e in zip(results[format], expected_results_obj[format])]))
        results = config_9.parseSysTestsDict({'sys_test_1': copy.deepcopy(self.sys_tests_3a),
            'sys_test_2': copy.deepcopy(self.sys_tests_3b),
            'sys_test_3': copy.deepcopy(self.sys_tests_3c)})
        expected_results['multiepoch-CCD-catalog'] = [{'name': 'ScatterPlot',
                                                       'type': 'StarVsPSFG1'},
            {'name': 'CorrelationFunction', 'type': 'BrightStarShear'},
            {'name': 'CorrelationFunction', 'type': 'StarXStarShear'}]
        self.assertEqual(results, expected_results)
        expected_results_obj['multiepoch-CCD-catalog'] = [
            {'sys_test': stile.ScatterPlotStarVsPSFG1SysTest(), 'bin_list': [], 'extra_args': {}}, 
            {'sys_test': stile.BrightStarShearSysTest(), 'bin_list': [], 'extra_args': {}}, 
            {'sys_test': stile.StarXStarShearSysTest(), 'bin_list': [], 'extra_args': {}}]
        results = config_9.parseSysTests({'sys_test_1': copy.deepcopy(self.sys_tests_3a),
            'sys_test_2': copy.deepcopy(self.sys_tests_3b),
            'sys_test_3': copy.deepcopy(self.sys_tests_3c)})
        self.assertEqual(results.keys(), expected_results_obj.keys())
        self.assertTrue(all([CompareTest(r, e) for format in results for r,e in zip(results[format], expected_results_obj[format])]))
        del expected_results['multiepoch-CCD-catalog']
        results = config_3.parseSysTestsDict({'sys_test_1': copy.deepcopy(self.sys_tests_3a),
            'sys_test_2': copy.deepcopy(self.sys_tests_3b),
            'sys_test_3': copy.deepcopy(self.sys_tests_3c)})
        expected_results['single-CCD-image'] = [{'name': 'Stat', 'field': 'g1',
                                                 'object_type': 'galaxy'}]
        expected_results['single-field-catalog'] = [{'name': 'ScatterPlot',
                                                     'type': 'StarVsPSFG1'},
            {'name': 'CorrelationFunction', 'type': 'GalaxyShear'}]
        self.assertEqual(results, expected_results)
        del expected_results_obj['multiepoch-CCD-catalog']
        expected_results_obj['single-CCD-image'] = [
            {'sys_test': stile.StatSysTest(field='g1'), 'bin_list': [], 'extra_args': {}}]
        expected_results_obj['single-CCD-image'][0]['sys_test'].objects_list = ['galaxy']
        expected_results_obj['single-field-catalog'] = [
            {'sys_test': stile.GalaxyShearSysTest(), 'bin_list': [], 'extra_args': {}},
            {'sys_test': stile.ScatterPlotStarVsPSFG1SysTest(), 'bin_list': [], 'extra_args': {}}]
        results = config_3.parseSysTests({'sys_test_1': copy.deepcopy(self.sys_tests_3a),
            'sys_test_2': copy.deepcopy(self.sys_tests_3b),
            'sys_test_3': copy.deepcopy(self.sys_tests_3c)})
        self.assertEqual(results.keys(), expected_results_obj.keys())
        self.assertTrue(all([CompareTest(r, e) for format in results for r,e in zip(results[format], expected_results_obj[format])]))
        

        results = config_0.parseSysTestsDict({'sys_test': copy.deepcopy(self.sys_tests_4)})
        expected_results = {'single-CCD-catalog':
            [{'name': 'CorrelationFunction', 'type': 'GalaxyShear'},
             {'name': 'CorrelationFunction', 'type': 'BrightStarShear'}]}
        self.assertEqual(results, expected_results)
        expected_results_obj = {'single-CCD-catalog':
            [{'sys_test': stile.GalaxyShearSysTest(), 'bin_list': [], 'extra_args': {}}, 
             {'sys_test': stile.BrightStarShearSysTest(), 'bin_list': [], 'extra_args': {}}]}
        results = config_0.parseSysTests({'sys_test': copy.deepcopy(self.sys_tests_4)})
        self.assertEqual(results.keys(), expected_results_obj.keys())
        self.assertTrue(all([CompareTest(r, e) for format in results for r,e in zip(results[format], expected_results_obj[format])]))
        results = config_9.parseSysTestsDict({'sys_test': copy.deepcopy(self.sys_tests_4)})
        expected_results['multiepoch-CCD-catalog'] = []
        self.assertEqual(results, expected_results)
        expected_results_obj['multiepoch-CCD-catalog'] = []
        results = config_9.parseSysTests({'sys_test': copy.deepcopy(self.sys_tests_4)})
        self.assertEqual(results.keys(), expected_results_obj.keys())
        self.assertTrue(all([CompareTest(r, e) for format in results for r,e in zip(results[format], expected_results_obj[format])]))
        del expected_results['multiepoch-CCD-catalog']
        results = config_3.parseSysTestsDict({'sys_test': copy.deepcopy(self.sys_tests_4)})
        expected_results['single-CCD-image'] = [{'name': 'ScatterPlot',
                                                 'type': 'StarVsPSFG1'},
                            {'name': 'Stat', 'field': 'g1', 'object_type': 'galaxy'}]
        expected_results['single-field-catalog'] = []
        self.assertEqual(results, expected_results)
        del expected_results_obj['multiepoch-CCD-catalog']
        results = config_3.parseSysTests({'sys_test': copy.deepcopy(self.sys_tests_4)})
        expected_results['single-CCD-image'] = [{'name': 'ScatterPlot',
                                                 'type': 'StarVsPSFG1'},
                            {'name': 'Stat', 'field': 'g1', 'object_type': 'galaxy'}]
        expected_results_obj['single-CCD-image'] = [
            {'sys_test': stile.ScatterPlotStarVsPSFG1SysTest(), 'bin_list': [], 'extra_args': {}},
            {'sys_test': stile.StatSysTest(field='g1'), 'bin_list': [], 'extra_args': {}}]
        expected_results_obj['single-CCD-image'][1]['sys_test'].objects_list = ['galaxy']
        expected_results_obj['single-field-catalog'] = []
        self.assertEqual(results.keys(), expected_results_obj.keys())
        self.assertTrue(all([CompareTest(r, e) for format in results for r,e in zip(results[format], expected_results_obj[format])]))
        
        results = config_0.parseSysTestsDict({'sys_test': copy.deepcopy(self.sys_tests_5)})
        expected_results = {'single-CCD-catalog': []}
        self.assertEqual(results, expected_results)
        expected_results_obj = {'single-CCD-catalog': []}
        results = config_0.parseSysTests({'sys_test': copy.deepcopy(self.sys_tests_5)})
        self.assertEqual(results.keys(), expected_results_obj.keys())
        self.assertTrue(all([CompareTest(r, e) for format in results for r,e in zip(results[format], expected_results_obj[format])]))
        
        results = config_9.parseSysTestsDict({'sys_test': copy.deepcopy(self.sys_tests_5)})
        expected_results['multiepoch-CCD-catalog'] = [{'name': 'CorrelationFunction',
                                                       'type': 'GalaxyShear'},
                            {'name': 'CorrelationFunction', 'type': 'BrightStarShear'}]
        self.assertEqual(results, expected_results)
        expected_results_obj['multiepoch-CCD-catalog'] = [
             {'sys_test': stile.GalaxyShearSysTest(), 'bin_list': [], 'extra_args': {}}, 
             {'sys_test': stile.BrightStarShearSysTest(), 'bin_list': [], 'extra_args': {}}]
        expected_results_obj['single-CCD-catalog'] = [
             {'sys_test': stile.GalaxyShearSysTest(), 'bin_list': [], 'extra_args': {}}, 
             {'sys_test': stile.BrightStarShearSysTest(), 'bin_list': [], 'extra_args': {}}]
        results = config_9.parseSysTests({'sys_test': copy.deepcopy(self.sys_tests_5)})
        self.assertEqual(results.keys(), expected_results_obj.keys())
        self.assertTrue(all([CompareTest(r, e) for format in results for r,e in zip(results[format], expected_results_obj[format])]))
        del expected_results['multiepoch-CCD-catalog']
        results = config_3.parseSysTestsDict({'sys_test': copy.deepcopy(self.sys_tests_5)})
        expected_results['single-CCD-image'] = []
        expected_results['single-field-catalog'] = []
        self.assertEqual(results, expected_results)
        del expected_results_obj['multiepoch-CCD-catalog']
        expected_results_obj['single-CCD-image'] = []
        expected_results_obj['single-field-catalog'] = []
        results = config_3.parseSysTests({'sys_test': copy.deepcopy(self.sys_tests_5)})
        self.assertEqual(results.keys(), expected_results_obj.keys())
        self.assertTrue(all([CompareTest(r, e) for format in results for r,e in zip(results[format], expected_results_obj[format])]))

    def test_make_bins_and_tests(self):
        """
        Test the makeBins and makeTest methods for various input dicts.
        """
        bin1 = {'name': 'Step', 'field': 'ra', 'low': 0, 'high': 7.5, 'n_bins': 8}
        expected_bin1 = stile.BinStep(field='ra', low=0, high=7.5, n_bins=8)
        bin2 = {'name': 'Step', 'field': 'dec', 'high': 10, 'n_bins': 13, 'step': 0.5, 'use_log': True}
        expected_bin2 = stile.BinStep(field='dec', high=10, n_bins=13, step=0.5, use_log=True)
        bin3 = {'name': 'Step', 'field': 'g1', 'high': -2, 'n_bins': 5}  # should fail
        bin4 = {'name': 'List', 'field': '?!', 'endpoints': [0,1,3,5,10]}
        expected_bin4 = stile.BinList(bin_list=[0,1,3,5,10], field='?!')
        bin5 = {'name': 'List', 'field': 'g2', 'endpoints': [5,7,-1,8,10]}  # should fail
        bin6 = {'name': 'List', 'endpoints': [5,7,-1,8,10]}  # should fail
        
        bin_obj = self.testConfigDataHandler.makeBins(bin1)
        self.assertEqual(len(bin_obj), 1)
        bin_obj = bin_obj[0]
        self.assertEqual(bin_obj, expected_bin1)
        bin_obj = self.testConfigDataHandler.makeBins(bin2)
        self.assertEqual(len(bin_obj), 1)
        bin_obj = bin_obj[0]
        self.assertEqual(bin_obj, expected_bin2)
        self.assertRaises(TypeError, self.testConfigDataHandler.makeBins, bin3)
        bin_obj = self.testConfigDataHandler.makeBins(bin4)
        self.assertEqual(len(bin_obj), 1)
        bin_obj = bin_obj[0]
        self.assertEqual(bin_obj, expected_bin4)
        self.assertRaises(ValueError, self.testConfigDataHandler.makeBins, bin5)
        self.assertRaises(ValueError, self.testConfigDataHandler.makeBins, bin6)
        
    
        test1 = {'name': 'CorrelationFunction', 'type': 'GalaxyShear'}
        test2 = {'name': 'CorrelationFunction', 'type': 'BrightStarShear', 'extra_args': {'ra': 7}}  # extra args
        test3 = {'name': 'CorrelationFunction', 'type': 'GalaxyDensity', 'bins': bin1}  # with bins
        test4 = {'name': 'CorrelationFunction', 'type': 'StarDensity',  #both
                 'extra_args': {'random keyword': 'random argument'}, 'bins': bin4}
        test5 = {'name': 'CorrelationFunction', 'type': 'PlanetDensity'}  # not a real type
        test6 = {'name': 'CorrelationFunction'}  # missing a type
        test7 = {'name': 'ScatterPlot', 'type': 'ResidualVsPSFG2'}
        test8 = {'name': 'ScatterPlot', 'type': 'StarVsPSFSigma', 'extra_args': {'ra': 7}}  
        test9 = {'name': 'ScatterPlot', 'type': 'StarVsPSFG1', 'bins': bin2}
        test10 = {'name': 'ScatterPlot', 'type': 'ResidualVsPSFSigma', 
                 'extra_args': {'random keyword': 'random argument'}, 'bins': bin1}
        test11 = {'name': 'ScatterPlot', 'type': 'StarVsResidualG2'}
        test12 = {'name': 'ScatterPlot'}
        test13 = {'name': 'WhiskerPlot', 'type': 'Residual'}
        test14 = {'name': 'WhiskerPlot', 'type': 'Star', 'extra_args': {'ra': 7}}  # xtra arg
        test15 = {'name': 'WhiskerPlot', 'type': 'PSF', 'bins': bin4}
        test16 = {'name': 'WhiskerPlot', 'type': 'PSF', 
                 'extra_args': {'random keyword': 'random argument'}, 'bins': bin2}
        test17 = {'name': 'WhiskerPlot', 'type': 'Planet'}
        test18 = {'name': 'WhiskerPlot'}
        test19 = {'name': 'CorrelationFunction', 'type': 'GalaxyShear', 'random keyword': 'random argument'}
        test20 = {'name': 'ScatterPlot', 'type': 'StarVsPSFG1', 'random keyword': 'random argument'}
        test21 = {'name': 'WhiskerPlot', 'type': 'Star', 'random keyword': 'random argument'}

        test_obj = self.testConfigDataHandler.makeTest(test1)
        self.assertIsInstance(test_obj['sys_test'], stile.GalaxyShearSysTest)
        self.assertEqual(test_obj['bin_list'], [])
        self.assertEqual(test_obj['extra_args'], {})
        test_obj = self.testConfigDataHandler.makeTest(test2)
        self.assertIsInstance(test_obj['sys_test'], stile.BrightStarShearSysTest)
        self.assertEqual(test_obj['bin_list'], [])
        self.assertEqual(test_obj['extra_args'], {'ra': 7})
        test_obj = self.testConfigDataHandler.makeTest(test3)
        self.assertIsInstance(test_obj['sys_test'], stile.sys_tests.GalaxyDensitySysTest)
        self.assertEqual(test_obj['bin_list'], [expected_bin1])
        self.assertEqual(test_obj['extra_args'], {})
        test_obj = self.testConfigDataHandler.makeTest(test4)
        self.assertIsInstance(test_obj['sys_test'], stile.sys_tests.StarDensitySysTest)
        self.assertEqual(test_obj['bin_list'], [expected_bin4])
        self.assertEqual(test_obj['extra_args'], {'random keyword': 'random argument'})
        test_obj = self.testConfigDataHandler.makeTest(test7)
        self.assertIsInstance(test_obj['sys_test'], stile.sys_tests.ScatterPlotResidualVsPSFG2SysTest)
        self.assertEqual(test_obj['bin_list'], [])
        self.assertEqual(test_obj['extra_args'], {})
        test_obj = self.testConfigDataHandler.makeTest(test8)
        self.assertIsInstance(test_obj['sys_test'], stile.sys_tests.ScatterPlotStarVsPSFSigmaSysTest)
        self.assertEqual(test_obj['bin_list'], [])
        self.assertEqual(test_obj['extra_args'], {'ra': 7})
        test_obj = self.testConfigDataHandler.makeTest(test9)
        self.assertIsInstance(test_obj['sys_test'], stile.sys_tests.ScatterPlotStarVsPSFG1SysTest)
        self.assertEqual(test_obj['bin_list'], [expected_bin2])
        self.assertEqual(test_obj['extra_args'], {})
        test_obj = self.testConfigDataHandler.makeTest(test10)
        self.assertIsInstance(test_obj['sys_test'], stile.sys_tests.ScatterPlotResidualVsPSFSigmaSysTest)
        self.assertEqual(test_obj['bin_list'], [expected_bin1])
        self.assertEqual(test_obj['extra_args'], {'random keyword': 'random argument'})
        test_obj = self.testConfigDataHandler.makeTest(test13)
        self.assertIsInstance(test_obj['sys_test'], stile.sys_tests.WhiskerPlotResidualSysTest)
        self.assertEqual(test_obj['bin_list'], [])
        self.assertEqual(test_obj['extra_args'], {})
        test_obj = self.testConfigDataHandler.makeTest(test14)
        self.assertIsInstance(test_obj['sys_test'], stile.sys_tests.WhiskerPlotStarSysTest)
        self.assertEqual(test_obj['bin_list'], [])
        self.assertEqual(test_obj['extra_args'], {'ra': 7})
        test_obj = self.testConfigDataHandler.makeTest(test15)
        self.assertIsInstance(test_obj['sys_test'], stile.sys_tests.WhiskerPlotPSFSysTest)
        self.assertEqual(test_obj['bin_list'], [expected_bin4])
        self.assertEqual(test_obj['extra_args'], {})
        test_obj = self.testConfigDataHandler.makeTest(test16)
        self.assertIsInstance(test_obj['sys_test'], stile.sys_tests.WhiskerPlotPSFSysTest)
        self.assertEqual(test_obj['bin_list'], [expected_bin2])
        self.assertEqual(test_obj['extra_args'], {'random keyword': 'random argument'})
        self.assertRaises(AttributeError, self.testConfigDataHandler.makeTest, test5)
        self.assertRaises(AttributeError, self.testConfigDataHandler.makeTest, test11)
        self.assertRaises(AttributeError, self.testConfigDataHandler.makeTest, test17)
        self.assertRaises(ValueError, self.testConfigDataHandler.makeTest, test6)
        self.assertRaises(ValueError, self.testConfigDataHandler.makeTest, test12)
        self.assertRaises(ValueError, self.testConfigDataHandler.makeTest, test18)
        self.assertRaises(ValueError, self.testConfigDataHandler.makeTest, test19)
        self.assertRaises(ValueError, self.testConfigDataHandler.makeTest, test20)
        self.assertRaises(ValueError, self.testConfigDataHandler.makeTest, test21)
        
    def test_errors(self):
        """
        Make sure that various errors are raised, if not already tested in previous methods.
        """
        bad_test = {'files': [{'epoch': 'coadd', 'extent': 'field', 'data_format': 'catalog', 
                               'name': 's1.dat', 'object_type': 'star'}],
                    'sys_tests': [{'name': 'CorrelationFunction', 'type': 'BrightStarShear'},
                                  'CorrelationFunction']}
        self.assertRaises(ValueError, stile.ConfigDataHandler, bad_test)
        
        bad_test['sys_tests'] = 'CorrelationFunction'
        self.assertRaises(ValueError, stile.ConfigDataHandler, bad_test)
        
        bad_files = {'files': 's1.dat'}
        self.assertRaises(ValueError, stile.ConfigDataHandler, bad_files)
        
        bad_files = {'files': [{'epoch': 'coadd', 'extent': 'field', 'data_format': 'catalog', 
                                'name': 's1.dat', 'object_type': 'star'},
                               'g1.dat']}
        self.assertRaises(ValueError, stile.ConfigDataHandler, bad_files)
        
        bad_formats = {'files': [{'epoch': 'coadd', 'extent': 'field', 'data_format': 'catalog', 
                                'name': 's1.dat'}]}
        self.assertRaises(ValueError, stile.ConfigDataHandler, bad_formats)
        bad_formats = {'files': {'epoch': 'coadd', 'extent': 'field', 'data_format': 'catalog', 
                                'name': 's1.dat'}}
        self.assertRaises(ValueError, stile.ConfigDataHandler, bad_formats)
        bad_formats = {'files': {'coadd': {'field': {'catalog': 's1.dat'}}}}
        self.assertRaises(ValueError, stile.ConfigDataHandler, bad_formats)
        duplicate_formats = {'files': {'coadd': {'field': {'catalog': {'name': 's1.dat', 
                             'object_type': 'star', 'epoch': 'coadd'}}}}}
        self.assertRaises(ValueError, stile.ConfigDataHandler, duplicate_formats)
                                
        bad_multiepoch = {'files': [{'epoch': 'multiepoch', 'extent': 'field', 
                                     'data_format': 'catalog', 'name': 's1.dat'}]}
        self.assertRaises(ValueError, stile.ConfigDataHandler, bad_multiepoch)
        bad_multiepoch = {'files': {'multiepoch': {'field': {'catalog': {'star': 
                                [['s1.dat', 's2.dat'], ['s3.dat', 's4.dat'], 's5.dat']}}}}}
        self.assertRaises(ValueError, stile.ConfigDataHandler, bad_multiepoch)
        
        extra_kwarg = {'files': {'coadd': {'field': {'catalog': {'name': 's1.dat', 
                             'object_type': 'star'}}, 'hello': 'goodbye'}}}
        self.assertRaises(ValueError, stile.ConfigDataHandler, extra_kwarg)

        bad_groups = {'files': {'coadd': {'field': {'catalog': {'star': 
                                                                {'name': 's1.dat', 'group': 1}}}},
                                          'CCD': {'catalog': {'galaxy': 
                                                                {'name': 'g1.dat', 'group': 1}}}}}
        self.assertRaises(ValueError, stile.ConfigDataHandler, bad_groups)

if __name__=='__main__':
    unittest.main()
