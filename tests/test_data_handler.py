try:
    import stile
except ImportError:
    import sys
    sys.path.append('..')
    import stile
from stile import stile_utils
import unittest
import copy

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
#        expected_results = {stile_utils.Format(epoch='single',extent='CCD',data_format='catalog').str: {
#                                'galaxy': [{'name': 'g1-0.dat', 'file_reader': 'ASCII'},
#                                           {'name': 'g1-1.dat', 'file_reader': 'ASCII'},
#                                           {'name': 'g2-0.dat'},
#                                           {'name': 'g2-1.dat'},
#                                           {'name': 'g3-0.dat'},
#                                           {'name': 'g3-1.dat'}]},
#                            stile_utils.Format(epoch='multiepoch',extent='CCD',data_format='catalog').str: {
#                                'galaxy': [{'name': ['g1-0.dat','g1-1.dat'], 'file_reader': 'ASCII'},
#                                           {'name': ['g2-0.dat','g2-1.dat']},
#                                           {'name': ['g3-0.dat','g3-1.dat']}]} 
#                           }

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

        
if __name__=='__main__':
    unittest.main()
