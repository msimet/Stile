from . import sys_tests
import binning
import file_io
import numpy
import stile_utils
import pprint

class ConfigDriver(object):
    """
    An object which can be called with a stile.ConfigDataHandler instance to run a set of tests
    on a set of data as defined by a config file.
    """
    file_indices = []

    def getName(self, files):
        if isinstance(files,(list,tuple)):
            return '_'.join([self.getName(f) for f in files])
        if isinstance(files, dict):
            if 'nickname' in files:
                return files['nickname']
            elif 'name' in files:
                if 'bin_list' in files:
                    return '_'.join([files['name']]+[bin.short_name for bin in files['bin_list']])
                else:
                    return files['name']
            else:
                if files in self.file_indices:
                    return str(self.file_indices.index(files))
                else:
                    self.file_indices.append(files)
                    return len(file_indices)-1
    
    def _runAllTests(self, config, data, index, data_list, undone_files, waiting_list, single_test, group_test):
        """
        A helper function to run both single-dataset tests and group-dataset tests without reading 
        in datasets multiple times.

        Note that this function relies heavily on the fact that dicts and lists are implicitly
        passed by reference.  DO NOT REASSIGN ANY VARIABLE PASSED TO THIS FUNCTION, you might
        screw up the bookkeeping that's implicitly taking place between recursive calls!
        """
        # Do all the single-dataset tests for this data array
        if index in single_test:
            self.RunSysTests(config, data, single_test[index], self.getName(data_list[index][0]))
        if index in group_test:
            # Now, loop through all the multi-dataset tests this data array is a part of.  Do any
            # that haven't already been done, read in any files that haven't been read yet, and
            # perform the necessary tests.
            for group_dict in group_test[index]:
                # If this next line is false, we must have already done this group test (because one
                # of the indices requested is neither in undone_files nor waiting_list, which means
                # it's been deleted or popped from both places, so all its tests are complete), so
                # we can skip it.  Otherwise, do the test.
                if all([i in undone_files or i in waiting_list or i==index 
                        for i in group_dict['indices']]):
                    for i in group_dict['indices']:
                        if i in waiting_list:
                            pass
                        elif not i==index:
                            undone_files.remove(i)
                            new_data = config.getData(data_list[i][0], data_list[i][1], data_list[i][2])
                            waiting_list[i] = new_data
                        else:
                            waiting_list[index] = data
                    self.RunMultiSysTests(config,
                                          [waiting_list[i] for i in group_dict['indices']],
                                          group_dict['sys_tests'], self.getName([data_list[i][0] for i in group_dict['indices']]))
        # Delete this file from waiting_list, so we don't do any tests involving it again.
        if index in waiting_list:
            del waiting_list[index]
        # Take care of any other files we've read in while doing multidataset tests with this one.
        while waiting_list:
            new_index = waiting_list.keys()[0]
            self._runAllTests(config, waiting_list[new_index], new_index, data_list, undone_files,
                              waiting_list, single_test, group_test)

    def _runSysTestHelper(self, config, data, sys_test, name):
        # Run a test and save or plot the results
        results = sys_test['sys_test'](data, **sys_test['extra_args'])
        if isinstance(results, numpy.ndarray):
            file_io.WriteASCIITable(config.getOutputPath(sys_test['sys_test'].short_name, name, 
                                                   '.txt'), results, print_header=True)
        elif isinstance(results, stile_utils.Stats):
            with open(config.getOutputPath([sys_test['sys_test'].short_name, name], 
                                           '.txt'), 'w') as f:
                f.write(str(results)+'\n')
        plot_results = sys_test['sys_test'].plot(results)
        plot_results.savefig(config.getOutputPath(sys_test['sys_test'].short_name, name, 
                                                  '.png'))

    def RunSysTests(self, config, data, sys_test_list, name):
        """
        Run a set of SysTests that require only one data file.

        @param config         A stile.ConfigDataHandler instance
        @param data           A data array
        @param sys_test_list  A list of dicts describing tests to be performed on the data array,
                              as output by ConfigDriver.makeTest()
        @param name           A string to be used in the output filenames, denoting this dataset
        """
        # Loop through the tests in the test_list, binning if necessary
        for sys_test_dict in sys_test_list:
            if sys_test_dict['bin_list']:
                bin_items = binning.ExpandBinList(sys_test_dict['bin_list'])
                for bin_scheme in bin_items:
                    bin_data = data
                    new_name = name
                    for bin in bin_scheme:
                        bin_data = bin(bin_data)
                        new_name = new_name+'_'+bin.short_name
                    self._runSysTestHelper(config, bin_data, sys_test_dict, name)
            else:
                self._runSysTestHelper(config, data, sys_test_dict, name)

    def _runMultiSysTestHelper(self, config, data, sys_test, name):
        # Run a test given a list of data arrays, then save or plot the results
        results = sys_test['sys_test'](*data, **sys_test['extra_args'])
        if isinstance(results, numpy.ndarray):
            file_io.WriteASCIITable(config.getOutputPath(sys_test['sys_test'].short_name, name,
                                                '.txt'), results, print_header=True)
        elif isinstance(results, str):
            with open(config.getOutputPath(sys_test['sys_test'].short_name, name, 
                                           '.txt')) as f:
                f.write(results+'\n')
        plot_results = sys_test['sys_test'].plot(results)
        plot_results.savefig(config.getOutputPath(sys_test['sys_test'].short_name, name,
                                                  '.png'))

    def RunMultiSysTests(self, config, data, sys_test_list, name):
        """
        Run a set of SysTests that require only one data file.

        @param config         A stile.ConfigDataHandler instance
        @param data           A list of data arrays
        @param sys_test_list  A list of dicts describing tests to be performed on the data arrays,
                              as output by ConfigDriver.makeTest()
        @param name           A string to be used in the output filenames, denoting this dataset
        """
        for sys_test_dict in sys_test_list:
            if sys_test_dict['bin_list']:
                # Right now there is no way to specify which dataset to apply the binning to
                raise NotImplementedError()
            else:
                self._runMultiSysTestHelper(config, data, sys_test_dict, name)

    def __call__(self, config):
        """
        Given a stile.ConfigDataHandler instance, determine which tests can be run given the data
        we have, run those tests, and write the outputs to a file.
        """
        if not hasattr(config, 'files') or not hasattr(config, 'sys_tests'):
            raise RuntimeError('Argument passed to ConfigDriver() should be a '+
                               'stile.ConfigDataHandler')

        # Generate a dict that contains information about the object_types needed by the sys_tests.
        format_obj_dict = {}
        for format in config.sys_tests:
            format_obj_dict[format] = {}
            for sys_test in config.sys_tests[format]:
                obj = sys_test['sys_test'].objects_list
                if len(obj)==1:
                    obj = obj[0]
                else:
                    obj = tuple(obj)  # so we can use it as a dictionary key
                if obj not in format_obj_dict[format]:
                    format_obj_dict[format][obj] = [sys_test]
                else:
                    format_obj_dict[format][obj].append(sys_test)

        # For most of this processing, the format-keyed form is the simplest thing to do.  But, once
        # we're actually reading in data, we'd rather not read it in multiple times.  So, now we
        # move to a form where we unify the list of data files and work out which of the tests apply
        # to each file.  Tests requiring one data file and tests requiring multiple data files are
        # tracked separately.
        data_list = []
        single_test = {}
        group_test = {}

        for format in format_obj_dict:
            for obj_type in format_obj_dict[format]:
                data = config.listData(obj_type, format)
                for item in data:
                    if hasattr(item, '__iter__') and not isinstance(item, dict) and len(item)>1:
                        # This is a set of data files to be analyzed together.  We will save the
                        # files themselves, and put a tuple of the indices of those files in the
                        # 'data_list' list into the group_test dict along with the tests to be
                        # performed on them.
                        indices = []
                        for indx, i in enumerate(item):
                            if (i,obj_type[indx],format) not in data_list:
                                data_list.append((i,obj_type[indx],format))
                                indices.append(len(data_list)-1)
                            else:
                                index = data_list.index((i,obj_type[indx],format))
                                indices.append(index)
                        for index in indices:
                            if index not in group_test:
                                group_test[index] = []
                            group_test[index].append({'sys_tests': 
                                format_obj_dict[format][obj_type], 'indices': tuple(indices)})
                    else:
                        if hasattr(item,'__len__') and not isinstance(item, dict):
                            item = item[0]
                        if hasattr(obj_type, '__iter__'):
                            obj_type = obj_type[0]
                        # This is a single file. We will save the info for the files themselves and
                        # also save the file index and the systematics tests to be done
                        # in single_test.
                        if (item,obj_type,format) not in data_list:
                            data_list.append((item,obj_type,format))
                            index = len(data_list)-1
                        else:
                            index = data_list.index((item,obj_type,format))
                        if index not in single_test:
                            single_test[index] = []
                        single_test[index].extend(format_obj_dict[format][obj_type])

        # Now we need to loop through the files we've found and do the requested sys_tests.  If
        # memory isn't a concern, then we'd like to save file I/O by only reading in the
        # files once.  We start out by making a list of "undone files"--the indices of files that
        # haven't been read in yet, ie all of them--then read in that data, set up a "waiting list"
        # queue of files which have been read in but not had all their tests completed yet,
        # and then call a subroutine to run the tests.
        if not config.stile_args.get('save_memory', False):
            undone_files = [i for i in range(len(data_list))]
            while undone_files:
                index = undone_files.pop(0)
                data = config.getData(data_list[index][0], data_list[index][1], data_list[index][2])
                waiting_list = {}

                self._runAllTests(config, data, index, data_list, undone_files, waiting_list,
                                  single_test, group_test)
        else:
            # If we fell through to this clause, then memory is a bit more precious and we'd rather
            # reread a few files than potentially keep a bunch of them in memory.  So, loop through
            # the single-dataset tests and do those, then loop through the multi-dataset tests and
            # do the ones where the FIRST index is the same as this index.  That way, we don't redo
            # any groups, and we can at least reuse the data we've read in for the single test.
            for index in single_test:
                data = config.getData(data_list[index][0], data_list[index][1], data_list[index][2])
                self.RunSysTests(config, data, single_test[index], self.getName(data_list[index][0]))
                if index in group_test:
                    for group_dict in group_test[index]:
                        if group_dict['indices'][0]==index:
                            multi_data = [data] + [config.getData(data_list[i][0], data_list[i][1], data_list[i][2]) 
                                                   for i in group_dict[indices][1:]]
                            self.runMultiSysTests(config, multi_data, 
                                                  group_dict['sys_tests'], self.getName(data_list[i][0]))
            # Now, do any files that *only* appear in groups
            group_only = group_test.viewkeys() - single_test.viewkeys()
            for index in group_only:
                for group_dict in group_test[index]:
                    if group_dict['indices'][0]==index:
                        multi_data = [data] + [config.getData(data_list[i][0], data_list[i][1], data_list[i][2]) 
                                               for i in group_dict[indices][1:]]
                        self.runMultiSysTests(config, multi_data, group_dict['sys_tests'], self.getName(data_list[i][0]))
        
        if self.file_indices:
            print "Gave index numbers to various files (for output filenames) as follows..."
            for i, file in enumerate(self.file_indices):
                print str(i)+':', file
