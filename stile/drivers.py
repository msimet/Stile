import stile

def ChooseIDs(stile_args):
    """
    Given a set of stile parameters (stile_args), choose the data IDs to run the various tests on.
    If there is an explicit set of IDs, use that; if every test has a list of IDs, use those; 
    otherwise, figure out which data formats are required for the requested tests, then call
    data_handler.listData() with those formats, and take the union of those IDs with any additional
    IDs from the requested tests, if they exist.
    
    @param stile_args  A dict of Stile options, generally created by stile.ParseArgs().
    @returns A list of data IDs
    """
    if ids in stile_args:
        return ids
    elif 'tests' in stile_args and all(['id' in test for test in stile_args['tests']]):
        id_list = [sublist for test in stile_args['tests'] for sublist in test['id']]
    else:
        test_options = chooseTestOptions(stile_args,None)
        formats = [to[0] for to in test_options]
        id_list = []
        for format in formats:
            id_list += stile_args.data_handler.listData(**format)
        if 'tests' in stile_args and any(['id' in test for test in stile_args['tests']):
            id_list += [sublist for test in stile_args['tests'] for sublist in test.get('id',[])]
    return list(set(id_list))

def _GetRequiredFields(test):
    """
    For a given catalog-type test, return a list of required fields, if it is defined by 
    test.required_fields; else return an empty list.
    """
    if hasattr(test,'required_fields'):
        req = test.required_fields
        if isinstance(req,str):
            return [req]
        else:
            return req
    else:
        return []
        
def _AddToFormatDict(format_dict, test, pairs=stile_utils.paired, epochs=stile_utils.epochs,
                                        extents=stile_utils.extents, formats=stile_utils.formats,
                                        bins=None):
    """
    Given a format dictionary of the kind used by ChooseTestOptions(), that is, a nested dict
    with: 
        format_dict[pair][epoch][extent][format]['test'] = list of tests,
        format_dict[pair][epoch][extent][format]['req'] = list of required fields,
        format_dict[pair][epoch][extent][format]['bins'] = list of bins,
    with each element of those lists corresponding to each other, add a test and its bins
    and required fields.  This will loop over all types of (pair/not-pair), epoch, extent, and
    format (catalog/image), unless the kwargs pairs, epochs, extents, or formats are set.
    
    @param format_dict A dict as described above
    @param test        A stile.SysTest
    @param pairs       A list containing one or more of ['pair','single'] (default: all)
    @param epochs      A list containing one or more of ['single','multiepoch'] (default: all)
    @param extents     A list containing one or more of ['CCD','field','catalog'] (default: all)
    @param formats     A list containing one or more of ['image','catalog'] (default: all)
    @param bins        A list of stile.BinObjects (default: None)
    """
    for pair in pairs:
        for epoch in epochs:
            for extent in extents:
                for format in formats:
                    this_format = Format(epoch,extent,format,pair=pair)
                    format_dict[this_format]['tests'] = format_dict[this_format].get(
                        'tests',[]) + [test]
                    format_dict[this_format]['req'] = format_dict[this_format].get(
                        'reqs',[]) + _GetRequiredFields(test)
                    format_dict[this_format]['bins'] = format_dict[this_format].get(
                        'bins',[]) + [bins]

def ChooseTestOptions(stile_args,id):
    """
    Given a dict of stile parameters (stile_args) and a data ID, figure out the tests, data formats,
    and binning schemes that should be used.  
    
    This returns a list.  Each element of the list is a tuple; the first element is a format
    (ie a combination of pair-or-single, single/multiepoch, spatial extent, and data format), and
    the second is a list.  This second list is also a list of tuples; the first element is a 
    stile.BinObject, and the second is a list of tests to use with that binning scheme.
    """
    if 'tests' in stile_args:
        tests = stile_args['tests']
    else:
        tests = stile.StandardTests()
    formats = stile.GetEmptyFormatDict(use_dict=True)
    for test in tests:
        if isinstance(test,stile.SysTest):
            _AddToFormatDict(formats,test)
        elif not hasattr(test,'__getitem__'):
            raise ValueError('Test list must include only stile.SysTest objects or dicts')
        elif 'id' in test:
            if id in test['id'] or id==test['id'] or not id:
                kwargs = {}
                for format in ['pairs','epochs','extents','formats']:
                    if format in test:
                        kwargs[format] = list(test[format])
                _AddToFormatDict(formats,test,bins=test.get('bins',None)**kwargs)
        else:
            kwargs = {}
            for format in ['pairs','epochs','extents','formats']:
                if format in test:
                    kwargs[format] = test[format]
            _AddToFormatDict(formats,test,bins=test.get('bins',None),**kwargs)
    format_list = []
    bin_and_test_list = []
    for key in format_dict.keys():
        if format_dict[key]['tests']:
            format_list.append(key.asKwargs())
            if format_dict[key]['req']:
                format_list[-1]['required_fields'] = list(set(format_dict[key]['req']))
            bin_list = []
            test_list = []
            for bin, test in zip(format_dict[key]['bins'],format_dict[key]['tests']):
                if hasattr(bin,'__iter__'):
                    for subbin in bin:
                        if subbin in bin_list:
                            test_list[bin_list.index(subbin)].append(test)
                        else:
                            bin_list.append(subbin)
                            test_list.append([test])
                else:
                    if bin in bin_list:
                        test_list[bin_list.index(bin).append(test)
                    else:
                        bin_list.append(bin)
                        test_list.append([test])
            #TODO: sort test_list by requiring data or not
            bin_and_test_list.append(zip(bin_list,test_list))
    return zip(format_list, bin_and_test_list)

def run(stile_args):
    if isinstance(stile_args,dict):
        stile_args = namespace(stile_args)
    data_handler = stile_args.data_handler
    ids = chooseIDs(stile_args)
    
    for id in ids:
        for format, bin_and_test_list in ChooseTestOptions(stile_args,id):
            for bin_obj, test_list in bin_and_test_list:
                for bin in bin_obj:
                    data = data_handler.getData(id,**format, bin=bin)
                    for test in test_list:
                        test(data,stile_args=stile_args,verbose=True)
                    
