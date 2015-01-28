"""
data_handler.py: defines the classes that serve data to the various Stile systematics tests in the 
default drivers.
"""
import os
import glob
import copy
import stile_utils

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
         -- data_format: "image" or "catalog."  Right now no image-level tests are implemented but we
              request this kwarg for future development.
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

class ConfigDataHandler(DataHandler):        
    def __init__(self, stile_args):
        if 'config_file' in stile_args:
            config = self.loadConfig('config_file')
            config.update(stile_args)
            stile_args = config
        self.parseFiles(stile_args)
        self.stile_args = stile_args
        
    def loadConfig(self,files):
        """
        Read in a config file or a list of config files.  If a list, condense into one config dict,
        with later config files taking precedence over earlier config files.
        """
        try:
            import yaml as config_reader
            has_yaml=True
        except:
            import json as config_reader
            has_yaml=False
        if isinstance(files,str):
            try:
                config = config_reader.read(files)
            except Exception as e:
                if not has_yaml and os.path.splitext(files)[-1].lower()=='.yaml':
                    raise ValueError('It looks like this config file is a .yaml file, but you '+
                                     "don't seem to have a working yaml module: %s"%files)
                else:
                    raise e
        elif hasattr(files,'__iter__'):
            config_list = [self.loadConfig(file) for file in files]
            config = config_list[0]
            for config_item in config_list[1:]:
                config.update(config_item)
        return config
    
    def parseFiles(self,stile_args):
        """
        Process the arguments from the config file/command line that tell Stile which data files
        to use and how to use them.
        """
        # Get the 'group' and 'wildcard' keys at the file level, which indicate whether to try to 
        # match star & galaxy etc files or to expand wildcards in filenames.
        group = stile_utils.PopAndCheckFormat(stile_args,'group',bool,default=True)
        wildcard = stile_utils.PopAndCheckFormat(stile_args,'wildcard',bool,default=False)
        keys = sorted(stile_args.keys())
        file_list = []
        n = 0
        for key in keys:
            # Pull out the file arguments and process them
            if key[:4]=='file' and key!='file_reader':
                file_obj = stile_args.pop(key)
                new_file_list, n = self._parseFileHelper(file_obj,start_n=n)
                file_list.append(new_file_list)
        # Now, update the file list with global arguments if lower levels didn't already override
        # them
        fields = stile_utils.PopAndCheckFormat(stile_args,'fields',(list,dict),default=[])
        if fields:
            file_list = self.addKwarg('fields',fields,file_list)
        flag_field = stile_utils.PopAndCheckFormat(stile_args,'flag_field',(str,list,dict),
                                                   default=[])
        if flag_field:
            file_list = self.addKwarg('flag_field',flag_field,file_list)
        file_reader = stile_utils.PopAndCheckFormat(stile_args,'file_reader',(str,dict),default='')
        if file_reader:
            file_list = self.addKwarg('file_reader',file_reader,file_list)
        # Clean up the grouping keywords and group names
        self.files, self.groups = self._fixGroupings(file_list)
        return self.files, self.groups # Return for checking purposes, mainly
        
    def _parseFileHelper(self,files,start_n=0):
        # Recurse through all the levels of the current file arg
        if isinstance(files,dict):
            # This is a nested dict, so recurse down through it and turn it into a list of dicts
            # instead.
            files = self._recurseDict(files)
        else:
            # If it's already a list, check that it's a list of dicts
            if not hasattr(files,'__iter__'):
                raise ValueError('file config keyword must be a list or dict: got %s of type %s'%(
                                    files,type(files)))
            for file in files:
                if not isinstance(file,dict):
                    raise ValueError('If file parameter is a list, each element must be a dict.  '+
                        'Got %s of type %s instead.'%(file,type(file)))
                # We don't group lists unless specifically instructed, so set that keyword
                file['group'] = file.get('group',False) 
        # Now run through the list of dicts that we've created, check all the formats are right,
        # and expand any wildcards.
        for item in files:
            # Check for proper formatting and all relevant keywords, plus expand wildcards if 
            # requested
            if not isinstance(item,dict):
                raise ValueError('Expected a list of dicts. Either the config file was in error, '
                                 'or the Stile processing failed.  Current state of "files" '
                                 'argument: %s, this item type: %s'%(files,type(item)))
            format_keys = ['epoch','extent','data_format','object_type']
            if not all([format_key in item for format_key in format_keys]):
                raise ValueError('Got file item %s missing one of the required format keywords %s'%
                                    (item,format_keys))
            item['name'] = self._expandWildcard(item)
        # Clean up formatting and group these files, if applicable
        return_list, n = self._group(files,start_n)
        return_val = self._formatFileList(return_list)
        return return_val, n

    def _recurseDict(self,files, require_format_args=True, **kwargs):
        """
        Recurse through a dictionary of dictionaries (of dictionaries...) contained in the first 
        arg, called here "files" although it can be any kind of object.  The kwargs are keys from
        a higher level of the dict which should be included in all lower-level items unless 
        overridden by the lower levels explicitly.  Return a list of dicts.
        
        Set require_format_args to False if the argument "files" doesn't need to have a complete 
        list of all format keys for each element, eg if this is being used to define tests instead
        of files.
        """
        format_keys = ['epoch','extent','data_format','object_type']

        # First things first: if this is a list, we've recursed through all the levels of the dict.
        # The kwargs contain the format keys from all the superior levels, so we'll update with
        # the things contained in this list and return a list of dicts that isn't nested any more.
        if isinstance(files,list):
            if not kwargs: # This means it's a top-level list of dicts and shouldn't be grouped
                for file in files:
                    file['group'] = file.get('group', default=False)
                return files
            elif (all([format_key in kwargs for format_key in format_keys]) or 
                  require_format_args is False):
                if kwargs.get('epoch')=='multiepoch':
                    # Multiepoch files come in a set, so we can't turn them into single items the
                    # way we do with coadds & single epoch files.
                    if isinstance(files,dict):
                        # Copy the kwargs, then update with the stuff in this dict (which should
                        # override higher levels).
                        pass_kwargs = copy.deepcopy(kwargs)
                        pass_kwargs.update(files)
                    elif isinstance(files,(list,tuple)):
                        # Okay.  It's a list of files.  If each item of the list is itself iterable,
                        # then we can split the list up; if none of them are iterable, it's a set
                        # that should be analyzed together.  Anything else is an error.
                        iterable = [hasattr(item,'__iter__') for item in files]
                        if all(iterable):
                            return_list = []
                            for item in files:  
                                pass_kwargs = copy.deepcopy(kwargs)
                                if isinstance(item,dict):
                                    pass_kwargs.update(item)
                                else:
                                    pass_kwargs['name'] = item
                                return_list.append(pass_kwargs)
                            return return_list
                        elif any(iterable):
                            raise ValueError('Cannot interpret list of items for multiepoch: '+
                                             files+'. Should be an iterable, or an iterable of '+
                                             'iterables.')
                        else:
                            pass_kwargs.update({'name': files})
                            return [pass_kwargs]
                    else:
                        raise ValueError('Cannot interpret list of items for multiepoch: '+
                                          files+'. Should be an iterable, or an iterable of '+
                                         'iterables.')
                else:
                    # This one's easier, just loop through if the file list is iterable or else return
                    # the item.
                    return_list = []
                    for file in files:
                        pass_kwargs = copy.deepcopy(kwargs)
                        if isinstance(file,dict):
                            pass_kwargs.update(file)
                        else:
                            pass_kwargs['name'] = file
                        return_list += [pass_kwargs]
                    return return_list
            else:
                raise ValueError('File description does not include all format keywords: %s, %s'%(files,kwargs))
            
        # We didn't hit the previous "if" statement, so this is a dict.  
        return_list = []
        pass_kwargs = copy.deepcopy(kwargs)

        # First check for the variables that control how we interpret the items in a given format.
        # We can't just pass_kwargs[key] = PopAndCheckFormat because we want to distinguish cases 
        # where we are explicitly given False (which can override higher-level Trues) and cases
        # where we would input False by default (which can't override).
        if 'group' in files:
            pass_kwargs['group'] = stile_utils.PopAndCheckFormat(files,'group',bool)
        if 'wildcard' in files:
            pass_kwargs['wildcard'] = stile_utils.PopAndCheckFormat(files,'wildcard',bool)
        if 'fields' in files:
            pass_kwargs['fields'] = stile_utils.PopAndCheckFormat(files,'fields',(dict,list))
        if 'flag_field' in files:
            pass_kwargs['flag_field'] = stile_utils.PopAndCheckFormat(files,'flag_field',(str,list))
        if 'file_reader' in files:
            pass_kwargs['file_reader'] = files.get('file_reader')
        
        # Now, if there are format keywords, recurse through them, removing the keys as we go.
        keys = files.keys()
        for format_name, default_vals in zip(format_keys,
                                 [stile_utils.epochs,stile_utils.extents,
                                  stile_utils.data_formats,stile_utils.object_types]):
            if any([key in default_vals for key in keys]):  # any() so users have the ability to
                if format_name in kwargs:                   # arbitrarily define eg extents 
                    raise ValueError("Duplicate definition of %s: already have %s, "
                                     "requesting %s"%(format_name,kwargs[format_name],keys))
                for key in keys:
                    new_files = files.pop(key)
                    pass_kwargs[format_name] = key
                    return_list += self._recurseDict(new_files,**pass_kwargs)
        # If there are keys left, it might be a single dict describing one file; check for that.
        if files:
            if any([format_key in files for format_key in format_keys]) and 'name' in files:
                if (all([format_key in files or format_key in kwargs for format_key in format_keys])
                    or require_format_args==False):
                    pass_kwargs.update(files)
                    return_list+=[pass_kwargs]
                else:
                    raise ValueError('File description does not include all format keywords: %s'%files)
            else:
                raise ValueError("Unprocessed keys found: %s"%files.keys())
        return return_list
        
    def _expandWildcard(self,item):
        #  Expand wildcards in a file list
        if isinstance(item,list):
            return_list = [self._expandWildcardHelper(i) for i in item]
        else:
            return_list = self._expandWildcardHelper(item)
        return [r for r in return_list if r] # filter empty entries
    
    def _expandWildcardHelper(self,item,wildcard=False,is_multiepoch=False):
        # Expand wildcards for an individual file item.
        if isinstance(item,dict):
            # If it's a dict, pull out the "name" attribute, and check if we want to do wildcarding.
            names = item['name']
            wildcard = item.pop('wildcard',wildcard)
            if 'epoch' in item:
                is_multiepoch = item['epoch']=='multiepoch'
        else:
            names = item
        if not wildcard:
            # Easy: just return what we have in a user-friendly format.
            if is_multiepoch:
                if isinstance(names,list):
                    return names
                else:
                    return [names]
            else:
                return stile_utils.flatten(names)
        else:
            if is_multiepoch:
                # We have to be a bit careful about expanding wildcards in the multiepoch case,
                # since we need to keep sets of files together.
                if not hasattr(names,'__iter__'):
                    return sorted(glob.glob(names))
                elif any([hasattr(n,'__iter__') for n in names]):
                    return [self._expandWildcardHelper(n,wildcard,is_multiepoch) for n in names]
                else:
                    return [sorted(glob.glob(n)) for n in names]
            elif hasattr(names,'__iter__'):
                return stile_utils.flatten([self._expandWildcardHelper(n,wildcard,is_multiepoch) for n in names])
            else:
                return glob.glob(names)
        
    def _group(self,list,n):
        """
        For a given list of dicts `list`, build up a list of all the files with the same format but
        different object types.  If the length of the file list for each object type with a given
        format is the same as the file list for the other object types with that format, and the
        `group` keyword is not given or is True, then group those together as files that should be
        analyzed together when we need multiple object types.  Return a list of files with the new
        group kwargs, plus a number "n" of groups which have been made.
        """
        format_dict = stile_utils.EmptyFormatDict(type=dict)
        return_list = []
        for l in list:
            if l:
                if not isinstance(l,dict):
                    raise TypeError('Outputs from _parseFileHelper should always be lists of dicts.  This is a bug.')
                if not 'group' in l or l['group'] is True:
                    format_obj = stile_utils.Format(epoch=l['epoch'],extent=l['extent'],data_format=l['data_format'])
                    if not format_obj.str in format_dict: # In case of user-defined formats
                        format_dict[format_obj.str] = {}
                    if not l['object_type'] in format_dict[format_obj.str]:
                        format_dict[format_obj.str][l['object_type']] = []
                    this_dict = format_dict[format_obj.str][l['object_type']]
                    if isinstance(l['name'],str) or l['epoch']=='multiepoch':
                        return_list.append(l)
                        this_dict.append(len(return_list)-1)
                    else:
                        for lname in l['name']:
                            new_dict = copy.deepcopy(l)
                            new_dict['name'] = lname
                            return_list.append(new_dict)
                            this_dict.append(len(return_list)-1)
                else:
                    return_list.append(l)
        for key in format_dict:
            if format_dict[key]:
                # Check if there are multiple object_types for this format and, if so, if the file 
                # lists are the same length
                len_files_list = [len(format_dict[key][object_type]) for object_type in format_dict[key]]
                if len(len_files_list)>1 and len(set(len_files_list))==1:
                    for object_type in format_dict[key]:
                        curr_n = n
                        for i in format_dict[key][object_type]:
                            return_list[i]['group'] = '_stile_group_'+str(curr_n)
                            curr_n+=1
                    if not curr_n-n == len_files_list[0]:
                        raise RuntimeError('Number of found files is greater than number of expected files: this is a bug')
                    n = curr_n
        return return_list, n
    
    def _formatFileList(self,list):
        """
        Turn a list of dicts back into a dict whose keys are the string versions of the format_obj 
        objects corresponding to the formats in each item of the original list and whose values are
        themselves dicts, with keys being the object types and values being a list of dicts
        containing all the other information about each file.  E.g.,
        {'multiepoch-CCD-catalog': {
            'galaxy': [file1_dict, file2_dict, ...],
            'star': [file3_dict, file4_dict, ...]
            }
        }
        """
        return_dict = {}
        for item in list:
            format_obj = stile_utils.Format(epoch=item.pop('epoch'),extent=item.pop('extent'),data_format=item.pop('data_format'))
            return_dict[format_obj.str] = return_dict.get(format_obj.str,{})
            object_type = item.pop('object_type')
            return_dict[format_obj.str][object_type] = return_dict[format_obj.str].get(object_type,[])
            if format_obj.epoch=="multiepoch":
                return_dict[format_obj.str][object_type].append(item)
            else:
                # If there name argument is a list of names, then turn it into one dict per name.
                names = stile_utils.flatten(item.pop('name'))
                for name in names:
                    new_dict = copy.deepcopy(item)
                    if isinstance(name,dict):
                        new_dict.update(name)
                    else:
                        new_dict['name'] = name
                    return_dict[format_obj.str][object_type].append(new_dict)
        return return_dict
    
    def _fixGroupings(self,list_of_dicts):
        """
        Take a list of dicts as output by self._formatFileList() and merge them into one dict.
        Return that dict and a dict describing the groups (as output by self._getGroups()).
        """
        list_of_dicts = stile_utils.flatten(list_of_dicts)
        files = list_of_dicts.pop(0)
        for dict in list_of_dicts:
            for key in dict.keys():
                if key in files:
                    files[key] = self._merge(files[key],dict[key])
                else:
                    files[key] = dict[key]
        for key in files:
            for obj_type in files[key]:
                del_list = []
                for i,item1 in enumerate(files[key][obj_type]):
                    if 'group' in item1 and isinstance(item1['group'],bool):
                        # If "group" is true but no group has been assigned, delete the 'group' key
                        del item1['group']
                    for j,item2 in enumerate(files[key][obj_type][i+1:]):
                        # Now cycle through the file list.  If there are two items which are the same except for the 'group' key,
                        # make the 'group' key of the first item a list containing all the group IDs from both, then mark the other 
                        # instance of the same item for deletion (but don't delete it yet since we're in the middle of a loop).
                        if 'group' in item2 and isinstance(item2['group'],bool):
                            del item2['group']
                        if not (j+i+1 in del_list) and (item1.keys()==item2.keys() or 
                            set(item1.keys()).symmetric_difference(set(item2.keys()))==set(['group'])) and all(
                            [item1[ikey]==item2[ikey] for ikey in item1.keys() if ikey!='group']):
                            item1['group'] = stile_utils.flatten(item1.get('group',[]))+stile_utils.flatten(item2.get('group',[]))
                            del_list.append(j+i+1)
                del_list.sort()
                del_list.reverse()
                for j in del_list:
                    files[key][obj_type].pop(j)
        return files, self._getGroups(files)
        
    def _merge(self,dict1,dict2):
        """
        Merge two dicts into one, concatenating rather than replacing any keys that are in both 
        dicts.
        """
        for key in dict1:
            if key in dict2:
                if isinstance(dict1[key], list):
                    if isinstance(dict2[key], list):
                        dict1[key] += dict2[key]
                    else:
                        dict1[key] += [dict2[key]]
                else:
                    if isinstance(dict2[key], list):
                        dict1[key] = [dict1[key]] + dict2[key]
                    else:
                        dict1[key] = [dict1[key], dict2[key]]
        for key in dict2:
            if key not in dict1:
                dict1[key] = dict2[key]
        return dict1
                
    def _getGroups(self,file_dict):
        """
        Make a dict corresponding to file_dict (an output of self._fixGroupings() above) with the
        following structure:
        {group_name: 
            {format_key:
                {object_type_1: index into the corresponding list in file_dict,
                 object_type_2: index into the corresponding list in file_dict, ...
                }
            }
        }
                
        """
        groups = {}
        for key in file_dict.keys():
            groups[key] = {}
            for obj_type in file_dict[key].keys():
                groups[key][obj_type] = [(i,item['group']) for i,item in enumerate(file_dict[key][obj_type]) if isinstance(item,dict) and 'group' in item]
        reverse_groups = {}
        for key in groups: 
            for obj_type in groups[key]:
                for i,group_names in groups[key][obj_type]:
                    if not isinstance(group_names,list):
                        group_names = [group_names]
                    for group_name in group_names:
                        if not isinstance(group_name,bool):
                            if not group_name in reverse_groups:
                                reverse_groups[group_name] = {key: {}}
                            elif not key in reverse_groups[group_name]:
                                raise ValueError('More than one format type found in group %s: %s, %s'%(group_name, key, reverse_groups[group_name].keys()))
                            if obj_type in reverse_groups[group_name][key]:
                                raise RuntimeError("Multiple files with same object type indicated for group %s: %s, %s"%(
                                    group_name, reverse_groups[group_name][key][obj_type], file_dict[key][obj_type][i]))
                            reverse_groups[group_name][key][obj_type] = i
        # Now eliminate any duplicate groups
        del_list = []
        keys = reverse_groups.keys()
        for i,group in enumerate(keys):
            if group not in del_list:
                for j,group2 in enumerate(keys[i+1:]):
                    if group2 not in del_list:
                        if reverse_groups[group]==reverse_groups[group2]:
                            del reverse_groups[group2]
                            del_list.append(group2)
                            self._removeGroup(file_dict,group2)
        return reverse_groups

    def _removeGroup(self,file_dict,group):
        """
        Remove a group name from every file descriptor it's part of.
        """
        for key in file_dict:
            for obj_type in file_dict[key]:
                for file in file_dict[key][obj_type]:
                    if 'group' in file:
                        if isinstance(file['group'],list):
                            if group in file['group']:
                                file['group'].remove(group)
                        else:
                            if file['group']==group:
                                file['group'] = True
        
    def addKwarg(self,key,value,file_dicts,format_keys=[],object_type_key=None):
        """
        Add the given (key,value) pair to all the lowest-level dicts in file_list if the key is not
        already present.  To be used for global-level keys AFTER all the individual file descriptors
        have been read in (ie this won't override anything that's already in the file dicts).
        
        The "value" can be a dict with the desired argument in "name" and specific directions for
        which formats/object types to apply to in the other key/value pairs.  For example, one could
        pass:
            key='file_reader'
            value={'name': 'ASCII', 'extent': 'CCD'}
        and then ONLY the lowest-level dicts whose extent is 'CCD' would be changed (again, only if
        there is no existing 'file_reader' key: this method never overrides already-existing keys).
        
        @param key              The key to be added
        @param value            The value of that key (plus optional limits, see above)
        @param file_dicts       A list of file_dicts in nested format
        @param format_keys      Only change these format keys! (list of strings, default: [])
        @param object_type_key  Only change this object type key! (string, default: None)
        @returns                The original file_dicts, with added keys as requested.
        
        Note that the end user generally shouldn't pass format_keys or object_type_key arguments:
        those kwargs are intended for use by this function itself in recursive calls.
        """
        # If there are no remaining restrictions to process:
        if not isinstance(value,dict) or value.keys()==['name']:  
            if isinstance(value,dict):
                value = value.pop('name')
            for file_dict in file_dicts:
                for format in file_dict:
                    for object_type in file_dict[format]:
                        for file in file_dict[format][object_type]:
                            if not key in file or not file[key]:
                                # If no restrictions are present, or if this file meets the restrictions:
                                if (not format_keys or (format_keys and all([format_key in format for format_key in format_keys]))) and (not object_type_key or object_type==object_type_key):
                                    file[key] = value
        else:
            object_types = [object_type for file_dict in file_dicts for format in file_dict for object_type in file_dict[format]]
            value_keys = value.keys()
            for value_key in value_keys:
                if value_key in value: # in case it was popped in a call earlier in this loop
                    new_value = value.pop(value_key)
                    if value_key=='extent' or value_key=='data_format' or value_key=='epoch':
                        self.addKwarg(key,value,file_dicts,format_keys=stile_utils.flatten([format_keys,new_value]),object_type_key=object_type_key)
                    elif value_key=='object_type':
                        self.addKwarg(key,value,file_dicts,format_keys=format_keys,object_type_key=new_value)
                    elif value_key in object_types:
                        self.addKwarg(key,new_value,file_dicts,format_keys=format_keys,object_type_key=value_key)
                    elif value_key=='name':
                        self.addKwarg(key,new_value,file_dicts,format_keys=format_keys,object_type_key=object_type_key)
                    else:
                        new_format_keys = stile_utils.flatten([format_keys,value_key])
                        self.addKwarg(key,new_value,file_dicts,format_keys=new_format_keys,object_type_key=object_type_key)
        return file_dicts

        
    def _checkAndCoerceFormat(self, epoch, extent, data_format): 
        """check for proper formatting of the epoch/extent/data_format kwargs and turn them into a string so we can use them as dict keys"""
        if not isinstance(epoch, stile_utils.Format) and (not hasattr(epoch,'__str__') or (extent and not hasattr(extent,'__str__')) or (data_format and not hasattr(data_format,'__str__'))):
            raise ValueError('epoch (and extent and data_format) must be printable as strings; given %s %s %s'%(epoch,extent,data_format))
        if extent or data_format:
            epoch = stile_utils.Format(epoch=epoch, extent=extent, data_format=data_format).str
        return epoch

    def queryFile(self,file_name):
        """
        Return a description of every place the named file occurs in the dict of file descriptors.
        Useful to figure out where a file is going if you're having trouble telling how the 
        parser is understanding your config file.
        """
        return_list = []
        for format in self.files:
            for object_type in self.files[format]:
                for item in self.files[format][object_type]:
                    if item['name']==file_name:
                        return_list_item = []
                        return_list_item.append("format: "+format)
                        return_list_item.append("object type: "+object_type)
                        for key in item:
                            if not key=='name':
                                return_list_item.append(key+': '+str(item[key]))
                        return_list.append(str(len(return_list)+1)+' - '+', '.join(return_list_item))
        return '\n'.join(return_list)
                
    def listFileTypes(self):
        """
        Return a list of format strings describing the epoch, extent, and data format of available files
        """
        return [format for format in self.files]

    def listObjects(self, epoch, extent=None, data_format=None):    
        """
        Return a list of object types available for the given format.  The format can be given as a string "{epoch}-{extent}-{data format}" or as three arguments, epoch, extent, data_format.
        """
        epoch = self._checkAndCoerceFormat(epoch, extent, data_format)
        if epoch in self.files:
            return [obj for obj in self.files[epoch]]
        else:
            return []
            
    def listData(self, object_type, epoch, extent=None, data_format=None):
        """
        Return a list of data files for the given object_type (or object_types, see below) and data format.
        
        The object_type argument can be a single type, in which case all data files meeting the format and object_type criteria are returned in a list.  The object_type argument can also be a list/tuple of object types; in that case, if there are pairs (triplets, etc) of files which should be analyzed together given those types, the returned value will be a list of lists, where the innermost list is a set of files to be analyzed together (with the order of the list corresponding to the order of items in the object_type) and the overall list is the set of such sets of files.
        
        Note that "multiepoch" formats will include a LIST of files instead of a single file in all cases where a single file is described above (so e.g. a list of object types will retrieve a list of lists of lists instead of just a list of lists).
        
        The format can be given as a string "{epoch}-{extent}-{data format}" or as three arguments, epoch, extent, data_format.
        """
        epoch = self._checkAndCoerceFormat(epoch, extent, data_format)
        if not hasattr(object_type,'__hash__') or (hasattr(object_type,'__iter__') and not all([hasattr(obj,'__hash__') for obj in object_type])):
            raise ValueError('object_type argument must be able to be used as a dictionary key, or be an iterable all of whose elements can be used as dictionary keys: given %s'%object_type)
        if not hasattr(object_type, '__iter__'):
            if epoch in self.files and object_type in self.files[epoch]:
                return [file for file in self.files[epoch][object_type]]
            else:
                return []
        elif isinstance(object_type, list):
            groups_list = []
            for group in self.groups:
                if epoch in self.groups[group] and all([obj in self.groups[group][epoch] for obj in object_type]):
                    groups_list.append(group)
            return [[self.groups[group][epoch][obj] for obj in object_type] for group in groups_list]
            
    def getData(self, data_id, object_type, epoch, extent=None, data_format=None, bin_list=None):
        """
        Return the data corresponding to 'data_id' for the given object_type and data format.
        
        The format can be given as a string "{epoch}-{extent}-{data format}" or as three arguments, epoch, extent, data_format.  For the ConfigDataHandler, the 'object_type', 'epoch' and 'extent' arguments are ignored; only the 'data_format' and 'data_id' kwargs (or the data format pulled from a format string) are considered.  The other arguments are kept for compatibility of call signatures.
        
        A list of Bin objects can be passed with the kwarg 'bin_list'.  These bins are "and"ed, not looped through!  This is not generally recommended for this ConfigDataHandler since data is not cached--it's better to bin the data at another stage, keeping the whole array in memory between binnings.  (Some of this caching is done by Python or your OS, but if you have many files it may not work.)  
        """
        epoch = self._checkAndCoerceFormat(epoch, extent, data_format)
        if not data_format:
            data_format = epoch.split('-')[-1]
        data_format=data_format.lower()
        if not hasattr(object_type,'__hash__') or (hasattr(object_type,'__iter__') and not all([hasattr(obj,'__hash__') for obj in object_type])):
            raise ValueError('object_type argument must be able to be used as a dictionary key, or be an iterable all of whose elements can be used as dictionary keys: given %s'%object_type)
            
        if isinstance(data_id, str):
            data_id = {'name': data_id}
        if 'fields' in data_id:
            fields = data_id['fields']
        else:
            fields=None
        if 'file_reader' in data_id:
            if d['file_reader']=='ASCII': 
                data = stile.ReadASCIITable(data_id['name'], fields=fields)
            elif d['file_reader']=='FITS':
                if data_format=='catalog':
                    data = stile.ReadFITSTable(data_id['name'], fields=fields)
                elif data_format=='image':
                    data = stile.ReadFITSImage(data_id['name'])
                else:
                    raise RuntimeError('Data format must be either "catalog" or "image", given %s'%data_format)
        elif 'catalog' in format:
            data = stile.ReadTable(data_id['name'], fields=fields)
        elif 'image' in format:
            data = stile.ReadImage(data_id['name'])
        else:
            raise RuntimeError('Data format must be either "catalog" or "image", given %s'%data_format)
        if data_id['flag_field']:
            flag = data_id['flag_field']
            if isinstance(flag,str):
                data = data[data[flag]==False]
            elif isinstance(flag, list):
                for f in flag:
                    data = data[data[f]==False]
            elif isinstance(flag,dict):
                for key in flag:
                    data = data[data[key]==flag[key]]
            else:
                raise ValueError('flag_field kwarg must be a string, list, or dict; given %s'%flag)
        if bin_list:
            for bin in bin_list:
                data = bin(data)
        return data
        
