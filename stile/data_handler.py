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
    def __init__(self,stile_args):
        if 'config_file' in stile_args:
            config = self.loadConfig('config_file')
            config.update(stile_args)
            stile_args = config
        self.data_files = self.parseFiles(stile_args)
        self.stile_args = stile_args
        pass

    def loadConfig(self,files):
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
        # Get the 'group' and 'wildcard' keys, which indicate whether to try to match star & galaxy
        # etc files or to expand wildcards in filenames
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
        # Clean up formatting
        return_list, n = self._group(files,start_n)
        return_val = self._formatFileList(return_list)
        return return_val, n

    def _recurseDict(self,files,**kwargs):
        format_keys = ['epoch','extent','data_format','object_type']

        # First things first: if this is a list, we've recursed through all the levels of the dict.
        # The kwargs contain the format keys from all the superior levels, so we'll update with
        # the things contained in this list and return a list of dicts that isn't nested any more.
        if isinstance(files,list):
            if all([format_key in kwargs for format_key in format_keys]):
                if not kwargs: # This means it's a top-level list of dicts and shouldn't be grouped
                    pass_kwargs = {'group': False}
                elif kwargs.get('epoch')=='multiepoch':
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
                    # This one's easier, just recurse if the file list is iterable or else return
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
            
        # We didn't hit the previous "if" statement, so this is a dict.  Check for the non-format-
        # related keywords and add them...
        return_list = []
        
        pass_kwargs = copy.deepcopy(kwargs)
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
        for name, default in zip(format_keys,
                                 [stile_utils.epochs,stile_utils.extents,
                                  stile_utils.data_formats,stile_utils.object_types]):
            if any([key in default for key in keys]):
                if key in kwargs:
                    raise ValueError("Duplicate definition of %s: already have %s, "
                                     "requesting %s"%(name,current_val,keys))
                for key in keys:
                    new_files = files.pop(key)
                    pass_kwargs[name] = key
                    return_list += self._recurseDict(new_files,**pass_kwargs)
        # If there are keys left, it might be a single dict describing one file; check for that.
        if files:
            if any([format_key in files for format_key in format_keys]) and 'name' in files:
                if all([format_key in files or format_key in kwargs for format_key in format_keys]):
                    pass_kwargs.update(files)
                    return_list+=[files]
                else:
                    raise ValueError('File description does not include all format keywords: %s'%files)
            else:
                raise ValueError("Unprocessed keys found: %s"%files.keys())
        return return_list
        
    def _expandWildcard(self,item):
        if isinstance(item,list):
            return_list = [self._expandWildcardHelper(i) for i in item]
        else:
            return_list = self._expandWildcardHelper(item)
        return [r for r in return_list if r] # filter empty entries
    
    def _expandWildcardHelper(self,item,wildcard=False,is_multiepoch=False):
        if isinstance(item,dict):
            names = item['name']
            wildcard = item.pop('wildcard',wildcard)
            if 'epoch' in item:
                is_multiepoch = item['epoch']=='multiepoch'
        else:
            names = item
        if not wildcard:
            if is_multiepoch:
                if isinstance(names,list):
                    return names
                else:
                    return [names]
            else:
                return stile_utils.flatten(names)
        else:
            if is_multiepoch:
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
        format_dict = stile_utils.EmptyFormatDict(type=dict)
        return_list = []
        for l in list:
            if l:
                if not isinstance(l,dict):
                    raise TypeError('Outputs from _parseFileHelper should always be lists of dicts.  This is a bug.')
                if not 'group' in l or l['group'] is True:
                    format_obj = stile_utils.Format(epoch=l['epoch'],extent=l['extent'],data_format=l['data_format'])
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
        return_dict = {}
        for item in list:
            format_obj = stile_utils.Format(epoch=item.pop('epoch'),extent=item.pop('extent'),data_format=item.pop('data_format'))
            return_dict[format_obj.str] = return_dict.get(format_obj.str,{})
            object_type = item.pop('object_type')
            return_dict[format_obj.str][object_type] = return_dict[format_obj.str].get(object_type,[])
            if format_obj.epoch=="multiepoch":
                return_dict[format_obj.str][object_type].append(item)
            else:
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
                        del item1['group']
                    for j,item2 in enumerate(files[key][obj_type][i+1:]):
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
        for key in dict1:
            if key in dict2:
                dict1[key] += dict2[key]
        for key in dict2:
            if key not in dict1:
                dict1[key] = dict2[key]
        return dict1
                
    def _getGroups(self,file_dict):
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
        if not isinstance(value,dict) or value.keys()==['name']:
            if isinstance(value,dict):
                value = value.pop('name')
            for file_dict in file_dicts:
                for format in file_dict:
                    for object_type in file_dict[format]:
                        for file in file_dict[format][object_type]:
                            if not key in file or not file[key]:
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


        
    def queryFile(self,file_name):
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
        return [['pair','single','pointing','table']]
    def listData(self,pair_or_single,epoch,extent,data_format,random=False):
        pass
    def getData(self,id,pair_or_single,epoch,extent,data_format,
                      random=False,bin_list=None,force=False):
        pass
    
                
        
