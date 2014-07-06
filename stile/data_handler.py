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
    
    def listData(self,object_types,epoch,extent,data_format,required_fields=None):
        raise NotImplementedError()
    
    def getData(self,ident,object_types=None,epoch=None,extent=None,data_format=None,bin_list=None):
        """
        Return some data matching the `ident` for the given kwargs.  This can be a numpy array, a 
        tuple (file_name, field_schema) for a file already existing on the filesystem, or a list of 
        either of those things.
        
        If it's a tuple (file_name, field_schema), the assumption is that it can be read by a simple
        FITS or ASCII reader.  The format will be determined from the file extension.
        """ 
        raise NotImplementedError()

    def getOutputPath(self,extension='.dat',multi_file=False,*args):
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
            nfiles = glob.glob(os.path.join(self.output_path,sys_test_string)+'*'+extension)
            return os.path.join(self.output_path,sys_test_string+'_'+str(nfiles)+extension)
        else:
            return os.path.join(self.output_path,sys_test_string+extension)

class FileInfo(object):
    def __init__(self,file_name,reader,reader_kwargs=None,fields=None,flag_cols=None):
        self.file_name = file_name
        self.reader = reader
        self.reader_kwargs = reader_kwargs
        self.fields = fields
        if isinstance(self.flag_cols,str):
            self.flag_cols = [flag_cols]
        else:
            self.flag_cols = flag_cols
    def __eq__(self,other):
        return (self.file_name==other.file_name and self.reader==other.reader and 
                self.reader_kwargs==other.reader_kwargs and self.flags==other.flags and
                self.fields==other.fields)
    def getData(self,force=False):
        if not force and not self.flag_cols:
            return self.file_name
        else:
            return_data = stile.FormatArray(
                            self.reader(self.file_name,**reader_kwargs),fields=self.fields)
            if self.flag_cols:
                for flag_col in flag_cols:
                    return_data = return_data[return_data[flag_col]]
            return return_data
        
        
class ConfigDataHandler(DataHandler):        
    def __init__(self,stile_args):
        if 'config_file' in stile_args:
            config = self.loadConfig('config_file')
            stile_args.update(config)
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
        group = stile_utils.PopAndCheckFormat(stile_args,'group',bool,default=True)
        wildcard = stile_utils.PopAndCheckFormat(stile_args,'wildcard',bool,default=False)
        keys = sorted(stile_args.keys())
        file_list = []
        n = 0
        for key in keys:
            if key[:4]=='file' and key!='file_reader':
                file_obj = stile_args.pop(key)
                if isinstance(file_obj,list):
                    new_file_list, n = self._parseFileHelper(file_obj,group=False,wildcard=wildcard,
                                                         start_n=n)
                else:
                    new_file_list, n = self._parseFileHelper(file_obj,group=group,wildcard=wildcard,
                                                         start_n=n)
                file_list.append(new_file_list)
        fields = stile_utils.PopAndCheckFormat(stile_args,'fields',(list,dict),default=[])
        if fields:
            file_list = self.addKwarg('fields',fields,file_list)
        flag_field = stile_utils.PopAndCheckFormat(stile_args,'flag_field',(str,list,dict),
                                                   default=[])
        if flag_field:
            file_list = self.addKwarg('flag_field',flag_field,file_list,append=True)
        file_reader = stile_utils.PopAndCheckFormat(stile_args,'file_reader',(str,dict),default='')
        if file_reader:
            file_list = self.addKwarg('file_reader',file_reader,file_list)
        self.files, self.groups = self._fixGroupings(file_list)
        return self.files, self.groups # Return for checking purposes, mainly
        
    def _parseFileHelper(self,files,epoch=None,extent=None,data_format=None,object_type=None,
                              group=True,wildcard=False,flag_field=None,file_reader=None,
                              fields=[],start_n=0,recursed=False,**kwargs):
        if isinstance(files,list):
            # Okay, this is a list.  That means it's either the innermost bit of a nested dict, or
            # a list of self-contained dicts.  So loop through them and make sure the kwargs are
            # satisfied either by the keys in the item itself (if it's a dict), or by the kwargs
            # passed to this function.
            return_list = []
            for item in files:
                required_keys = []
                for keyval,keyname in [(epoch,'epoch'),(extent,'extent'),(data_format,'data_format'),
                                       (object_type,'object_type')]:
                    if not keyval:
                        required_keys.append(keyname)
                if isinstance(item,(list,str)):
                    # This is the innermost bit of a loop by definition, so: all the info must be
                    # in the kwargs or the data definition itself..  Make a dict, add a "name" 
                    # field, and add it to the list of dicts to be returned.
                    if required_keys:
                        raise ValueError('Received a list at a dict level missing %s: %s'%(
                                          required_keys,item))
                    else:
                        base_dict = {'epoch': epoch, 'extent': extent, 'data_format': data_format,
                                     'object_type': object_type, 'group': group, 
                                     'file_reader': file_reader, 'fields': fields}
                        base_dict.update(kwargs)
                        if base_dict['epoch']=='multiepoch':
                            if isinstance(item,list) and any([isinstance(i,dict) for i in item]):
                                raise ValueError("Cannot use dicts at this level for multiepoch to ensure proper processing: %s"%item)
                            names = self._expandWildcard(item,wildcard,True)
                            base_dict['name'] = names
                            return_list.append(base_dict)
                        else:
                            names = self._expandWildcard(item,wildcard,False)
                            for name in names:
                                # Make a new dict in the list for each item.
                                new_dict = copy.deepcopy(base_dict)
                                if isinstance(name,dict):
                                    new_dict.update(name)
                                else:
                                    new_dict['name'] = name
                                return_list.append(new_dict)
                elif isinstance(item,dict):
                    # Okay, this is a dict; either it's a list member and the format will come from 
                    # the kwargs, or it's a standalone dict.  Technically it can be somewhere in 
                    # between--it doesn't really matter for this processing, so I'm not going to
                    # *advertise* that you can switch in the middle, but it doesn't really matter.
                    # Apart from some trickery of updating dicts instead of just adding 'name', this
                    # code is doing the same thing as the same as the previous code block.
                    if item.get('epoch',epoch)=='multiepoch':
                        names = self._expandWildcard(item,item.get('wildcard',wildcard),True)
                        base_dict = {'epoch': epoch, 'extent': extent, 'data_format': data_format,
                                     'object_type': object_type, 'group': group, 
                                     'file_reader': file_reader, 'fields': fields}
                        base_dict.update(kwargs)
                        base_dict.update(item)
                        base_dict['name'] = names
                        return_list.append(base_dict)
                    else:
                        names = self._expandWildcard(item,item.get('wildcard',wildcard),False)
                        for name in names:
                            item_dict = copy.deepcopy(item)
                            base_dict = {'epoch': epoch, 'extent': extent, 'data_format': data_format,
                                         'object_type': object_type, 'group': group, 
                                         'file_reader': file_reader, 'fields': fields}
                            base_dict.update(kwargs)
                            base_dict.update(item_dict)
                            base_dict['name'] = name
                            return_list.append(base_dict)
        elif isinstance(files,dict):
            group = stile_utils.PopAndCheckFormat(files,'group',bool,default=group)
            wildcard = stile_utils.PopAndCheckFormat(files,'wildcard',bool,default=wildcard)
            fields = stile_utils.PopAndCheckFormat(files,'fields',(dict,list),default=fields)
            file_reader = files.get('file_reader',file_reader)
            if file_reader and not isinstance(file_reader,str) and not hasattr(file_reader,'__call__'):
                raise ValueError("Do not understand file reader: %s"%file_reader)
            if flag_field:
                flag_field += stile_utils.flatten(stile_utils.PopAndCheckFormat(
                                      files,'flag_field',(str,list),default=[]))
            else:
                flag_field = stile_utils.PopAndCheckFormat(files,'flag_field',(str,list),
                                                           default=[])
            return_list = []
            keys = files.keys()
            pass_kwargs = {'epoch': epoch, 'extent': extent, 'data_format': data_format, 
                           'object_type': object_type, 'group': group, 'wildcard': wildcard, 
                           'flag_field': flag_field, 'recursed': True}
            pass_kwargs.update(kwargs)
            for name, default, current_val in zip(['epoch','extent','data_format','object_type'],
                                              [stile_utils.epochs,stile_utils.extents,
                                               stile_utils.data_formats,stile_utils.object_types],
                                              [epoch,extent,data_format,object_type]):
                if any([key in default for key in keys]):
                    if current_val:
                        raise ValueError("Duplicate definition of %s: already have %s, "
                                         "requesting %s"%(name,current_val,keys))
                    for key in keys:
                        new_files = files.pop(key)
                        pass_kwargs[name] = key
                        new_return, new_n = self._parseFileHelper(new_files,**pass_kwargs)
                        return_list += new_return
            if files:
                if 'epoch' in files or 'extent' in files or 'data_format' in files or 'object_type' in files:
                    if (epoch or 'epoch' in files) and (extent or 'extent' in files) and (data_format or 'data_format' in files) and (object_type or 'object_type' in files) and 'name' in files:
                        return_list = [files]
                    else:
                        raise ValueError('File description does not include all format keywords: %s'%files)
                else:
                    raise ValueError("Unprocessed keys found: %s"%files.keys())
        else:
            raise ValueError("Do not understand file input: %s"%files)
        if not recursed:
            if group:
                return_list, start_n = self._group(return_list,start_n)
            return_val = self._formatFileList(return_list)
        else:
            return_val = return_list
        return return_val, start_n

    def _expandWildcard(self,item,wildcard,multiepoch):
        if isinstance(item,list):
            return_list = []
            for i in item:
                return_list += self._expandWildcardHelper(i,wildcard,multiepoch)
        else:
            return_list = self._expandWildcardHelper(item,wildcard,multiepoch)
        return [r for r in return_list if r] # filter empty entries
    
    def _expandWildcardHelper(self,item,wildcard,multiepoch):
        if isinstance(item,dict):
            names = item['name']
            wildcard = item.get('wildcard',wildcard)
            if 'epoch' in item:
                multiepoch = item['epoch']=='multiepoch'
        else:
            names = item
        if not wildcard:
            if multiepoch:
                if isinstance(names,list):
                    return names
                else:
                    return [names]
            else:
                return stile_utils.flatten(names)
        else:
            if multiepoch:
                if not hasattr(names,'__iter__'):
                    return sorted(glob.glob(names))
                elif any([hasattr(n,'__iter__') for n in names]):
                    return [self._expandWildcard(n,wildcard,multiepoch) for n in names]
                else:
                    return [sorted(glob.glob(n)) for n in names]
            elif hasattr(names,'__iter__'):
                return stile_utils.flatten([self._expandWildcard(n,wildcard,multiepoch) for n in names])
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
                for i,item1 in enumerate(files[key][obj_type][:-1]):
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
        
    def addKwarg(self,key,value,file_dicts,format_keys=[],object_type_key=None,append=False):
        if not isinstance(value,dict) or value.keys()==['name']:
            if isinstance(value,dict):
                value = value.pop('name')
            for file_dict in file_dicts:
                for format in file_dict:
                    for object_type in file_dict[format]:
                        for file in file_dict[format][object_type]:
                            if not append and (not key in file or not file[key]):
                                if (not format_keys or (format_keys and all([format_key in format for format_key in format_keys]))) and (not object_type_key or object_type==object_type_key):
                                    file[key] = value
                            elif append:
                                if key in file:
                                    file[key] = stile_utils.flatten([file[key],value])
                                else:
                                    file[key] = value
        else:
            object_types = [object_type for file_dict in file_dicts for format in file_dict for object_type in file_dict[format]]
            value_keys = value.keys()
            for value_key in value_keys:
                if value_key in value: # in case it was popped in a call earlier in this loop
                    new_value = value.pop(value_key)
                    if value_key=='extent' or value_key=='data_format' or value_key=='epoch':
                        self.addKwarg(key,value,file_dicts,format_keys=stile_utils.flatten([format_keys,new_value]),object_type_key=object_type_key,append=append)
                    elif value_key=='object_type':
                        self.addKwarg(key,value,file_dicts,format_keys=format_keys,object_type_key=new_value,append=append)
                    elif value_key in object_types:
                        self.addKwarg(key,new_value,file_dicts,format_keys=format_keys,object_type_key=value_key,append=append)
                    elif value_key=='name':
                        self.addKwarg(key,new_value,file_dicts,format_keys=format_keys,object_type_key=object_type_key,append=append)
                    else:
                        new_format_keys = stile_utils.flatten([format_keys,value_key])
                        self.addKwarg(key,new_value,file_dicts,format_keys=new_format_keys,object_type_key=object_type_key,append=append)
        return file_dicts
                
    def queryFile(self,file_name):
        return_list = []
        for item in self.file_list:
            if item['name']==file_name:
                return_list_item = []
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
    
                
        
