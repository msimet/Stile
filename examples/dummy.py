import sys
sys.path.append('..')
import stile

class DummyDataHandler(stile.DataHandler):
    def __init__(self):
        self.source_file_name = 'example_source_catalog.dat'
        self.lens_file_name = 'example_lens_catalog.dat'
        self.read_method = stile.ReadASCIITable
        self.fields={'id': 0, 'ra': 1, 'dec': 2, 'z': 3, 'g1': 4, 'g2': 5} 
        self.output_path='.'

    def listData(self,object_types,epoch,extent,data_format):
        if (epoch=='single' and 
            (extent=='field' or extent=='patch' or extent=='tract') and
            data_format=='table'):
            return_list = []
            for object_type in object_types:
                if object_type=="galaxy":
                    return_list.append(self.source_file_name)
                elif object_type=="galaxy lens":
                    return_list.append(self.lens_file_name)
                else:
                    raise NotImplementedError("Can only serve 'galaxy' or 'galaxy lens' "+
                                                 "object types")
            return return_list
        else:
            raise ValueError('DummyDataHandler does not contain data of this type: %s %s %s %s'%(
                                str(object_types),epoch,extent,data_format)) 

    def getData(self,id,object_types,epoch,extent,data_format,bin_list=None):
        if hasattr(id,'__iter__'):
            return [self.getData(iid,ot,epoch,extent,data_format,bin_list) 
                     for iid,ot in zip(id,object_types)]
        #TODO: think about what happens if a file gets rewritten due to field schema...
        if not data_format=='table':
            raise ValueError('Only table data provided by DummyDataHandler')
        if not epoch=='single':
            raise ValueError('Only single-epoch data provided by DummyDataHandler')
        if id==self.lens_file_name or id==self.source_file_name:
            if not bin_list:
                return (id,self.fields)
            else:
                data = stile.FormatArray(self.read_method(id),fields=self.fields)
                for bin in bin_list:
                    data = data[bin(data)]
                return data
        else:
            raise ValueError('Unknown data ID')
    
                
        
