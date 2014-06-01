import sys
sys.path.append('..')
import stile

class DummyDataHandler(stile.DataHandler):
    def __init__(self):
        self.source_file_name = 'test_source_catalog.dat'
        self.lens_file_name = 'test_lens_catalog.dat'
        self.read_method = stile.ReadAsciiTable
        self.fields={'id': 0, 'ra': 1, 'dec': 2, 'z': 3, 'g1': 4, 'g2': 5} 
        self.output_path='.'
    def listFileTypes(self):
        return [['pair','single','pointing','table']]
    def listData(self,pair_or_single,epoch,extent,data_format,random=False):
        if (epoch=='single' and 
            (extent=='pointing' or extent=='field' or extent=='survey') and
            data_format=='table' and not random):
            if pair_or_single=='pair':
                return [self.lens_file_name,self.source_file_name]
            elif pair_or_single=='single':
                return [self.source_file_name]
            else:
                raise ValueError('Do not understand value of pair_or_single: %s (should be '+
                                 '"pair" or "single")'%pair_or_single)
        elif random:
            raise ValueError('DummyDataHandler does not contain random data')
        else:
            raise ValueError('DummyDataHandler does not contain data of this type: %s %s %s %s'%(
                                pair_or_single,epoch,extent,data_format)) 
    def getData(self,id,pair_or_single,epoch,extent,data_format,
                      random=False,bin_list=None,force=False):
        #TODO: think about what happens if a file gets rewritten due to field schema...
        if not data_format=='table':
            raise ValueError('Only table data provided by DummyDataHandler')
        if not epoch=='single':
            raise ValueError('Only single-epoch data provided by DummyDataHandler')
        if random:
            raise ValueError('Random data not provided by DummyDataHandler')
        if id==self.lens_file_name or id==self.source_file_name:
            if not bin_list and not force:
                return (id,self.fields)
            else:
                data = stile.MakeRecarray(self.read_method(id),fields=self.fields)
                if bin_list:
                    for bin in bin_list:
                        data = data[bin(data)]
                return data
        else:
            raise ValueError('Unknown data ID')
    
                
        
