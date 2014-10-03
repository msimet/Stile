import sys
sys.path.append('..')
import stile
import dummy

def main():
    # setups
    dh = dummy.DummyDataHandler()
    bin_list = [stile.BinStep('ra',low=-1,high=1,step=1),
                stile.BinStep('dec',low=-1,high=1,step=1)]
    sys_test = stile.GalaxyShearSysTest()
    
    stile_args = {'ra_units': 'degrees', 'dec_units': 'degrees',
                  'min_sep': 0.05, 'max_sep': 1, 'sep_units': 'degrees', 'nbins': 20}
    data_ids = dh.listData(object_types = ['galaxy lens','galaxy'], epoch='single',
                           extent='field', data_format='table')
    
    # do a test without binning
    data = dh.getData(data_ids[0],'galaxy lens','single','field','table')
    data2 = dh.getData(data_ids[1],'galaxy','single','field','table')
    
    # run the test
    results = sys_test(data, data2=data2, config=stile_args)
    
    #fig = sys_test.plot(results)
    #fig.savefig(sys_test.short_name+'.png')

    stile.WriteASCIITable('realshear.dat',results)
    print "Done with unbinned systematics test"

    # do with binning
    data = dh.getData(data_ids[0],'galaxy lens','single','field','table')
    # turns a list of binning schemes into a pseudo-nested list of single bins
    expanded_bin_list = stile.ExpandBinList(bin_list)
    handles_list = []
    deletes_list = []
    # for each set of bins, do the systematics test as above
    for bin_list in expanded_bin_list:
        bins_name = '-'.join([bl.short_name for bl in bin_list])
        data2 = dh.getData(data_ids[1],'galaxy','single','field','table',bin_list=bin_list)
        results = sys_test(data, data2=data2, config=stile_args)
        stile.WriteASCIITable('realshear-'+bins_name+'.dat',results)
        fig = sys_test.plot(results)
        fig.savefig(sys_test.short_name+bins_name+'.png')
        print "Done with binned systematics test", bins_name

if __name__=='__main__':
    main()

