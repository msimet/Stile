import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as P
import sys
import os
sys.path.append('..')
import stile
import dummy

def main():
    # setups
    dh = dummy.DummyDataHandler()
    bin_list = [stile.BinStep('ra',low=-1,high=1,step=1),
                stile.BinStep('dec',low=-1,high=1,step=1)]
    sys_test = stile.RealShearSysTest()
    
    stile_args = {'corr2_kwargs': { 'ra_units': 'degrees', 
                                    'dec_units': 'degrees',
                                    'min_sep': 0.05,
                                    'max_sep': 1,
                                    'sep_units': 'degrees',
                                    'nbins': 20
                                    } }
    data_ids = dh.listData(object_types = ['galaxy lens','galaxy'], epoch='single',
                           extent='field',data_format='table')
    
    # do a test without binning
    data = dh.getData(data_ids[0],'galaxy lens','single','field','table')
    data2 = dh.getData(data_ids[1],'galaxy','single','field','table')
    
    # convert all files to files on disk if they aren't already
    corr2_kwargs = stile.MakeCorr2FileKwargs(data,data2)
    
    # run the test
    results = sys_test(stile_args,**corr2_kwargs)
    
    fig = sys_test.plot(results)
    fig.savefig(sys_test.short_name+'.png')

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
        
        corr2_kwargs = stile.MakeCorr2FileKwargs(data,data2)
        
        results = sys_test(stile_args,**corr2_kwargs)
        stile.WriteASCIITable('realshear-'+bins_name+'.dat',results)
        fig = sys_test.plot(results)
        fig.savefig(sys_test.short_name+bins_name+'.png')
        print "Done with binned systematics test", bins_name

if __name__=='__main__':
    main()
    
