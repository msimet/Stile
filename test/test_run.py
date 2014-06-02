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
    test = stile.TestXShear()
    
    stile_args = {'corr2_options': { 'ra_units': 'degrees', 
                                     'dec_units': 'degrees',
                                     'min_sep': 0.05,
                                     'max_sep': 1,
                                     'sep_units': 'degrees',
                                     'nbins': 20
                                     } }
    data_ids = dh.listData('pair','single','field','table')
    
    # do a test without binning
    data = dh.getData(data_ids[0],'pair','single','field','table')
    data2 = dh.getData(data_ids[1],'pair','single','field','table')
    
    # convert all files to files on disk if they aren't already
    data, data2, _, _, corr2_kwargs, handles, deletes = stile.MakeFiles(dh, data, data2)
    stile_args['corr2_options'].update(corr2_kwargs) 
    
    # run the test
    results = test(stile_args,dh,data,data2)
    # close and delete any temporary files
    for handle in handles:
        os.close(handle)
    for delete in deletes:
        if os.file_exists(delete):
            os.remove(delete)
    # Plot the results
    P.errorbar(results['<R>'],results['<gamX>'],yerr=results['sig'],fmt='og',label='cross')
    P.errorbar(results['<R>'],results['<gamT>'],yerr=results['sig'],fmt='or',label='tangential')
    P.xscale('log')
    P.xlim([0.05,0.7])
    P.xlabel('<R> [deg]')
    P.ylabel('<gam>')
    P.title('All data')
    P.legend()
    P.savefig(test.short_name+'.png')
    P.clf()
    print "Done with unbinned test"
    
    # do with binning
    data = dh.getData(data_ids[0],'pair','single','field','table')
    # turns a list of binning schemes into a pseudo-nested list of single bins
    expanded_bin_list = stile.ExpandBinList(bin_list)
    handles_list = []
    deletes_list = []
    # for each set of bins, do the test as above
    for bin_list in expanded_bin_list:
        stile_args['bins_name'] = '-'.join([bl.short_name for bl in bin_list])
        data2 = dh.getData(data_ids[1],'pair','single','field','table',bin_list=bin_list)
        
        new_data, new_data2, _, _, corr2_kwargs, handles, deletes = stile.MakeFiles(dh,data,data2)
        handles_list.append(handles)
        deletes_list.append(deletes)
        stile_args['corr2_options'].update(corr2_kwargs)
        
        results = test(stile_args,dh,new_data,new_data2)
        P.errorbar(results['<R>'],results['<gamX>'],yerr=results['sig'],fmt='og',label='cross')
        P.errorbar(results['<R>'],results['<gamT>'],yerr=results['sig'],fmt='or',label='tangential')
        P.xlabel('<R> [deg]')
        P.ylabel('<gam>')
        P.xscale('log')
        P.xlim([0.05,0.7])
        P.title('Bins'+stile_args['bins_name'])
        P.legend()
        P.savefig(test.short_name+stile_args['bins_name']+'.png')
        P.clf()
        print "Done with binned test", stile_args['bins_name']
    for handle in set(handles):
        os.close(handle)
    for delete in set(deletes):
        if os.path.isfile(delete):
            os.remove(delete)

if __name__=='__main__':
    main()
    
