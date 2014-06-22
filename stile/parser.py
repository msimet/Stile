
parser_dict = 
    {'config': stile.ConfigParser}

    
 def Parser(args):
    """
    Returns an argparse Parser object with input args used by Stile and corr2.
    """
    import corr2_utils
    import argparse
    parents = [corr2_utils.Parser()]
    if '--data-handler' in args:
        dh_type = args[args.index('--data-handler')+1]
        if dh_type in parser_dict:
            parents.append(parser_dict[dh_type]())
    p = argparse.Parser(parents=parents)
    p.add_argument('config_file',nargs='*',help='Stile configuration file(s)')
    stile_args = vars(p.parse_known_args(args)) # argparse returns a namespace
    _AddConfig(stile_args) 
    corr2_utils.CheckArguments(stile_args)
    corr2_utils.AddCorr2Dict(stile_args)
    if stile_args['data_handler']=='config':
        stile.ConfigDataHandler(stile_args)
    else:
        raise NotImplementedError('No data handler of type %s implemented'%(
                                   stile_args['data_handler']))
    return stile_args

def _AddConfig(stile_args):    
    if stile_args.config_file:
        if isinstance(stile_args['config_file'],str):
            stile_args['config_file'] = [stile_args['config_file']]
        if not hasattr(stile_args['config_file'],'__iter__'):
            raise ValueError('stile_args.config_file is neither a string nor a list/tuple')
        import os
        config_file_type = [os.path.splitext(f)[1] for f in stile_args.config_file]
        try:
            import yaml
            config_reader = yaml
        except:
            if '.yaml' in config_file_type:
                raise ImportError('YAML config file specified, but no yaml file reader found')
            import json
            config_reader = json
        config_list = [config_reader.load(f) for f in stile_args.config_file]
        for config_file in config_list:
            stile_args.update(config_file)
    
   
