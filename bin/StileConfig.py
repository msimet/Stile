#!/usr/bin/env python

try:
    import stile
except ImportError:
    import sys
    sys.path.append('..')
    import stile

def main(files):
    config_obj = stile.ConfigDataHandler(files)
    driver = stile.ConfigDriver()
    driver(config_obj)

if __name__=='__main__':
    import sys
    main(sys.argv[1:])
