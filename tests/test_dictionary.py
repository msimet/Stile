import numpy as np
try:
    from stile import dictionary
except:
    import sys
    sys.path.append('..')
    from stile import dictionary


s16tags=['imag_cmodel','iflux_cmodel'] 
s17tags=['i_cmodel_mag','i_cmodel_flux'] 
#print dictionary._crossfind('S16A','S18A',s16tags)
print dictionary._crossfind('S17A','S16A',s17tags)

