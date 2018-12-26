"""
This script convert SQL based on S16a to S18a.
example:
$>python table_conversion.py  samples16a.sql samples18a.sql
The new quantities are changed to S18a standard except for
the "WHERE statement", where you need to change S16a manually
to S18a.
"""
import numpy as np
import re
import sys

fname =sys.argv[1]
fout  =sys.argv[2]
catname16f,names16f,catnamenewf,namesnewf = np.loadtxt('forced_tables',\
                                          comments='#',dtype='string',unpack=True)
catname16m,names16m,catnamenewm,namesnewm = np.loadtxt('meas_tables',\
                                          comments='#',dtype='string',unpack=True)
finput  = open(fname,"rt")
foutput = open(fout,"wt")
for line in finput:

   for ix in range(len(catname16m)): 
     found_m    = re.search(names16m[ix],line)
     if found_m is not None:
   	 line = line.replace(names16m[ix],namesnewm[ix])
         line = line.replace('_err','sigma')
   for iy in range(len(catname16f)): 
     found_f    = re.search(names16f[iy],line)
     if found_f is not None:
   	 line = line.replace(names16f[iy],namesnewf[iy])
         line = line.replace('_err','sigma')
   #print line.strip()
   foutput.write(line.strip())
   foutput.write('\n')
finput.close()
foutput.close()
