#!/usr/bin/env python

from distutils.core import setup
try:
    import treecorr
except ImportError:
    import warnings
    warnings.warn("treecorr package cannot be imported. Installation will proceed, but you may "+
                  "wish to install it if you would like to use the correlation functions within "+
                  "Stile.")


setup(name='Stile',
      version='0.1',
      description='Stile: Systematics Tests in Lensing pipeline',
      author='The Stile team',
      requirements=['numpy'],
      author_email='melanie.simet@gmail.com',
      url='https://github.com/msimet/Stile',
      packages=['stile', 'stile.hsc'],
      scripts=['bin/StileVisit.py', 'bin/StileVisitNoTract.py', 'bin/StileCCD.py',
               'bin/StileCCDNoTract.py', 'bin/StilePatch.py', 'bin/StileTract.py']
      )
