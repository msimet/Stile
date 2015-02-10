#!/usr/bin/env python

from distutils.core import setup

setup(name='Stile',
      version='0.1',
      description='Stile: Systematics Tests in Lensing pipeline',
      author='The Stile team',
      author_email='melanie.simet@gmail.com',
      url='https://github.com/msimet/Stile',
      packages=['stile', 'stile.hsc'],
      scripts=['bin/StileVisit.py', 'bin/StileVisitNoTract.py', 'bin/StileCCD.py',
               'bin/StileCCDNoTract.py', 'bin/StilePatch.py', 'bin/StileTract.py']
      )
