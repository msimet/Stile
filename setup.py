#!/usr/bin/env python
from setuptools import setup
from io import open

# read the contents of the README file
with open('README.md', encoding="utf-8") as f:
    long_description = f.read()

setup(name='Stile',
      version='0.1',
      description='Stile: Systematics Tests in Lensing pipeline',
      author='The Stile team',
      install_requires=['numpy', 'treecorr', 'pyfits', 'matplotlib', 'astropy<3'],
      author_email='melanie.simet@gmail.com',
      url='https://github.com/msimet/Stile',
      packages=['stile', 'stile.hsc'],
      scripts=['bin/StileVisit.py', 'bin/StileVisitNoTract.py', 'bin/StileCCD.py',
               'bin/StileCCDNoTract.py', 'bin/StilePatch.py', 'bin/StileTract.py'],
      test_suite='nose.collector',
      tests_require=['nose'],
      classifiers=[
          'Development Status :: 4 - Beta',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved',
          'Operating System :: MacOS :: MacOS X',
          'Operating System :: POSIX :: Linux',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Topic :: Scientific/Engineering :: Physics'
      ]
      )
