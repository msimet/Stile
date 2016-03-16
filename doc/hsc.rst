==============
The HSC module
==============

Documentation for the HSC module of Stile.  This documentation file may be empty if you do not have the HSC or LSST pipelines installed.

These adapters are most easily run via command-line tasks.  Their arguments look like any other HSC/LSST pipeline task; for instance,
running on a single tract is

>>> StileTract.py $DATA_DIR --rerun=$rerun --id tract=$tract filter=$filter 

The current command-line executables are:

- **StileCCD.py**: run on one or more CCDs individually
- **StileCCDNoTract.py**: run on one or more CCDs individually, without using any tract-level information such as background fits
- **StileVisit.py**: run on one or more visits individually (that is, analyze all CCDs from a visit at once)
- **StileVisitNoTract.py**: run a visit without any tract-level calibrations applied
- **StileMultiVisit.py**: run on multiple visits analyzed as one data set
- **StilePatch.py**: run on one or more patches individually
- **StileTract.py**: run on one or more tracts individually
- **StileMultiTract.py**: run on multiple tracts analyzed as one data set

Systematics test adapters
=========================

.. automodule:: stile.hsc.sys_test_adapters
   :members:
   
Command-line tasks   
==================
.. automodule:: stile.hsc.base_tasks
   :members:   
