#

Standard Model + real singlet example, Z_2 symmetric potential. Run with
```python singletStandardModelZ2.py -h```
for info on usage and available options. There are command line flags for including the QCD gluon as an out-of-equilibrium particle,
and for doing a full recalculation of collision integrals instead of loading provided data files. 

##

This example is somewhat of a mess right now because we are not sure if the matrix elements produced by DRalgo are correct or not.
What is included now:

- MatrixElements/MatrixElements_QCD_BenoitBenchmark.txt contains QCD matrix elements in the form that Benoit used them ("Benoit matrix elements").
Only 3 processes, gluon cannot be off-eq.
- MatrixElements/MatrixElements_QCD.txt contains QCD matrix elements as I got from Andreas' DRalgo setup a long time ago ("DRalgo matrix elements").
It adds the tt -> tt process and processes relevant for off-eq gluons.
However it seems possible that they are not fully correct! Indeed they don't lead to quite the same wall speed as with Benoit's matrix elements

- Folder CollisionOutput_N5 is N=5 collision data generated using the DRalgo matrix elements. It has support for off-eq gluon, but works of course with just the top too.
- Folder CollisionOutput_N11 is like the N=5 version but has N=11. Currently these get loaded if the example is ran with --outOfEquilibriumGluon flag,
however remember that we are skeptical about the matrix elements.
- Folder CollisionOutput_N11_BenoitBenchmark is N=11 collision data generated using the Benoit matrix elements (but WallGoCollision code).
Off-eq gluon is not supported. This is supposed to be the "reliable" benchmark data,
therefore THIS IS WHAT GETS LOADED BY THE EXAMPLE BY DEFAULT, unless the --outOfEquilibriumGluon or --recalculateCollsions flag is passed. 

Once the matrix element stuff has been settled and we know what's actually correct, we should tidy this example by removing the "Benoit" files and keeping only
corrected matrix elements / collision data as produced by WallGo. Must also remove if statements in the .py that choose between "Benoit" or "DRalgo" versions of files.

The Makefile is outdated as the collision calculation stuff is now in the .py file(s).

PLEASE also keep this README updated if you change any data files (or matrix elements), so that we know what we're comparing to.