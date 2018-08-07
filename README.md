Instructions for install:  

1. create a directory $DIR
2. clone gizmo_analysis into $DIR
3. copy setup.py from gizmo_analysis into $DIR (moving it will give you hg issues when pulling updates in the future)
4. run python setup.py develop

In commands, that is:

```
#!bash

DIR=$HOME/code/wetzel_repos/
mkdir -p $DIR
cd $DIR
hg clone ssh://hg@bitbucket.org/sheagk/gizmo_analysis
cp gizmo_analysis/setup.py .
python setup.py develop
```

You'll then be able to import gizmo_analysis.<whatever>

To update the repo, cd into $DIR/gizmo_analysis and run hg pull && hg update.


# Description

Python package for running and analyzing Gizmo simulations.


---
# Requirements

This package relies on my [utilities/](https://bitbucket.org/sheagk/utilities) Python package for low-level utility functions.

I developd this package using Python 3.6 and recommend that you use that version. However, I have tried to maintain backward compatibility with Python 2.7.


---
# Contents

## gizmo_io.py
* read Gizmo snapshot files

## gizmo_analysis.py
* high-level analysis and plotting of Gizmo particle data

## gizmo_diagnostic.py
* run diagnostics on Gizmo simulations

## gizmo_file.py
* delete snapshot files or transfer them across machines

## gizmo_ic.py
* generate cosmological zoom-in initial conditions from Gizmo snapshot files

## gizmo_track.py
* track particles across snapshots

## submit_track_slurm.py
* script for submitting job for particle tracking

## gizmo_yield.py
* print/plot information about nucleosynthetic yields in Gizmo

## gizmo_tutorial.ipynb
* ipython notebook tutorial for using this package and reading particles from snapshots


---
# Units

Unless otherwise noted, all quantities are in (or converted to during read-in) these units (and combinations thereof):

* mass in [M_sun]
* position, distance, radius in [kpc physical]
* velocity in [km / s]
* time, age in [Gyr]
* elemental abundance in [(linear) mass fraction]
* metallicity in log10(mass_fraction / mass_fraction_solar), assuming Asplund et al 2009 for Solar


---
# License

Copyright 2014-2018 by Andrew Wetzel <arwetzel@gmail.com>.

In summary, you are free to use, edit, share, and do whatever you want. But please keep me informed. Have fun!

Less succinctly, this software is governed by the MIT License:

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.