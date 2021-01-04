# Description

Python package for reading and analyzing Gizmo simulations.


---
# Requirements

python 3, numpy, scipy, h5py, matplotlib.

This package also requires the [utilities/](https://bitbucket.org/awetzel/utilities) Python package for low-level utility functions.

We develop and test this package using the latest version of Python (3.8).


---
# Contents

## gizmo_io.py
* read particles from Gizmo snapshot files

## gizmo_star.py
* models of stellar evolution as implemented in FIRE-2 and FIRE-3: rates and yields from supernovae (core-collapse and Ia) and stellar winds

## gizmo_agetracer.py
* generate elemental abundances in star and gas particles in post-processing, using the age-tracer module in FIRE-3

## gizmo_track.py
* track star and gas particles across snapshots

## gizmo_plot.py
* analyze and plot particle data

## gizmo_file.py
* clean, compress, delete, or transfer Gizmo snapshot files

## gizmo_diagnostic.py
* run diagnostics on Gizmo simulations

## gizmo_ic.py
* generate cosmological zoom-in initial conditions from snapshot files

## snapshot_times.txt
* example file for storing information about snapshots: scale-factors, redshifts, times, etc

## gizmo_tutorial.ipynb
* jupyter notebook tutorial for using this package


---
# Units

Unless otherwise noted, all quantities are in (or converted to during read-in) these units (and combinations thereof):

* mass [M_sun]
* position [kpc comoving]
* distance, radius [kpc physical]
* velocity [km / s]
* time [Gyr]
* elemental abundance [(linear) mass fraction]
* metallicity [log10(mass_fraction / mass_fraction_solar)], assuming Asplund et al 2009 for Solar


---
# Installing

This package functions either as a subfolder in your `$PYTHONPATH` or by installing it with `setup.py develop`, which should place an egg.link to the source code in a place that whichever `python` you used to install it knows where to look.

## Instructions for placing in PYTHONPATH:

1. create any directory $DIR
2. add $DIR to your `$PYTHONPATH`
3. clone gizmo_analysis into $DIR

In commands, that would be something like:
```
#!bash

DIR=$HOME/code
echo $PYTHONPATH=$DIR:$PYTHONPATH >> ~/.bashrc
mkdir -p $DIR
cd $DIR
git clone git@bitbucket.org:awetzel/gizmo_analysis.git
```

That is, you should end up with `$DIR/gizmo_analysis/gizmo_*.py`, with `$DIR` in your `$PYTHONPATH`

You then will be able to import gizmo_analysis.<whatever>

To update, cd into $DIR/gizmo_analysis and execute `git pull`.


## Instructions for installing as a package:

1. create any directory $DIR
2. clone gizmo_analysis into $DIR
3. copy setup.py from gizmo_analysis into $DIR
4. run python setup.py develop

In commands, that is:

```
#!bash

DIR=$HOME/code/
mkdir -p $DIR
cd $DIR
git clone git@bitbucket.org:awetzel/gizmo_analysis.git
cp gizmo_analysis/setup.py .
python setup.py develop
```


---
# Using

Once installed, you can call individual modules like this:

```
import gizmo_analysis
gizmo_analysis.gizmo_io
```

or more succinctly like this

```
import gizmo_analysis as gizmo
gizmo.io
```


---
# License

Copyright 2014-2021 by Andrew Wetzel <arwetzel@gmail.com>, Shea Garrison-Kimmel <sheagk@gmail.com>, and Andrew Emerick <aemerick11@gmail.com>.

If you use this package in work that you publish, please cite it, along the lines of: 'This work used GizmoAnalysis (http://ascl.net/2002.015), which first was used in Wetzel et al 2016 (https://ui.adsabs.harvard.edu/abs/2016ApJ...827L..23W).'

You are free to use, edit, share, and do whatever you want. But please keep us informed and report bugs. Have fun!

Less succinctly, this software is governed by the MIT License:

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the 'Software'), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE aAUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
