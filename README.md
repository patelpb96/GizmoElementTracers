# Description

Python package for reading and analyzing simulations that were generated using the Gizmo code, in particular, the FIRE cosmological simulations.


---
# Requirements

python 3, numpy, scipy, h5py, matplotlib.

This package also requires the [utilities/](https://bitbucket.org/awetzel/utilities) Python package for low-level utility functions.

We develop this package using python 3.9 and recommend that you use it with this package.


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
* generate cosmological zoom-in initial conditions from existing snapshot files

## snapshot_times.txt
* example file for storing information about snapshots: scale-factors, redshifts, times, etc

## gizmo_tutorial.ipynb
* comprehensive tutorial for using many features of this package (jupyter notebook)

## gizmo_tutorial_minimal.ipynb
* minimal/quick tutorial for using the basic features of this package (jupyter notebook)

---
# Units

Unless otherwise noted, all quantities are in (or converted to during read-in to) these units (and combinations thereof):

* mass [M_sun]
* position [kpc comoving]
* distance, radius [kpc physical]
* velocity [km / s]
* time [Gyr]
* elemental abundance [linear mass fraction]
* metallicity [log10(mass_fraction / mass_fraction_solar)], assuming Asplund et al 2009 for Solar


---
# Installing

## Installation via install_helper

The easiest way to install the analysis code and all dependencies is to navigate to the directory you would like the code to be placed in, and then to run the following two lines.

```
#!bash

git clone https://bitbucket.org/awetzel/gizmo_analysis.git
bash ./gizmo_analysis/install_helper.sh
```

## Instructions for placing in PYTHONPATH:

This is an alternative installation method.
This will not automatically install any dependencies.

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
git clone https://bitbucket.org/awetzel/gizmo_analysis.git
```

That is, you should end up with `$DIR/gizmo_analysis/gizmo_*.py`, with `$DIR` in your `$PYTHONPATH`

You then will be able to import gizmo_analysis.<whatever>

To update, cd into $DIR/gizmo_analysis and execute `git pull`.

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

Copyright 2014-2022 by:
* Andrew Wetzel <arwetzel@gmail.com>
* Shea Garrison-Kimmel <sheagk@gmail.com>
* Andrew Emerick <aemerick11@gmail.com>
* Zach Hafen <zachary.h.hafen@gmail.com>

If you use this package in work that you publish, please cite it, along the lines of: 'This work used GizmoAnalysis (http://ascl.net/2002.015), which first was used in Wetzel et al 2016 (https://ui.adsabs.harvard.edu/abs/2016ApJ...827L..23W).'

You are free to use, edit, share, and do whatever you want. But please cite it and report bugs!

Less succinctly, this software is governed by the MIT License:

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the 'Software'), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE aAUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
