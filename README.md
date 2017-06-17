Instructions for install:  

1. create a directory $DIR
2. clone gizmo_analysis-SGK into $DIR
3. move setup.py from gizmo_analysis-SGK into $DIR and rename gizmo_analysis-SGK to gizmo_analysis
4. run python setup.py develop

In commands, that is:

```
#!bash

mkdir $DIR
cd $DIR
hg clone ssh://hg@bitbucket.org/sheagk/gizmo_analysis-sgk
mv gizmo_analysis-SGK/setup.py .
mv gizmo_analysis-SGK gizmo_analysis
python setup.py develop

```

You'll then be able to import gizmo_analysis.<whatever>


# Description

Python package for running and analyzing Gizmo simulations.


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

## tutorial.ipynb
* ipython notebook tutorial for using this package and reading particles from snapshots


# Requirements

This package relies on my [utilities/](https://bitbucket.org/awetzel/utilities) Python package for low-level utility functions.

I develop this package using the latest version of the Anaconda Python environment.
I use Python 3.6 and recommend that you do the same.
However, I try to maintain backward compatibility with Python 2.7.


# Licensing

Copyright 2014-2017 by Andrew Wetzel.

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