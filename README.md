# Description

Python package for running and analyzing Gizmo simulations.


# Contents

## gizmo_io.py
* functions to read in Gizmo snapshot files, which are in HDF5 format.

## gizmo_analysis.py
* functions for analyzing Gizmo particle data.

## gizmo_diagnostic.py
* functions for diagnosing Gizmo simulations.

## gizmo_file.py
* functions for deleting snapshot files or transferring them across machines.

## gizmo_ic.py
* functions for generating cosmological zoom-in initial conditions.

## tutorial.ipynb
* ipython notebook tutorial for using this package and reading in particle data from a snapshot.


# Requirements

This package relies on my [utilities/](https://bitbucket.org/awetzel/utilities) Python package for low-level utility functions.

I develop this package using the latest version of the Anaconda Python environment, which I update weekly.
I use Python 3.5 and recommend that you do the same.
However, I try to maintain backward compatibility with Python 2.7.


# Licensing

Copyright 2014-2016 by Andrew Wetzel.

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