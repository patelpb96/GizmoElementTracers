# Description

Python package for running and analyzing Gizmo cosmological simulations.


# Content

## gizmo_io.py
* routines to read in Gizmo snapshot file in HDF5 format.

## gizmo_analysis.py
* routines for analyzing particle data from a Gizmo simulation snapshot.

## agora.py
* ARCHIVE routines for analyzing AGORA simulation box with yt.

## gizmo_ic.py
* routines for generating cosmological zoom-in initial conditions.

## gizmo_transfer.py
* routines for transferring files/snapshots across machines.


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