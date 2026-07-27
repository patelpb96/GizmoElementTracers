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

## gizmo_mcmc.py
* infer the Maoz SNe Ia rate parameters (normalization and delay-time exponent) with MCMC, using the element-tracer forward model to map parameters onto stellar abundances ([Mg/Fe] or [alpha/Fe] vs [Fe/H])
* includes a bimodal (Milky-Way-like) star-formation-history weight generator and a post-processing routine to render a movie of the MCMC walkers converging (with an optional panel showing the abundance distribution evolving as the parameters change)
* supports two inference modes: a star-by-star likelihood against mock data, and a summary-statistic likelihood that matches a dual-Gaussian description (two sequences, each a mean and standard deviation along both axes) of the simulation to an external (real Milky Way) target -- the right posture when the model is imperfect
* includes a fast, factorized yield integrator (`integrate_rate_over_bins` / `fast_element_yields`) that replaces the per-element scipy.quad tabulation and is ~25x faster (agrees to ~1e-6); it works for any vectorized rate function, including a new tunable "kinked" Ia model (`ia_rate_kink`) -- a broken power-law delay-time distribution with a modulatable kink location and post-kink slope (reduces to Maoz when the slopes match)
* extra dependencies for the demos: `emcee` (MCMC), and `corner` + `imageio` + `imageio-ffmpeg` for the corner plot and the mp4 movie (all imported lazily with clear messages if missing)

## mcmc_maoz_demo.py
* end-to-end demo for gizmo_mcmc.py: generate a mock abundance data set and recover the input Maoz Ia parameters (run `python mcmc_maoz_demo.py`)

## mcmc_mw_bimodal_demo.py
* infer the Maoz Ia parameters that best describe a Milky-Way-like *bimodal* [alpha/Fe] vs [Fe/H] distribution offset ~1-2 sigma from the MW reference, and render a 30 fps movie of the walkers converging with the abundance distribution evolving (run `python mcmc_mw_bimodal_demo.py`)

## mcmc_mw_data_match_demo.py
* summary-statistic version: quantify the simulation and an external Milky Way target as dual Gaussians (mean and standard deviation along both axes for each of the two sequences) and walk the Maoz Ia parameters to best match the MW summary, reporting the residual mismatch per statistic; renders a movie of the simulation walking onto the MW target ellipses (run `python mcmc_mw_data_match_demo.py`)

## fast_yields_demo.py
* validate the fast yield integrator against the scipy.quad path, report the speedup, and plot the Ia rate models including the new tunable "kinked" model and how it reshapes the [alpha/Fe] vs [Fe/H] plane (run `python fast_yields_demo.py`)

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
