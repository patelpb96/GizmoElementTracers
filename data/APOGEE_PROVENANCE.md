# APOGEE DR17 sample — provenance

`apogee_dr17_stellar_labels.csv` is a per-star APOGEE DR17 stellar-label table used as the
real Milky Way target for the SNe Ia delay-time-distribution inference in this repository.

## Source
- Retrieved 2026-07-28 from the public GitHub mirror
  `NolanKoblischke/SpectraFM_NeurIPS_FM4Science`, file
  `results/attention/stellar_labels.csv`
  (raw: https://raw.githubusercontent.com/NolanKoblischke/SpectraFM_NeurIPS_FM4Science/main/results/attention/stellar_labels.csv).
  That repository accompanies the SpectraFM spectral-foundation-model work; the file is the
  ASPCAP label table for its held-out APOGEE test stars.
- The underlying measurements are APOGEE DR17 / SDSS-IV ASPCAP stellar parameters and
  chemical abundances:
  - Abdurro'uf et al. 2022, ApJS 259, 35 (SDSS DR17)
  - Majewski et al. 2017, AJ 154, 94 (APOGEE)
  - García Pérez et al. 2016, AJ 151, 144 (ASPCAP)

## Columns
`TEFF`, `LOGG`, `O_FE`, `MG_FE`, `FE_H` — effective temperature (K), surface gravity (dex),
and [O/Fe], [Mg/Fe], [Fe/H] abundance ratios (dex, relative to solar). 5000 stars.

## Selection used here
`gizmo_mcmc.load_apogee_disk` applies giant-disk quality cuts (see `APOGEE_DEFAULT_CUTS`):
`0.5 < log g < 3.5`, `3500 < Teff < 5500 K`, `-1.2 < [Fe/H] < 0.5`, `-0.2 < [Mg/Fe] < 0.6`.
This keeps 3446 field disk giants that show the canonical high-alpha / low-alpha bimodality;
`apogee_analysis.py` characterizes the sample and fits the dual-Gaussian target.

## Note
This is a convenience subset committed for reproducibility under the network constraints of
this environment (only GitHub is reachable). For a definitive science analysis, pull the full
`allStar-dr17-synspec_rev1` value-added catalog from the SDSS SAS
(https://data.sdss.org/sas/dr17/apogee/spectro/aspcap/dr17/synspec_rev1/) and re-derive the
target with the same cuts.
