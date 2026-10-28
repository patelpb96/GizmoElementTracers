# FIRE-2 m09_res30 star particles at z = 0 — provenance

`output/snapdir_600/snapshot_600.0.hdf5` holds **all 1349 star particles** (`PartType4`, every
dataset unchanged) of the z = 0 snapshot of the public FIRE-2 ultra-faint dwarf simulation
`m09_res30` (baryonic particle mass ~30 Msun). The original `Header` attributes are kept, with
`NumPart_ThisFile` / `NumPart_Total` set to the star count and `NumFilesPerSnapshot` = 1, so
`gizmo_io.Read.read_snapshots('star', 'index', 600, simulation_directory=...)` reads it like any
snapshot. `snapshot_times.txt` is the run's own file.

## Source
- FIRE-2 public data release (Wetzel et al. 2023, ApJS 265, 44; DR2: Wetzel et al. 2025),
  Core suite, `core/m09_res30/output/snapdir_600/snapshot_600.{0..7}.hdf5` (~16.6 GB), via
  https://users.flatironinstitute.org/~mgrudic/fire2_public_release/core/m09_res30/
  (also on FlatHUB: flathub.flatironinstitute.org/fire). Retrieved 2026-10-28.
- Simulation introduced in Wheeler et al. 2019, MNRAS 490, 4447 ("Be it therefore resolved") —
  cite it when using these data.
- Only the star datasets were read, with HTTP byte-range requests (~30 MB transferred); gas,
  dark matter and the rest of each file were never downloaded.

## Contents
`Metallicity` has 15 columns: 0 Z, 1 He, 2 C, 3 N, 4 O, 5 Ne, 6 Mg, 7 Si, 8 S, 9 Ca, 10 Fe,
11-14 r-process passive scalars (all zero in this run). **No age-tracers**: the run (2018)
predates them, and its `gizmo_parameters.txt` has no AgeTracer settings. The abundances are the
in-code FIRE-2 values (Mannucci 2006 Ia DTD).

The run's `notes.txt` lists corrupt snapshots (316-320, 370-371, 380-381, 433-438, 483-484,
498-499, 571-572, 585); snapshot 600 is not among them.
