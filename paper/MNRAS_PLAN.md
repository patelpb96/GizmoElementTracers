# Plan for an MNRAS paper: inferring the SN Ia delay-time distribution from the Milky Way's α-bimodality with FIRE age-tracers

Status: planning document (no prose drafted yet). It covers what the paper should claim, what is
still missing before those claims hold, a section-by-section outline with the equations and
figures each section needs, a validation matrix, and a citation map.

The organising rule: **every step from a gas cell's age-tracer array to a posterior over DTD
parameters is written out as an equation, a figure, or a numbered list.** Very few people have
used the element tracers, and the method itself is still cited as "in prep." (see §0.2), so a
referee cannot be assumed to know any of it.

---

## 0. Framing decisions to make first

### 0.1 One-sentence thesis (draft)

> Because FIRE age-tracers record *when* (in stellar age) every gas cell was enriched, the
> [Mg/Fe]–[Fe/H] distribution of a simulated galaxy can be recomputed for any SN Ia delay-time
> distribution (DTD) at fixed total Ia number, in seconds and without rerunning the simulation.
> This turns a FIRE galaxy into a forward model that can be sampled with MCMC against APOGEE.
> We use it to ask which DTD shapes are consistent with the Milky Way's high-/low-α sequences,
> and which are ruled out.

### 0.2 Method paper, science paper, or both?

What the literature says about the tracers:

- The FIRE-3 paper (Hopkins et al. 2023, MNRAS 519, 3154, §3.1.2) says FIRE-3 runs by default
  include "the abundance 'tracers' (following an additional set of passive scalars to model
  trace metal species) model of **Wetzel et al., in prep.**"
- Hopkins' FIRE-2.x change notes attribute the same model to **Emerick et al., in prep.**
- FIRE-3 §3.11.1 explicitly defers Ia-yield variations to "our tracer-element approach in
  post-processing", but the approach itself has no refereed description.
- `gizmo_agetracer.py` lists Andrew Wetzel and Andrew Emerick as authors.
  `FIREYieldClass2` / `gizmo_model.py` are Preet Patel's.

So the method paper does not exist yet. That gives two options:

| Option | What it is | Pros | Cons |
|---|---|---|---|
| **A (recommended)** | Paper I: the age-tracer method (definition, validation against in-code abundances, resolution/bin tests, fast integrator) plus a first application (DTD inference vs APOGEE). Paper II: a multi-galaxy suite, more elements, metallicity-dependent rates. | Delivers the reference the community is missing. A referee can check every step. Clean authorship story with Wetzel/Emerick as method co-authors. | Longer paper (~20 pp). Needs explicit agreement with the in-prep authors. |
| B | Application only, citing "Wetzel et al. in prep." for the method | Shorter, faster | Unverifiable core method. Referees will ask for the Appendix A material anyway. |

**Action:** settle authorship and scope with A. Wetzel / A. Emerick before drafting §2. Option A
assumes they agree to fold the method description in.

### 0.3 Candidate titles

- "Rewinding the clock on Type Ia supernovae: inferring the delay-time distribution from the
  Milky Way's α-bimodality with age-tracers in FIRE simulations"
- "Age-tracers in FIRE: post-processing nucleosynthesis to constrain the SN Ia delay-time
  distribution with APOGEE"

---

## 1. Gap analysis: what the current code shows vs what MNRAS needs

The repository now has a working, fast, end-to-end pipeline, validated on mocks, run on 10 DTD
families, with movies and a leaderboard (`results/apogee_dtd_conserved/`). These items still have
to change before the results count as a scientific claim:

| # | Current state | Needed for the paper | Blocking? |
|---|---|---|---|
| G1 | Age-tracer weights are **mock** (`generate_bimodal_weights`: hand-built high-/low-α SFH profiles) | Weights read from a real FIRE run with `GALSF_FB_FIRE_AGE_TRACERS` (16 log bins, 1–13 700 Myr), star particles selected to match APOGEE (solar annulus / radial range, \|z\|, age range) | **Yes**: every current number depends on the mock SFH |
| G2 | APOGEE = 5000-star label subset from a GitHub mirror (3446 after cuts) | Full DR17 `allStar` (synspec_rev1) with ASPCAP flags (`ASPCAPFLAG`, `STARFLAG`, `EXTRATARG`), S/N > 70, duplicate removal, a spatial cut (e.g. 5 < R < 11 kpc, \|z\| < 2 kpc via Gaia/StarHorse/astroNN distances), and a statement on the selection function | **Yes** |
| G3 | Summary σ vector is hand-set (`DEFAULT_SUMMARY_SIGMA`) | σ derived as the quadrature sum of (i) APOGEE bootstrap scatter of each statistic, (ii) simulation particle-sampling noise (bootstrap of sim stars at fixed θ), (iii) an explicit model-discrepancy floor, plus the 9×9 covariance, not a diagonal | **Yes**: χ² values and posterior widths depend on it |
| G4 | Leaderboard ranks by χ² at the **posterior median** | Max-likelihood χ², Bayesian evidence (dynesty / nested sampling) or WAIC/LOO, and the effective number of parameters. Nested models must never score worse than their parent (as `maoz_onset` vs `maoz` currently do: 29.8 vs 26.2, a median-vs-max artefact) | **Yes** |
| G5 | Zero-point offsets d[Fe/H] ≈ +0.5, d[Mg/Fe] ≈ +0.05–0.10 sampled as nuisance | Check them against the real sim: d[Fe/H] = 0.5 dex is far larger than APOGEE's systematic zero-point (~0.05 dex) and must be explained (mock SFH? yield normalisation? solar scale?) or replaced by a physically motivated yield-scaling parameter. Report results with and without the offsets | **Yes** |
| G6 | No closure test | With FIRE's own in-code DTD, rates and yields, the tracer-derived abundances must reproduce the abundances the simulation tracked in-code, star by star (see §5, V1) | **Yes**: the method's credibility rests on it |
| G7 | 10 age bins in the demos | Production runs have 16. Show how the inferred DTD changes with bin count (re-bin a high-resolution run, or use a run with custom bins) and state which DTD features are below the resolution limit | Partially |
| G8 | Posterior median for `mannucci_prompt` puts ~81 % of Ia at t ≈ 43 Myr, i.e. right after the last CCSNe | Treat it as a physical red flag, not a win. It is degenerate with CCSN Fe yield / the CC [Mg/Fe] plateau. Test it with a yield-uncertainty nuisance parameter and against observed prompt fractions (Maoz & Mannucci 2012; Maoz & Graur 2017; Strolger et al. 2020) | Discussion-critical |
| G9 | Conservation fixes the Ia **mass-loss** integral N = 2.2526×10⁻³ M☉/M☉ (= 1.4 M☉ × 1.609×10⁻³ Ia/M☉ from 37.53 Myr to 13.7 Gyr) | State this explicitly. Note it matches the FIRE-3 value of 1.6×10⁻³ Ia/M☉ over a Hubble time (Hopkins et al. 2023 §3.3.4). Justify conservation physically: total Ia energy injected is fixed, so the run's feedback history stays (to first order) self-consistent; see §2.6 | Writing only |
| G10 | Single element ratio ([Mg/Fe]) | At least add [O/Fe] or [Si/Fe] as a consistency check. APOGEE O, Si and Mg have different systematics, and the Ia vs CC yield ratios differ by element | Strongly recommended |

**Recommended order:** G6 → G1 → G2 → G3 → G4 → G5 → G7 → G10. Rerun the 10-family suite only
once G1–G4 are done; the current results become the "mock-SFH demonstration" appendix.

---

## 2. Section-by-section outline

Target length: ~15 pages main text plus ~6 pages of appendices. MNRAS style (`mnras.cls`),
serif figures (already the house style in the code), single-column figures where possible.

### Abstract (≤250 words)

Five sentences:

1. The problem: the SN Ia DTD sets the timing of Fe enrichment, but simulations fix it at run time.
2. The method: age-tracers plus a fast factorised integrator let any DTD be applied in
   post-processing at fixed Ia number.
3. The data: APOGEE DR17 high-/low-α sequences, summarised as a 9-statistic dual Gaussian.
4. The result: which families and parameter ranges are allowed, quoted with an evidence ratio.
5. The caveat or outlook: feedback is held fixed; the method generalises to other channels
   and elements.

### 1. Introduction (~1.5 pp)

1. **The DTD as a progenitor and chemical-evolution constraint.** Observational DTDs from cosmic
   rates and SFH-resolved hosts: Mannucci, Della Valle & Panagia 2006; Totani et al. 2008;
   Maoz & Mannucci 2012; Maoz, Mannucci & Nelemans 2014; Maoz & Graur 2017;
   Strolger et al. 2020; Wiseman et al. 2021 (DES); Chen et al. 2021. Theory: power law
   ∝ t⁻¹ from GW-driven double-degenerate inspiral (Greggio 2005); recent review Ruiter &
   Seitenzahl 2025 (arXiv 2412.01766).
2. **The α-bimodality as a clock.** APOGEE: Hayden et al. 2015; Nidever et al. 2014;
   Weinberg et al. 2019; Abdurro'uf et al. 2022 (DR17). In one-zone or multi-zone GCE the DTD
   shapes the knee and the low-α sequence (Weinberg, Andrews & Freudenburg 2017; Dubay et al.
   2024, who find GCE favours an *extended* DTD; Palla 2021).
3. **Bimodality in FIRE.** Barry et al. 2026 (arXiv 2601.02520) find strong bimodality in ~¼ of
   16 FIRE-2 MW-mass hosts, driven by a rapid decline in star formation at low gas fraction.
   Parul et al. 2025 (arXiv 2501.12342) find gas accretion drives it in 11 hosts, with
   Romeo/Juliet/Romulus/Remus/Louise bimodal. The "Portrait of a Galaxy on FIRE" paper on Romeo
   (arXiv 2512.14897) finds the sequences less separated in [Mg/Fe] than observed. Parul et al.
   2023 (MNRAS 520, 1672) report FIRE-2 [O/Fe] higher than observed and attribute it to the Ia
   yields/DTD. These motivate asking whether a different DTD alone could fix the separation.
4. **Why not just rerun?** Gandhi et al. 2022 (MNRAS 516, 1941) recompute Ia *rates* in
   post-processing but had to **re-simulate** to get the abundances. Each FIRE MW-mass run costs
   ~10⁶ CPU-h. Age-tracers sit in between: abundances recomputed self-consistently with the
   simulated gas flows and mixing, with feedback fixed.
5. **This paper:** (i) the first full description and validation of the tracers (if Option A);
   (ii) a fast integrator; (iii) DTD inference against APOGEE at fixed Ia number.
   End with a roadmap paragraph.

### 2. Method I: FIRE simulations and the age-tracer model (~4 pp; the heart of the paper)

Write it so a reader who has never opened GIZMO could reimplement it. Use subsections
2.1–2.7, each ending with a one-line "in practice" pointer to the code (as footnotes, or a
code-availability table in §8).

**2.1 Simulation(s).**
- GIZMO MFM hydro (Hopkins 2015) and the FIRE-2 physics (Hopkins et al. 2018) or FIRE-3
  (Hopkins et al. 2023): name which, and the exact version (FIRE-2.1 / 2.x vs FIRE-3).
- Host(s), with mass resolution, z = 0 stellar mass, and why chosen (bimodal per
  Parul/Barry?).
- Turbulent metal diffusion (Hopkins 2018 App. F3; Escala et al. 2018): the tracers are passive
  scalars that diffuse **exactly like** the metals, so mixing is captured.
- **To confirm:** which existing runs carry age-tracers. The FIRE-2 public release (Wetzel et al.
  2023, ApJS 265, 44) has 11 tracked elements but, as far as I can tell, no age-tracer fields.

**2.2 What an age-tracer weight is.** Lift the definition from `gizmo_agetracer.py` and
formalise it.
- Age bins: edges {τ_0 … τ_B}, by default B = 16 log-spaced over [1, 13 700] Myr
  (`AgeTracerBinStart/End`; custom lists via `GALSF_FB_FIRE_AGE_TRACERS_CUSTOM`).
- At each deposition step, star particle *s* of age τ_s ∈ bin *b* adds to neighbour gas cell *g*:

  Δw_{g,b} = m_s · f_{s→g} / m_g · (Δt / Δτ_b)

  Here f_{s→g} is the same geometric coupling fraction the feedback uses. The normalisation is
  stated so that w = 1 means "received ejecta-coupling equal to its own mass for a full bin
  duration". **Write this equation with Wetzel/Emerick** and confirm the Δt/Δτ_b factor and
  how `AgeTracerActiveTimestepFraction` (stochastic deposition targeting N events per bin)
  enters.
- Star particles inherit their parent gas cell's w at formation.
- Surface return (default) vs `…_DISABLE_SURFACE_YIELDS`: with surface return, weights are
  conserved through recycling.
- **Figure 1 (schematic):** a star particle ageing through bins; at each step it "paints" its
  neighbours' bin-b weight; a gas cell's weight vector; a new star inheriting it. Use a
  three-panel cartoon plus a real w-vector from one particle.

**2.3 From weights to abundances.**

  X_{e,i} = X_{e,i}^{init} + Σ_b w_{i,b} · Y_{e,b},   with   Y_{e,b} = ∫_{τ_b}^{τ_{b+1}} Σ_c y_{e,c} R_c(τ) dτ

- R_c is the mass-loss rate per unit IMF-averaged stellar mass for channel c ∈ {winds, CCSN, Ia}.
- y_{e,c} is the yield mass fraction of element e in channel c.
- The units chain must be explicit: R_c in M☉ M☉⁻¹ Myr⁻¹, so Y is a mass fraction and X is a
  mass fraction.
- Then [X/H] = log10(X_e/X_H) − log10(X_e/X_H)_☉ with Asplund et al. 2009. State that the code
  uses log10(X_e / X_{e,☉}) (mass-fraction ratios, as in the README) and quantify any difference
  from the H-normalised definition.
- **The one approximation, stated as an equation.** The tracer replaces ∫ ẇ(τ) ψ(τ) dτ by
  Σ_b ⟨ẇ⟩_b ∫_b ψ dτ, i.e. the deposition is assumed uniform within a bin. The error is
  non-zero only when the coupling to a given gas cell *and* ψ both vary inside one bin. This is
  exactly the regime of a narrow prompt Ia peak inside a wide bin, so it motivates G7.
- Initial abundances X^{init}: the metallicity floor or pre-enrichment.
- The metallicity-dependent variant `ElementAgeTracerZClass` if used. Otherwise state that
  progenitor-metallicity dependence of the CC/wind yields is neglected.

**2.4 Rates and yields (Table 1).** List every channel exactly as implemented (FIRE-2.1 in
`gizmo_model.py`).
- CCSN: piecewise-constant rate with breaks at 3.4, 10.37, 37.53 Myr; normalisations 5.408×10⁻⁴
  and 2.516×10⁻⁴; ejecta 10.5 M☉; yields Mg 9.40×10⁻³, Fe 7.06×10⁻³, O 0.111. Note that these
  set the high-α plateau: [Mg/Fe]_CC ≈ +0.4 with Asplund solar.
- Winds: breaks at 1, 3.5, 100 Myr; yields He, C, N, O only.
- Ia: Maoz with n_Ia = 2.6×10⁻⁷ Myr⁻¹ M☉⁻¹ at 1 Gyr, slope −1.1, onset 37.53 Myr, ejecta 1.4 M☉;
  Iwamoto 1999 W7 yields (Fe 0.531, Mg 6.14×10⁻³).
- Compare to FIRE-3: Ia onset 44 Myr, a = 0.0083 Gyr⁻¹ M☉⁻¹ at onset, slope −1.1; Leung &
  Nomoto 2018 W7/WDD2 mean yields (Fe 0.558, Mg 6.21×10⁻³). Fe is ~5 % higher, about
  0.02 dex. **Say which set the paper uses.** The code currently uses FIRE-2.1 yields.
- Mannucci 2006 as implemented in FIRE-2: 5.3×10⁻⁸ + 1.6×10⁻⁵ exp(−½((τ−50)/10)²) Myr⁻¹ M☉⁻¹.

**2.5 The fast factorised integrator.** Y_{e,b} = Σ_c y_{e,c} ∫_b R_c dτ separates the
element-independent rate integrals from the yields. Integrate each R_c once on a dense log grid
augmented with bin edges and rate breakpoints (cumulative trapezoid), then contract with y.
Report:
- cost per likelihood (vs `scipy.quad` per element: ~25×);
- accuracy (~10⁻⁶ relative);
- that it works for any vectorised R(τ).

Appendix B gives the convergence test.

**2.6 Varying the DTD at fixed Ia number.**
- Define the constraint: ∫_{t_Ia}^{t_H} R_Ia dτ = N_0, with N_0 = 2.2526×10⁻³ M☉ of Ia ejecta
  per M☉ formed (1.609×10⁻³ events M☉⁻¹, consistent with FIRE-3's 1.6×10⁻³).
- Implementation: all amplitude parameters are rescaled by one common factor, so shape
  parameters (and component *shares*) are free and normalisation is not.
- **Why:** (i) the number of Ia per unit mass is constrained to ~30–50 % by cosmic rates;
  (ii) it holds the total Ia energy, and so the simulated feedback history, fixed to first
  order. That is what makes post-processing legitimate.
- **What is not captured:** redistribution of Ia energy in time (e.g. more Ia at 50–100 Myr
  overlapping CCSN-driven outflows). Estimate its size by comparing to Gandhi et al. 2022's
  re-simulations: Maoz vs Mannucci changed stellar mass and morphology little.

**2.7 DTD families (Table 2).** For each family give R(τ), its free parameters, priors, the
literature motivation, and its N-conserved share definition. Use the current `IA_MODEL_SPECS`:
maoz, maoz_onset, kink, prompt_delayed, long_delay, skewnorm (Strolger 2020; note the 1/(τ ln10)
Jacobian), exponential, mannucci (fixed), mannucci_prompt, and the new peak_growth family. For
peak_growth, define t_tail as the first age past the peak's maximum at which it drops to 1 % of
the maximum. **Figure 2:** all families at their prior medians on one log-log plot, with the
16 age-bin edges marked so the reader sees what the tracers can resolve.

### 3. Method II: the observational target (~1.5 pp)

**3.1 APOGEE DR17 sample (resolves G2).**
- Cuts as a numbered list, with star counts after each step (a cut-flow table).
- Giants: the current cut is 0.5 < log g < 3.5. Consider raising the lower bound to ~1.0, where
  ASPCAP abundance systematics for the most luminous giants grow, and state the reason either
  way. Keep 3500 < T_eff < 5500 K.
- Abundance and spatial ranges.
- **Figure 3:** the APOGEE [Mg/Fe]–[Fe/H] plane with the 2-Gaussian decomposition, plus
  [Mg/Fe] slices in [Fe/H] bins (the current `apogee_analysis.py` figure, upgraded).

**3.2 Matching simulation and survey.**
- Select simulated stars in the same R and |z| ranges, in the host's disc frame.
- Weight by an APOGEE-like age/selection proxy, or argue it is unnecessary for giants.
- Convolve the simulated abundances with APOGEE errors (~0.03–0.05 dex).
- Say explicitly why the comparison is distributional (no star-by-star correspondence).

**3.3 Summary statistic.**
- A 2-component diagonal GMM (EM, fixed iteration count) → 9 statistics: f_high; and for each
  sequence the mean and σ of [Fe/H] and of [Mg/Fe].
- Justify: robust to selection-function normalisation, insensitive to particle-count noise,
  and directly encodes "two sequences with positions and widths".
- Show that it is a deterministic, smooth function of θ (the EM is initialised identically
  every call; otherwise MCMC sees label-switching noise).
- Mention alternatives tested or considered (2-D histograms with a Poisson likelihood;
  sliced-Wasserstein; ABC) and why not chosen.

**3.4 Likelihood.**
- ln L = −½ (s(θ) − s_obs)ᵀ C⁻¹ (s(θ) − s_obs), with C = C_APOGEE + C_sim + C_model (G3).
- Zero-point nuisances d[Fe/H], d[Mg/Fe] with priors set by APOGEE/solar-scale systematics
  (G5).

### 4. Method III: inference (~1 pp)

- emcee (Foreman-Mackey et al. 2013) affine-invariant ensemble: 48 walkers × 3000 steps,
  1000 burn-in.
- Report integrated autocorrelation times τ_int, N_steps / τ_int > 50, and acceptance fractions.
- Flat priors (Table 2), log-amplitudes.
- Initialisation deliberately displaced (−1σ in [Fe/H], +1σ in [Mg/Fe]) to show the
  convergence is not set by the initial position.
- Model comparison with nested sampling (dynesty; Speagle 2020) for ln Z, and a cross-check
  with WAIC/LOO.
- Compute cost per likelihood evaluation and total CPU time. This is a selling point.

### 5. Validation (~2 pp; this is what convinces a referee)

| Test | What it shows | Figure / table |
|---|---|---|
| **V1 Closure** | Fiducial (in-code) DTD + yields through the tracers reproduces the simulation's directly tracked [Fe/H], [Mg/Fe] for every star. Residual distribution, median and 1st/99th percentile. Explains residuals as bin-discretisation + stochastic deposition | Fig. 4: tracer vs in-code scatter + residual histogram |
| V2 Integrator | Fast integrator vs `scipy.quad` across all families and parameters at the prior edges | App. B figure |
| V3 Bin resolution | Re-derive the posterior with 8 / 16 / (32) bins (re-binning) | Fig. or App. C |
| V4 Injection–recovery | Generate synthetic "observations" from the sim at known θ_true (per family), recover θ with the full pipeline including noise. Coverage test: the fraction of 68 % intervals containing the truth over ~50 injections | Fig. 5 (P–P plot) |
| V5 Prior sensitivity | Widen/narrow priors (the earlier caps at a 50 % prompt share mattered) | Table in appendix |
| V6 Summary-statistic stability | GMM iterations (20 vs 40), initialisation, outlier clipping | App. |
| V7 Event conservation | N(θ) / N_0 − 1 < 10⁻¹² across the chain (currently ~10⁻¹⁹) | One sentence |
| V8 Zero-point | Results with offsets fixed to 0, sampled, or replaced by a yield-scaling parameter | Table 4 |

The existing mock demos (`mcmc_maoz_demo.py`, `mcmc_kink_demo.py`) are a proto-V4 and can be
cited as such.

### 6. Results (~3 pp)

1. **The fiducial FIRE DTD vs APOGEE.** Where the sim sits relative to MW (Fig. 6: sim
   contours over APOGEE density, with dual-Gaussian ellipses). Quantify the offset per
   statistic.
2. **What the DTD can and cannot move** (Fig. 7, a sensitivity "Jacobian" figure): ∂s/∂θ for
   each statistic and family. Which of the 9 statistics respond to the DTD at all? Current
   finding: the DTD moves the summary by ≲0.3σ on its own, and the low-α [Mg/Fe] width is
   insensitive. That is a *result*, not a nuisance: it identifies the observables a DTD
   inference can use.
3. **Posteriors per family** (Fig. 8: corner plots for Maoz, skew-normal, peak_growth,
   mannucci_prompt; the rest in an appendix). Fig. 9 shows the posterior-predictive DTD band for
   each family, overlaid on observational DTD measurements (Maoz & Graur 2017; Strolger 2020;
   Wiseman 2021 points), with the bin edges.
4. **Model comparison** (Table 3): ln Z, Δln Z vs Maoz, max-L χ², number of free parameters,
   prompt share (fraction of Ia at τ < 100 or < 420 Myr, to match observational conventions),
   and posterior on the Maoz slope.
5. **Posterior-predictive check** (Fig. 10): the best model's [Mg/Fe]–[Fe/H] plane and slices vs
   APOGEE, with residual maps. Show what *no* DTD fixes.
6. **Convergence movie**: a supplementary online material item (MNRAS hosts videos). One still
   frame goes in the paper (Fig. 11 or appendix).

Pre-registered questions (decide before rerunning, so results aren't read post hoc):

- Is the Maoz slope consistent with −1.1? (Current mock-SFH value: −1.49 +0.15/−0.08.)
- Is a prompt component > X % required or excluded?
- Does any late-time growth (k_grow > 0) get support?
- Is the FIRE-2 Mannucci DTD disfavoured relative to FIRE-3 Maoz at fixed N?

### 7. Discussion (~2 pp)

- **Degeneracies with yields.** The CC [Mg/Fe] plateau and the Ia Fe yield shift the sequences
  in the same direction as the DTD (G8). Show a 2-D marginal of prompt share vs a CC-Fe or
  Ia-Fe yield multiplier, if added as nuisances.
- **What is held fixed.** Feedback timing and SFH (justify via conservation and Gandhi 2022).
  Metal mixing is *not* held fixed; it is captured by the tracers. Metallicity-independent DTD
  (connect to Gandhi 2022's Z-dependent rates as future work, using `ElementAgeTracerZClass`).
- **Host-to-host variance.** One galaxy's SFH is not the MW's. Either run ≥3 bimodal hosts
  (Parul 2025 / Barry 2026 lists) and report the spread, or state this clearly as the leading
  systematic.
- **Comparison to GCE inferences** (Dubay 2024 extended DTD; Weinberg 2017) and to
  dwarf-spheroidal DTD constraints ("It's a matter of time", arXiv 2602.22333).
- **Implications for FIRE-3 / next-generation runs.** Should the in-code DTD change?
- **Other uses of the tracers:** r-process (NSM DTDs), AGB yields, Ia sub-classes with different
  yields (Kirby 2019; de los Reyes 2020; FIRE-3 §3.11.1 explicitly anticipates this).

### 8. Summary (numbered list of 5–7 findings)

### Data / code availability (required by MNRAS)

- GizmoAnalysis (ASCL 2002.015; Wetzel et al. 2016 citation as the README requests).
- This repository at a tagged release.
- FIRE data (flathub.flatironinstitute.org/fire) or the statement that tracer snapshots are
  available on request.
- APOGEE DR17 (SDSS SAS).
- A code-availability table mapping each paper equation to its function
  (e.g. Eq. 3 → `ElementAgeTracerClass.get_element_massfractions`; Eq. 5 →
  `integrate_rate_over_bins`; Eq. 7 → `MaozElementTracerModel._ia_kwargs_from_params`).

### Appendices

- **A.** Age-tracer implementation details in GIZMO: Config/param flags, the deposition
  algorithm, the stochastic timestep fraction, and the memory cost (B extra floats per cell).
- **B.** Fast integrator convergence.
- **C.** Bin-resolution tests.
- **D.** Corner plots for all families.
- **E.** Mock-SFH demonstration: the current results, framed as a pipeline demonstration.
- **F.** Glossary: age bin, weight, yield mass fraction, DTD, prompt share, conserved N.

---

## 3. Figure and table list (planned)

| # | Content | Exists in some form? |
|---|---|---|
| F1 | Age-tracer schematic + one real weight vector | No |
| F2 | All DTD families on log-log, with bin edges | Partly (rate panels in movies; `apogee_peak_growth_dtd_shapes.png`) |
| F3 | APOGEE sample + dual-Gaussian + slice histograms | Yes (`apogee_dr17_analysis.png`), redo on full DR17 |
| F4 | Closure test (tracer vs in-code abundances) | No (needs real sim) |
| F5 | Injection–recovery coverage | Partly (mock demos) |
| F6 | Fiducial sim vs APOGEE | Partly |
| F7 | Summary-statistic sensitivity to DTD parameters | No |
| F8 | Corner plots (4 families) | Yes (mock SFH) |
| F9 | Posterior DTD bands vs observed DTDs | Partly (rate panel) |
| F10 | Posterior-predictive residual maps | No |
| F11 | Movie still frame (+ movies as online material) | Yes |
| T1 | Rates and yields per channel (FIRE-2.1 vs FIRE-3) | No |
| T2 | DTD families: formula, parameters, priors, motivation | Partly (README / chat) |
| T3 | Model comparison (ln Z, χ²_maxL, k, prompt share) | Partly (`apogee_dtd_suite_summary.md`) |
| T4 | Zero-point / systematics variants | No |
| T5 | APOGEE cut-flow | No |

---

## 4. Work plan (in dependency order)

1. **Scope and authorship** with Wetzel/Emerick (§0.2). Obtain the exact deposition equation and
   a list of FIRE runs that carry age-tracer fields (16 bins).
2. **Real simulation I/O.** Read tracer weights with `gizmo_io` + `gizmo_agetracer` for one MW
   host at z = 0. Select stars as in §3.2.
3. **Closure test (V1)**, then Figure 4. If it fails, stop and diagnose before anything else.
4. **Full APOGEE DR17** with cuts and a cut-flow table (needs SDSS SAS access, which is blocked
   in this cloud container; run locally).
5. **Error model (G3)**: bootstrap APOGEE and the simulation, add a model floor, build the
   covariance.
6. **Rerun the 10-family suite** on the real weights, plus nested sampling for ln Z.
7. **Validation set** (V2–V8).
8. **Add [O/Fe] or [Si/Fe]** (G10) and a yield-nuisance variant (G8).
9. **Second and third hosts** (if Option A includes them, else Paper II).
10. **Figures**, then the draft in the order §2 → §5 → §6 → §3/4 → §1 → §7 → abstract.
11. **Internal FIRE collaboration review** (FIRE papers circulate to the collaboration before
    submission), then arXiv + MNRAS.

---

## 5. Citation map

Facts marked ✔ were checked against the source text during planning. The rest are standard
references to confirm in ADS before use. arXiv itself was unreachable from this environment;
the ✔ checks used the publisher page and search abstracts.

**FIRE / GIZMO / tracers**
- ✔ Hopkins et al. 2023, MNRAS 519, 3154 (FIRE-3). Contains: the tracers model attributed to
  "Wetzel et al., in prep." (§3.1.2); the Ia DTD a = 0.0083 Gyr⁻¹ M☉⁻¹ (t/44 Myr)^−1.1 with
  1.6×10⁻³ Ia M☉⁻¹ over a Hubble time (§3.3.4); Ia yields and the "tracer-element approach in
  post-processing" remark (§3.11.1); the factor 2.4/1.8/1.5 more Ia than Mannucci at
  1/10/13.7 Gyr.
- ✔ Hopkins, "Changes from FIRE-2 to FIRE-2.x" notes: abundance tracers model of "Emerick et
  al., in prep."
- Hopkins et al. 2018, MNRAS 480, 800 (FIRE-2; metal diffusion App. F3).
- Hopkins 2015 (GIZMO).
- ✔ Wetzel et al. 2023, ApJS 265, 44 (FIRE-2 public data release).
- Wetzel et al. 2016 / GizmoAnalysis ASCL 2002.015.
- Escala et al. 2018 (metal mixing).
- Emerick, Bryan & Mac Low 2018/2019 (individual-star chemical tracers; context only).

**FIRE abundance / Ia studies**
- ✔ Gandhi et al. 2022, MNRAS 516, 1941 (post-processed rates vs re-simulation).
- Parul et al. 2023, MNRAS 520, 1672 (bursty SF and α; FIRE-2 [O/Fe] high, attributed to the Ia
  yields/DTD).
- ✔ Parul et al. 2025, arXiv 2501.12342 (gas accretion and α-bimodality, 11 hosts).
- ✔ Barry et al. 2026, arXiv 2601.02520 (strong bimodality in ~¼ of 16 hosts).
- arXiv 2512.14897 ("Portrait of a Galaxy on FIRE", Romeo).
- Bellardini et al. 2021, 2022 (3-D abundances); Carrillo et al. 2023 (age–[X/Fe] in m12i);
  Graf et al. 2025.

**DTD observations and theory**
- Mannucci, Della Valle & Panagia 2006.
- Totani et al. 2008.
- Maoz & Mannucci 2012.
- Maoz, Mannucci & Nelemans 2014.
- Maoz & Graur 2017.
- Strolger et al. 2020.
- Wiseman et al. 2021.
- Greggio 2005.
- Ruiter & Seitenzahl 2025 (arXiv 2412.01766).
- Freundlich & Maoz 2021 (clusters).

**GCE and the bimodality**
- Hayden et al. 2015.
- Nidever et al. 2014.
- Weinberg, Andrews & Freudenburg 2017.
- Weinberg et al. 2019.
- ✔ Dubay et al. 2024 (arXiv 2404.08059; GCE favours an extended DTD).
- arXiv 2602.22333 (dSph DTD constraints).

**Yields / solar**
- Iwamoto et al. 1999.
- Leung & Nomoto 2018.
- Mori et al. 2018.
- Seitenzahl et al. 2013.
- Asplund et al. 2009.
- Kirby et al. 2019; de los Reyes et al. 2020 (Ia yield variation).

**Data and statistics**
- Abdurro'uf et al. 2022 (DR17).
- Majewski et al. 2017.
- García Pérez et al. 2016 (ASPCAP).
- Foreman-Mackey et al. 2013 (emcee).
- Speagle 2020 (dynesty).
- Hogg & Foreman-Mackey 2018 (MCMC reporting).

---

## 6. Facts found while preparing this plan that should go into the text

1. **The conserved count is a mass, not a number.** N_0 = 2.2526×10⁻³ is the integrated Ia
   *mass-loss* (rate × 1.4 M☉). The event count is N_0 / 1.4 = 1.609×10⁻³ M☉⁻¹, matching
   FIRE-3's 1.6×10⁻³ (§3.3.4). The code's docstrings say "proportional to"; the paper should
   give both numbers.
2. **The Ia yields differ between FIRE-2 and FIRE-3.** The fits use FIRE-2.1 Iwamoto W7 values
   (Fe 0.531, Mg 6.14×10⁻³). FIRE-3 uses Leung & Nomoto W7/WDD2 (Fe 0.558, Mg 6.21×10⁻³),
   about +0.02 dex in Ia Fe. It is small, but Table 1 must name the set.
3. **The onset time differs.** FIRE-2.1 (code) uses 37.53 Myr (last CCSN); FIRE-3 uses 44 Myr.
   The `maoz_onset` posterior (45 +28/−7 Myr) is consistent with FIRE-3's choice. Worth one
   sentence.
4. **The `mannucci` family is not FIRE-2 Mannucci.** Conservation rescales FIRE-2 Mannucci up to
   the Maoz N. FIRE-2's native Mannucci produces ~1.5× fewer Ia by 13.7 Gyr (FIRE-3 §3.3.4).
   Label it "Mannucci shape at Maoz normalisation", and optionally run the native-N version as
   a reference point.
5. **Median-χ² ranking can invert nested models** (maoz_onset below maoz). Switch to max-L χ² +
   evidence (G4).
