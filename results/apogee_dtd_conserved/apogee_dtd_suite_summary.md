# APOGEE DR17 fits at fixed Ia event count

Every family is fit to the same dual-Gaussian APOGEE target with the same conserved total Ia event count N = 2.2526e-03, starting 1 sigma off the MW mean. chi^2 is over the 9 summary statistics, evaluated at the posterior median (not the maximum-likelihood point), so differences of a few are not significant.

| rank | DTD family | chi^2 (posterior median) | chi^2 start | bump share | acc. | posterior median (16-84%) |
|---|---|---|---|---|---|---|
| 1 | Mannucci prompt + tardy (free shape) (`mannucci_prompt`) | 18.49 | 173.8 | 80.8% | 0.39 | log10_a_prompt = -3.91 (+0.55/-0.56); t_p = 43.5 (+15/-2.6); sigma_p = 10.4 (+16/-4.3) |
| 2 | Skewed peak + exponential growth (`peak_growth`) | 24.36 | 198.3 | 67.8% | 0.40 | log10_a_peak = -3.61 (+0.38/-0.42); sigma_peak = 117 (+55/-56); skew_peak = -2.48 (+1.9/-1.7); k_grow = 0.139 (+0.097/-0.095) |
| 3 | Skew-normal (Strolger 2020) (`skewnorm`) | 25.19 | 187.9 | - | 0.48 | xi = 2.3 (+0.25/-0.2); omega = 1.06 (+0.29/-0.37); a = -2.6 (+1.3/-1.5) |
| 4 | Maoz power law (`maoz`) | 26.17 | 170.8 | - | 0.61 | t_dd = -1.49 (+0.15/-0.079) |
| 5 | Power law + long-delay bump (`long_delay`) | 26.27 | 170.9 | 0.0% | 0.36 | t_dd = -1.48 (+0.17/-0.085); log10_a_long = -9.49 (+1.8/-1.7); t_long = 8.99e+03 (+2.8e+03/-2.7e+03); sigma_long = 1.68e+03 (+9e+02/-8.1e+02) |
| 6 | Broken power law (`kink`) | 26.30 | 170.8 | - | 0.39 | t_dd = -1.45 (+0.24/-0.11); t_kink = 1.31e+03 (+1.2e+03/-1e+03); t_dd2 = -1.86 (+0.82/-0.52) |
| 7 | Prompt + delayed (`prompt_delayed`) | 26.64 | 170.8 | 0.9% | 0.31 | t_dd = -1.41 (+0.36/-0.15); log10_a_prompt = -5.5 (+1.2/-2.4); t_p = 77.3 (+45/-29); sigma_p = 23.7 (+11/-12) |
| 8 | Mannucci (FIRE-2, fixed shape) (`mannucci`) | 28.99 | 173.8 | 33.1% | 0.71 | fixed shape (only the zero-point offsets fitted) |
| 9 | Maoz power law + free onset (`maoz_onset`) | 29.84 | 170.8 | - | 0.48 | t_dd = -1.46 (+0.19/-0.1); t_ia = 45.2 (+28/-6.7) |
| 10 | Exponential (`exponential`) | 31.03 | 184.0 | - | 0.54 | tau = 273 (+2.1e+02/-56) |
