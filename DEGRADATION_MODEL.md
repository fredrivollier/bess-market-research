# Degradation model calibration & validation

Infrastructure tracks for `lib/models/degradation*`. Not editorial notes — ongoing research that strengthens the quantitative backbone used across notes (`degradation-drivers`, `trader-aging-aware`, `warranty`, `missed-value`).

**Current posture (2026-04-16).** Two independent out-of-sample anchors are live — SNL Preger 2020 (cycling, 30 A123 18650 cells) and Stanford Lam 2024 (calendar, 80 K2 Energy LFP18650 cells). Both act as regression tripwires, not fit targets — SNL residuals (A123 small-format) and Stanford 60/85 °C extrapolation artefacts should not drive preset retuning. The library is cleared for **open-source release** at this validation posture: baseline (25 °C / 0.5 C) is anchored to datasheet presets, calendar channel generalises to a second manufacturer within its envelope, and SNL bias direction at high-T × high-C is disclosed. Recalibration of `Ea_cyc_eV` / `c_rate_exponent` remains deferred until a public 280 Ah prismatic dataset lands; Strübbel field telemetry feeds a *private* calibration overlay, not the public presets.

**Kernel changes 2026-04-16 (pre-open-source hardening).** Four fixes prompted by a scientific review before public release; none move numerical results on existing calibrations (Naumann MAE 3.23 % → 3.23 %; SimSES median |ΔSoH| 0.049 → 0.021; all 55 tests pass).
- **Continuous SoC form.** `k_cal(SoC) = a + b·u + c·u³` with `u = max(SoC − 0.5, 0)` replaces the 3-step bucket function. Default coefficients `(0.60, 2.694, −1.218)` solved to reproduce legacy bucket averages `{0.60, 1.00, 1.60}` at midpoints `{0.25, 0.65, 0.90}` exactly. `from_mean()` linearly interpolates bucket hour allocations between anchor `mean_soc` values `{0.30, 0.55, 0.75, 0.90}`. Kills the branch-aliasing that hid rest-SoC resolution above `mean_soc = 0.70`; unlocks Note 3b dispatch work.
- **Split Monte Carlo noise across cycle and calendar channels.** `sigma_cyc = k_cyc_cov · |q_cyc|`, `sigma_cal = k_cal_cov · |q_cal|`, combined in quadrature. Old formula tied sigma to cycle stress alone — a stored cell had zero MC spread, wrong for FCR-narrow, insurance, second-life cases. New `k_cal_cov` field defaults to 0.08 (= `k_cyc_cov`) pending a matched calendar-aging spread study.
- **Optional self-heating** on cycle channel: `T_cell = T_amb + coeff · C²`. Default 0 on every shipped preset (caller passes cell-internal T). Enables calibrated hot-climate + high-C runs once preset-level coefficients land (Strübbel track).
- **`return_kind="min"`** on `project_capacity_detailed` for warranty / insurance / EOL paths that need worst-cell SoH rather than weakest-decile `pack` (p10) default.

Three infrastructure-level honesty fixes to the editorial Note 3 text, prompted by the same review: climate-chart caption used `Ea ≈ 17 kJ/mol` (code is 0.55 eV ≈ 53 kJ/mol; Naumann 2018 reports 0.55–0.73 eV across SoC); a comment attributed Wang 2011's `z ≈ 0.55` exponent to NMC (Wang is LFP, the z=1 departure is justified by Naumann 2020 / Sarasketa-Zabala 2014); back-test expander framed all four tests as "passing" with SNL 12.3 pp MAE unflagged (now split into calibration-referential vs tripwire, with SNL bias disclosed and Stanford split by preset calibration-quality tier).

## SNL LFP out-of-sample validation — shipped 2026-04-14
First public LFP/graphite anchor independent of the Trina/EVE/CATL/BYD datasheet presets. 30 A123 18650 cells on a T × DoD × C-rate grid (Sandia, via batteryarchive.org Redash API). See [`lib/validation/snl_lfp.py`](lib/validation/snl_lfp.py), [`scripts/fetch_snl_lfp.py`](scripts/fetch_snl_lfp.py), [`lib/models/tests/test_snl_lfp_validation.py`](lib/models/tests/test_snl_lfp_validation.py).

**Result.** All five presets under-predict on SNL (bias −4.7 to −19.7 pp, MAE 4.9 to 19.8 pp on 30 in-range anchor points). Dominant residual source: the model over-penalises high-T × high-C stress (Trina bias at 35°C/2C = −25.6 pp). The presets are calibrated on 25°C/0.5C datasheet anchors and this corner was previously unvalidated. `Ea_cyc_eV=0.30` and `c_rate_exponent=1.0` are likely too aggressive for small-format A123 cells — not necessarily wrong for 280 Ah prismatic, which is why we don't retune.

## PyBaMM synthetic anchors — tried, not viable 2026-04-14
Attempted `Prada2013` LFP parameter set overlaid on `OKane2022` degradation parameters (SEI, Li plating, particle cracking, porosity clogging — combination from PyBaMM discussion #4126). 50-cycle DFN solve finished in 28 s but retention dropped 0.01 % — the OKane degradation rate constants are NMC-fit and produce nil LFP degradation in this pairing. Calibrating those k-constants against SNL would collapse PyBaMM from an independent anchor into a fit-derivative. Shelved unless someone publishes LFP-native degradation parameter values for PyBaMM.

## MATR/Severson 2019 — deprioritised 2026-04-14
124 A123 18650 LFP cells at a single operating point (T=30 °C, 4C discharge, 100 % DoD), varied only by fast-charge protocol. Same cell format as SNL — does not broaden physics coverage. Would sharpen `k_cyc_cov`, but Severson 2019 is already the cited default of 8 %, so this is regression to the source. Not pulled.

## Stanford long-term calendar validation — shipped 2026-04-15
Lam et al. 2024 Joule "A decade of insights" — 80 K2 Energy LFP18650E/P cells on a 4T × 2SoC storage grid (24/45/60/85 °C × 50/100 %), diagnostic RPT every 6 months for up to 8 years. Different manufacturer from Naumann (Sony/Murata → K2 Energy), genuinely independent. Paper OA at [osti.gov/pages/biblio/2536694](https://www.osti.gov/pages/biblio/2536694); data on OSF [`ju325`](https://osf.io/ju325/). See [`scripts/fetch_stanford_calendar.py`](scripts/fetch_stanford_calendar.py), [`lib/validation/stanford_calendar.py`](lib/validation/stanford_calendar.py), [`lib/models/tests/test_stanford_calendar_validation.py`](lib/models/tests/test_stanford_calendar_validation.py).

**Result (primary range, T ≤ 45 °C — within Naumann's own calibration envelope).** Trina + EVE presets fit out-of-sample K2 cells to bias −2.5 to −3.4 pp, MAE 4.1–4.6 pp on 112 anchor crossings — calendar channel generalises cleanly to a second manufacturer. Single-marketing-anchor presets (BYD / CATL / baseline) underpredict more (bias −6.8 to −10.1 pp) — a calibration artefact already documented in the preset notes.

**Out-of-range (60/85 °C).** Bias blows up to −16 to −30 pp. This is a known limitation of the Naumann Arrhenius power-law form: Li-inventory exhaustion saturates at high T in a regime the model does not capture. Stanford 60/85 °C anchors are retained for informational drift-detection only; tests do not gate on them. The practical implication — already in preset temperature ranges — is that the calendar channel should not be trusted beyond ~50 °C.

## Naumann 2018 raw data on Mendeley — optional sanity check
Raw XLSX/CSV for the 17 Naumann calendar test points is public at [data.mendeley.com/datasets/kxh42bfgtj/1](https://data.mendeley.com/datasets/kxh42bfgtj/1) (CC BY 4.0). Not an independent anchor — same dataset that pinned our current `k_cal` — but could verify we haven't misread the paper's reported constants. With Stanford now shipped as an independent calendar anchor, this is lower priority.

## Pre-publication gate — Trina multi-anchor source
`PRESETS["trina_elementa_280ah"]` is the only preset tagged `multi_anchor` (cycle + two calendar points), and it carries the library's best calibration. The two calendar anchors (40 % SoC / 2 yr ≥98 %, 100 % SoC / 10000 days → 70 %) come from an *internal AES Technology Research Institute report, 2025.04*, shared with BayWa r.e. — not publicly quoted. The `source_url` field points at Trina's marketing page, which does not carry those numbers.

Before Note 3 (or any note that uses this preset) ships public, resolve one of:
- **Option A — replace with public anchors.** Find a Trina datasheet / Trina-authored whitepaper / OEM press release that reproduces at least one of the calendar anchors publicly, and update `source_url` + anchor `source` strings accordingly.
- **Option B — downgrade preset.** Drop the two calendar anchors, refit `k_cal` from the cycle anchor alone (falls back to Naumann defaults), change `calibration_status` to `single_anchor_datasheet`, and document the downgrade in preset notes + Note 3 validation section.
- **Option C — keep private, remove from public release.** Move the preset to a private overlay (same pattern as Strübbel) and ship the public lib with four presets instead of five.

Default preference is A; B is the clean fallback. C is only if AES / Trina explicitly object to either.

## Next triggers for recal
- Public dataset on **280 Ah prismatic** LFP/graphite (not available today, watch EU H2020/Horizon releases and Fraunhofer) → recal `Ea_cyc_eV`/`c_rate_exponent` and validate against SNL.
- Long-duration LFP calendar data at 50–70 °C with finer time sampling *beyond Stanford Lam 2024* → fit a saturating calendar form to replace the current power law and close the 60/85 °C gap Stanford exposed. (Stanford's 60 °C weekly traces at 50/100 % SoC are already part of current posture; the gate is *additional* independent data, not re-using Stanford.)
- Strübbel field telemetry accumulates ≥18 months of production data → internal recal track, not public.
