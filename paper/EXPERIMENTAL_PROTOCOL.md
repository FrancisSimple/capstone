# Experimental protocol: low-cost orange-spoilage measurement under warm storage

Version 1.0, 30 October 2026. Prepared in response to the supervisor's review (Phase 2) and Section 5.5 of the manuscript.
Fix this protocol, date it and share it before any outcome data are collected, so that nothing can later be tuned to the results.

---

## 1. Purpose

Collect, for the first time, data that can answer four questions:

1. Do the low-cost gas sensors separate fresh, medium and spoilt oranges **across independent batches**, and not only across recording sessions?
2. How long does an orange take to reach a defined spoilage endpoint under warm storage, and how much does this vary between fruit and between batches?
3. How well do temperature, humidity and sensor signals explain that time (survival model)?
4. Does the image model generalise to oranges and lighting it has not seen?

## 2. Design at a glance

| Item | Specification | Why |
|---|---|---|
| Batches | **≥ 10 independent batches** of **20 oranges**, each bought on a different day and preferably from different markets or traders | The number of batches drives precision (sample-size simulation in the manuscript); 20 fruit per batch is enough |
| Boxes | 3 identical sensing boxes run in parallel, plus 1 **empty control box** | Parallel boxes make 10 batches take ~3 months; the control box measures drift and background |
| Storage | Room conditions (no fridge), as in a trader's store; batches stored at **deliberately different temperatures** (for example a cooler and a warmer room, or two seasons), to vary temperature and humidity | A simulation of the analysis plan (script 15) shows that ten batches spread over the ~5 °C of one room detect a Q10 of 2 in fewer than half of studies |
| Duration | Day 0 (purchase) to day 21, or until every fruit reaches the endpoint | > 96% of fruit are expected to fail by day 21 |
| Inspections | Daily, at the same time (± 1 h), by trained assessors | Gives interval-censored event times |
| Sensor sessions | Continuous logging, plus **scheduled recording sessions** on days 0, 2, 4, 7, 10, 14 and 21 | Several sessions per spoilage stage, at different storage times |

## 3. Fruit-level measurements

Label every fruit with a unique ID: `B<batch>-F<fruit>`, for example B03-F17. Write the ID on the peel with a food-safe marker, away from the photographed side.

### 3.1 Day 0 (initial condition)

| Measurement | Instrument | Notes |
|---|---|---|
| Mass (g) | Kitchen scale with 0.1 g resolution or better | Weigh twice, record the mean |
| Equatorial diameter (mm) | Calliper or ruler | |
| Peel colour | Score 1-5 against a printed colour chart; plus a photograph (3.3) | Same chart for all batches |
| Firmness | Penetrometer if available; otherwise a finger-press score 1-5 with written anchors | Record the method used |
| Visible defects | Yes/no and description (bruise, cut, blemish) | Exclude fruit with open wounds |
| Origin | Market, trader, purchase time, price | Batch metadata |

### 3.2 Daily

| Measurement | Definition |
|---|---|
| Visual quality score | 0 = sound; 1 = slight softening or dull peel; 2 = soft spot < 10 mm, no mould; 3 = soft-rot lesion ≥ 10 mm, or visible mould; 4 = extensive rot, leakage or collapse |
| Mould | Presence and colour (green/blue sporulation = *Penicillium*); estimated % of the surface |
| Mass (g) | Same scale as day 0 → **mass loss %** |
| Firmness | Same method as day 0 (every 2 days is acceptable) |
| Removal | Date, time and reason if a fruit is removed |

**Spoilage endpoint (primary):** the first inspection at which a fruit scores **≥ 3**: visible fungal sporulation, a soft-rot lesion of at least 10 mm, or skin rupture with leakage.

**Censoring:**
- Fruit still sound at day 21 are right-censored at 21.
- Fruit removed for other reasons are censored at removal.

**Assessor agreement:** a second assessor independently scores a random 20% of inspections. Report Cohen's κ; the target is κ ≥ 0.8. Retrain the assessors if κ falls below 0.6.

### 3.3 Photographs

- Photograph each fruit daily from the side with the most visible change, plus one fixed side. Include an ID card and a colour reference card in the frame.
- Use **two lighting conditions**: (a) the fixed lamp in the box; (b) ambient room or market light.
- Vary the backgrounds over the study.
- Name the files `B03-F17_d07_lampA.jpg`.
- Images are split **by batch** for training and testing; images of one fruit never appear in both.

## 4. Sensor protocol

| Step | Procedure | Basis |
|---|---|---|
| Preheat | Power the MQ sensors continuously for at least **48 h** before the first batch (data sheets: over 48 h for MQ2 and MQ135, over 24 h for MQ3; longer if the sensors were stored unpowered for months); then never switch off | Transients after every restart in the released data; the sensors were characterised by their makers at 20 °C and 55-65% RH, well below storage conditions |
| Stabilisation | After any unavoidable restart, discard at least **90 min** of data | Measured: MQ3 settles in a median of 12 min (3-52); MQ2 and MQ135 showed no stable plateau within sessions of up to 2.5 h, so continuous operation matters more than any exclusion window |
| Calibration | Each day before the first session, record 10 min in clean outdoor air (or the empty control box) → *R*<sub>0</sub> per sensor. Store raw ADC counts **and** *R*<sub>s</sub>/*R*<sub>0</sub>. Note the load resistor and supply voltage | Separates raw counts from calibrated response |
| Cross-sensitivity | Once per box: run the empty box for 24 h across a day-night cycle, logging T and RH | Sensor response to T and RH without fruit |
| Repeatability | Once per batch: two consecutive 30-min sessions without moving anything | Within-box repeatability |
| Reference temperature | A calibrated logger beside the DHT22 in each box; a **probe in one fruit per box** (core temperature). Before the first batch, check every DHT22 and reference logger for 24 h over a saturated NaCl solution in a sealed jar (75.3% RH at 25 °C) | DHT22 vs reference vs fruit temperature; DHT22 stated accuracy ±0.5 °C, ±2% RH |
| Logging | 5-s sampling; ISO timestamps; box ID; batch ID; every restart, door opening and fruit removal logged | Lets every reading be traced to a unit |

## 5. External climate data

For the storage site and period, retrieve hourly ERA5 (via Open-Meteo) and NASA POWER temperature and RH using the scripts in `paper/analysis/`. Keep the reference logger running throughout: it gives the overlap needed to calibrate outdoor data to the room (manuscript Eq. 7).

## 6. Data management

- One CSV per box with columns `timestamp, box_id, batch_id, MQ2, MQ3, MQ135, Rs_R0_MQ2, Rs_R0_MQ3, Rs_R0_MQ135, T_dht, RH_dht, T_ref, RH_ref, T_fruit, event`.
- One inspection sheet per batch with columns `date, time, assessor, fruit_id, score, mould, mould_pct, mass_g, firmness, removed, note`.
- A data dictionary (`data/DATA_DICTIONARY.md` format) for every file.
- Daily backup to two locations.

## 7. Analysis plan (fixed in advance)

1. **Units:** report batches, fruit, events and readings separately.
2. **Survival:** Weibull model, *S*(*t*) = exp[−(*t*/*λ*)<sup>*k*</sup>], with interval censoring. Covariates (thermal exposure, mean RH, sensor summaries, initial condition) act on log *λ*, with a random effect per batch. Validate by leave-one-batch-out and on the last batches collected. Metrics: time-dependent Brier score at days 7 and 14, concordance, coverage of predictive intervals.
3. **Gas classification across batches:** the quality class at each session is defined from the fruit scores (for example, the median score of the box). Evaluate by leave-one-batch-out with proper scores and calibration.
4. **Sensor reliability:** mixed-effects models with random effects for box and batch; ICC; drift from the control box; cross-sensitivity to T and RH.
5. **Images:** leave-one-batch-out; precision, recall, F1, confusion matrices; with and without adaptation on other batches.
6. **Climate:** calibrate ERA5/POWER to the reference logger per box and season; test gap-filling with real outages.

## 8. Equipment checklist

- 4 sensing boxes (3 with fruit, 1 empty)
- MQ2/MQ3/MQ135 + DHT22 + ESP32 per box
- 4 reference loggers
- 3 fruit probes
- Scale (0.1 g)
- Calliper
- Penetrometer (optional)
- Colour chart and colour reference card
- Fixed lamp per box
- Camera or phone on a stand
- Labels and food-safe marker
- Printed inspection sheets

## 9. Timeline (three boxes in parallel)

| Weeks | Activity |
|---|---|
| 0-1 | Build and test boxes, preheat, cross-sensitivity runs, train assessors (κ check) |
| 1-4 | Batches 1-3 |
| 4-7 | Batches 4-6 |
| 7-10 | Batches 7-9 |
| 10-13 | Batch 10 (+ optional extra batches in a second room or season) |
| 13-16 | Analysis (Section 7) and manuscript update |
