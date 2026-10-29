# SCADA Pipeline: Detecting Degradation with a Simple Rule

Exploratory analysis of simulated SCADA data from oil and gas pipelines, done in Python (pandas) on Google Colab. The question: can a simple rule on pump speed catch the fault type that the built-in alarm mostly misses, and how honest is that result on data the rule has not seen?

<img width="835" height="363" alt="image" src="https://github.com/user-attachments/assets/b64c16c2-8c41-419f-bbe4-80fbdb815bd4" />


## Dataset

- **Source:** [SCADA Pipeline Operations Dataset](https://www.kaggle.com/datasets/zara2099/scada-pipeline-operations-dataset) on Kaggle (CSV). Check the license on the Kaggle page before reusing it.
- **Nature:** simulated SCADA system (digital twin), not data from a real pipeline.
- **Columns (13):** `timestamp`, `segment_id`, `pressure`, `flow_rate`, `temperature`, `valve_status`, `pump_state`, `pump_speed`, `compressor_state`, `energy_consumption`, `alarm_triggered`, `event_type`, `target`.
- **Event labels:** normal (694 rows), degradation (135), leak (65), surge (61), blockage (45).
- **The file I used:** 1,000 rows, about 17 minutes (timestamps from 0:00 to 0:16 on 1 Jan 2024), roughly 60 rows per minute.
- **A mismatch I noticed:** the description mentions 12 months at 1 Hz and about 2.5 million records, but 12 months at one reading per second would be about 31.5 million rows. I did not resolve this, so I treat my file as a small sample.

The CSV is not included in this repository.

## Questions

1. Does the built-in alarm catch every fault type?
2. Which measurements separate degradation from normal operation?
3. Can a simple rule catch degradation, and is the result still true on data the rule has not seen?

## Method

1. **Understand the data:** shape, types, event counts, and which numeric columns are really codes (`valve_status`, `pump_state`, `compressor_state`, `alarm_triggered`, `target`), where a mean is meaningless.
2. **Check the labels:** `target` turned out to be derived from `event_type` (normal is 0, every fault is 1). Neither `event_type` nor `alarm_triggered` should be used as a model input, because they leak the answer.
3. **Drop `segment_id`:** 50 segments with about 20 rows each. The spread of fault rates between segments (standard deviation 0.115) was close to what chance alone gives (about 0.10), so I found no evidence it matters in this sample.
4. **Compare measurements by event type**, then restrict to rows where the pump is running, so that rows with speed 0 do not distort the speed averages.
5. **Build a rule:** flag degradation when pump speed falls below a threshold.
6. **Test on unseen data:** split the pump-running normal and degradation rows into two random halves, choose the threshold from the normal rows of the first half only, evaluate on the second half, and repeat for 10 random splits.
7. **Vary the false alarm rate** the rule is allowed (1%, 2%, 5%, 10%, 20%).
8. **Cover the rows the speed rule cannot see.** With the pump off, speed is always 0, so I checked the other measurements for the 47 degradation rows with the pump off. Only energy consumption differed (about 22.6 versus 15.9 for normal rows), and a second rule flags rows with energy above a threshold.
9. **Combine the detectors.** A row is flagged if the alarm fired, or the pump is running and speed is below its threshold, or the pump is off and energy is above its threshold. Both thresholds are chosen from the normal rows of the training half at a 5% target, and the combined system is evaluated per event type on the test half, over 10 random splits.

## Key findings

- **The alarm has no false alarms in this sample** (0 of 694 normal rows) and fired on 100% of blockage, leak and surge rows. For degradation it fired on only 45 of 135 rows (29 of 88 while the pump was running).
- **Degradation shows up in the pump, not in pressure or flow.** With the pump running, mean speed was about 1,110 for degradation versus about 1,395 for normal (standard deviation about 100 in both), and energy consumption was about 39.8 versus 30.0. Pressure and flow stayed close to normal for degradation. Temperature was about 32 for every event type.
- **A fixed threshold (speed below 1,250)** caught 78 of 88 degradation rows (about 89%) with about 7% false alarms, but this threshold was chosen on the same data it was evaluated on, so it is optimistic.
- **On held-out data** (threshold chosen from normal rows only, at a 5% false alarm target), the rule caught on average 84% of degradation rows (standard deviation 0.05) with a 5% false alarm rate, almost the same as on the half used to choose the threshold. Across the 10 splits, detection ranged from 74% to 91%, so a single split would have been misleading.
- **The trade-off.** Averaged over the 10 splits, as the allowed false alarm rate goes from 1% to 2%, 5%, 10% and 20%, detection goes from 63% to 75%, 84%, 91% and 96%. Gains shrink as more false alarms are accepted, and the right point depends on whether a missed fault or a false alarm costs more.
- **The two detectors complement each other** on this sample: the speed rule (threshold 1,250) caught 51 of the 59 pump-running degradation rows that the alarm missed, while the alarm added only 2 rows the rule missed.
- **With the pump off, only energy separates degradation.** Pressure, flow and temperature were the same as for normal rows. Energy was higher (about 22.6 versus 15.9, roughly 1.2 standard deviations), so the two distributions overlap much more than pump speed does. An energy rule caught about 60% of these rows at a 5% false alarm target (standard deviation 0.07 to 0.09 across splits), and allowing 10% or 20% false alarms barely helped (62% and 63%). I did not investigate why.
- **Final combined system** (alarm, speed rule, energy rule; averages over 10 random splits, thresholds chosen from normal training rows only):

  | Event type | Share of rows flagged |
  |---|---|
  | blockage | 1.00 |
  | leak | 1.00 |
  | surge | 1.00 |
  | degradation | 0.80 (std 0.03) |
  | normal (false alarms) | 0.05 (std 0.01) |

  For blockage, leak and surge, the built-in alarm already fired on every row, so the 1.00 comes from the alarm and not from my rules. For degradation, the combination raised detection from about 33% (alarm alone) to about 80%, at the cost of about 5% false alarms on normal rows.

## Limitations

- The data is **simulated**. The clean separation between degradation and normal operation may be much weaker on a real pipeline.
- The sample is **small**: 88 degradation rows and 487 normal rows with the pump running. Each test half holds about 44 degradation rows, so results move by several points between splits.
- The 10 splits **reuse the same rows**, so averaging them shows sensitivity to the split, not independent evidence.
- Splits are **random**, not by time. For real time series data, train on earlier data and test on later data.
- The speed rule alone only covers rows where the pump is running. **47 of the 135 degradation rows had the pump off**, and for those the energy rule is weaker (about 60%), which is why the combined detection on all degradation rows is about 80% rather than the 84% seen on pump-running rows.
- The spread across the 10 splits (standard deviation 0.03 for degradation) understates the real uncertainty, because the splits overlap. With 135 degradation rows in total, a rough range for the true detection rate would be about 0.73 to 0.87.
- Faults make up about 31% of this sample, which is far more than on a real pipeline. On real data, where faults are rare, a 5% false alarm rate would produce many more false alarms than real detections.
- At higher false alarm rates, a growing share of the flagged rows are normal, because normal rows outnumber degradation rows.

## How to run

1. Download the CSV from the Kaggle link and name it as the notebook expects.
2. Open `scada_pipeline_fault_detection.ipynb` in Google Colab.
3. Upload the CSV through the Files panel (uploaded files are deleted when the session ends).
4. Run all cells (Runtime, then Run all).

Setup cell used at the top of the notebook:

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("your_file_name.csv")
df = df.drop(columns=["segment_id"])

run = df[df["pump_state"] == 1]
sub = run[run["event_type"].isin(["normal", "degradation"])].copy()
```

## Tools

Python, pandas, NumPy, matplotlib, Google Colab.

## What I practiced

`value_counts`, `crosstab`, `groupby` with `mean` and `agg`, boolean filtering, quantiles, random train and test splits with repeated runs, and reading a detection versus false alarm curve.

## Author

muzan eltinay · https://www.linkedin.com/in/muzan-eltinay/
