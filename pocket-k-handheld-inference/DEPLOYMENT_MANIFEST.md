# Pocket-K Handheld Inference Manifest

Repository directory:

`pocket-k-handheld-inference`

Copy this whole directory to the deployment server. Do not copy only the `.pth`, because the backend also needs the exact architecture, preprocessing logic, output schema, and handheld aggregation code.

## Required Runtime Files

```text
code/deploy_test_inference.py
code/handheld_pipeline.py
code/api_server.py
code/net1d.py
config/deployment_config.json
model/pocket_k_model.pth
```

`code/util.py` is included for provenance from the training code. `code/make_demo_data.py` regenerates the synthetic demo examples and contains no patient-specific source data.

## De-Identification Status

The bundled example inputs are synthetic demo ECG arrays with demo-only metadata:

```text
test_data/sample_metadata.csv
test_data/ecg_npy/demo_ecg_000.npy
test_data/ecg_npy/demo_ecg_001.npy
test_data/ecg_npy/demo_ecg_002.npy
test_data/ecg_npy/demo_ecg_003.npy
test_data/ecg_npy/demo_handheld_30s.npy
```

The demo metadata does not include patient IDs, encounter IDs, exact ECG timestamps, lab collection timestamps, raw patient labels, or patient-specific laboratory values.

## Standard Backend Contract

Input:

- ECG `.npy`
- accepted shapes: `(time, 12)`, `(12, time)`, `(1, time)`, or `(time, 1)`
- sample rate: 500 Hz
- lead used: I
- signal length: one 10-second clip, center-cropped or zero-padded to 5000 samples

Output:

- `pred_prob`: raw sigmoid model risk score
- optional `risk_group`: present only when private deployment thresholds are configured

## Handheld Backend Contract

Input:

- one anonymized 30-second ECG `.npy`
- accepted shapes: `(time, 12)`, `(12, time)`, `(1, time)`, or `(time, 1)`
- `source_sample_rate` supplied by the caller
- lead used: I

Pipeline:

1. Convert to Lead I
2. Resample to 500 Hz if needed
3. Apply 50 Hz notch filtering, 0.5-40 Hz band-pass filtering, and median-filter baseline removal
4. Crop or pad to 30 seconds
5. Split into three 10-second clips
6. z-score normalize each clip
7. Score each clip
8. Average clip scores into one measurement-level score

Output:

- `pred_prob` / `measurement_pred_prob`: mean of the three clip scores
- `clip_pred_probs`: three clip-level scores
- `aggregation`: `mean_clip_score`
- optional `risk_group`: present only when private deployment thresholds are configured

## Threshold Policy

Operating thresholds are deployment- and site-specific and are intentionally omitted from this public demo package. The public demo outputs continuous scores only. Private deployments can add a private `thresholds` object to `config/deployment_config.json` with `low_risk_threshold` and `high_risk_threshold` to enable categorical risk-group output without exposing cutoff values.

## Test Commands

From the package root:

```bash
python code/make_demo_data.py
python code/deploy_test_inference.py --device cpu
python code/handheld_pipeline.py --device cpu
```

## Start API

From the package root:

```bash
pip install -r requirements.txt
uvicorn code.api_server:app --host 0.0.0.0 --port 8000
```

The API exposes:

```text
GET  /api/health
POST /api/predict
POST /api/predict_upload
POST /api/predict_handheld
POST /api/predict_handheld_upload
```
