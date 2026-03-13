# Pocket-K

Pocket-K is a toolkit for **single-lead ECG-based hyperkalemia risk screening**.

It provides a practical pipeline for:

- ECG preprocessing,
- model inference,
- batch evaluation,
- result analysis, and
- deployment-oriented integration in handheld workflows.

This repository is designed for **tool use and reproducible execution**, including input preparation, inference, and downstream analysis.

---

**Pocket-K** is a lightweight toolkit for **single-lead ECG preprocessing, hyperkalemia risk inference, and deployment-oriented screening workflows**.

It is designed for scenarios such as:

- offline model evaluation on ECG datasets,
- batch inference on single-lead ECG recordings,
- rapid risk estimation in connected handheld workflows,
- visualization and downstream analysis of model outputs.

Pocket-K focuses on the **tooling pipeline** rather than paper-style cohort description. The repository is intended to help users quickly understand:

1. what input the tool expects,  
2. what output it produces, and  
3. how to run training, evaluation, or inference in practice.

---

## Features

- **Single-lead ECG support**  
  Works with Lead I ECG signals and is suitable for handheld or wearable style workflows.

- **Unified preprocessing pipeline**  
  Supports filtering, resampling, normalization, and clip preparation before model inference.

- **Hyperkalemia risk inference**  
  Produces a risk score for potassium abnormality from ECG waveform input.

- **Batch processing**  
  Can be used for large-scale offline evaluation on multiple ECG samples.

- **Deployment-friendly design**  
  Can be integrated into smartphone-connected or edge-assisted inference workflows.

- **Analysis utilities**  
  Supports visualization, prediction review, and model behavior inspection.

---

## Workflow Overview

```text
Single-lead ECG
    ↓
Preprocessing
    ↓
Model Inference
    ↓
Risk Score Output
    ↓
Visualization / Analysis / Deployment
```

---

## Input Format

Pocket-K expects **single-lead ECG recordings** as model input.

Recommended format:

* file type: `.npy` or equivalent tensor format
* signal type: Lead I ECG
* sampling rate: `500 Hz`
* clip length: `10 s` per inference clip

If a longer recording is used, it can be split into multiple consecutive clips and then aggregated into a recording-level prediction.

---

## Output Format

The toolkit outputs one or more of the following:

* **risk probability** for hyperkalemia
* **sample-level prediction results**
* **batch inference results**
* **optional visualizations** for waveform inspection or downstream analysis

Depending on your pipeline, outputs may be saved as:

* console logs
* `.csv` prediction tables
* `.json` records
* figure files for visualization

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Quick Start

### 1. Prepare ECG data

* Organize your single-lead ECG files
* Ensure sampling rate and clip length match the model expectation
* Apply the required preprocessing pipeline if not handled automatically

### 2. Run inference

```bash
python inference.py --input path/to/ecg.npy
```

### 3. Run batch evaluation

```bash
python evaluate.py --input path/to/data.csv
```

### 4. Train or fine-tune

```bash
python train.py
```

> Replace the script names above with the actual entry points in your repository.

---

## Typical Use Cases

### Offline dataset evaluation

Use Pocket-K to evaluate model performance on a prepared ECG dataset and export prediction results for later analysis.

### Single-record risk estimation

Run inference on one ECG recording and obtain an immediate risk score.

### Handheld or smartphone-connected workflow

Integrate the inference pipeline into a mobile or connected acquisition setting for near-real-time screening.

### Research analysis

Inspect waveform samples, compare predictions, and generate visual summaries for model interpretation.

---

## Repository Structure

A typical repository layout may include:

```text
Pocket-K/
├── README.md
├── requirements.txt
├── train.py
├── evaluate.py
├── inference.py
├── models/
├── preprocess/
├── analysis/
├── data/
└── outputs/
```

You can adjust this section to match your actual repository structure.

---

## Notes

* Pocket-K is intended as a **screening-oriented tool**, not a replacement for laboratory confirmation.
* Input quality matters: noisy recordings, motion artifacts, or rhythm abnormalities may affect model behavior.
* For real deployment, additional engineering work such as signal-quality control, calibration, and device integration may be required.

---

## Citation

If you use this repository in your research or development workflow, please cite the corresponding paper.
