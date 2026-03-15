# Pocket-K: Single-Lead ECG Hyperkalemia Risk Screening

Pocket-K is a lightweight, open-source toolkit designed for **single-lead ECG-based hyperkalemia risk screening**.

It provides a streamlined pipeline for model training, batch inference, and result analysis. Built with reproducibility and real-world deployment in mind, this repository is tailored for scenarios ranging from offline model evaluation on large datasets to rapid risk estimation in connected, handheld, or wearable workflows.

---

## Features

* **Single-Lead ECG Support:** Optimized for Lead I ECG signals (1D series), making it highly suitable for handheld or smartwatch-connected workflows.
* **Dynamic Preprocessing:** Built-in PyTorch Dataset automatically handles signal loading, lead extraction, center-cropping, or zero-padding to ensure uniform input lengths.
* **Robust Evaluation:** Includes out-of-the-box support for dynamic threshold evaluation, Bootstrap Confidence Intervals (CI), AUPRC, and F1-score calculations.
* **Unified Pipeline:** Seamless transition from training/fine-tuning to batch inference using the same core utilities.

---

## Repository Structure

```text
Pocket-K/
├── checkpoint/
│   └── checkpoint.pth    # Default directory for pre-trained/fine-tuned weights
├── data/                 # Recommended directory for your .npy ECG arrays and labels.csv
├── utils/
│   ├── dataset.py        # PyTorch Dataset for 1D signal loading & preprocessing
│   ├── net1d.py          # 1D Neural Network architectures
│   └── util.py           # Metrics, checkpoints, and CI calculation tools
├── train.py              # Main training and validation script
├── inference.py          # Batch inference and evaluation script
├── requirements.txt      # Python dependencies
└── README.md

```

---

## Data Format

Pocket-K expects 1D signal data and a corresponding metadata CSV.

**ECG Input:**

* **File type:** `.npy`
* **Signal type:** Lead I ECG (Automatically extracted if using standard 12-lead arrays)
* **Sampling rate:** `500 Hz`
* **Target Length:** `10 s` per inference clip (5000 time steps). *Note: The `dataset.py` will automatically pad or crop signals to meet this length.*

**Labels CSV (`labels.csv`):**
Should contain at least the relative paths/filenames and the target labels.
Example columns: `file_path`, `hyperkalemia_label`, `original_value`.

---

## Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/Pocket-K.git
cd Pocket-K
pip install -r requirements.txt

```

---

## Quick Start

### 1. Train or Fine-tune

Run the training pipeline using your prepared data directory and labels CSV. The script automatically handles imbalanced data via weighted sampling and tracks validation metrics.

```bash
python train.py \
    --data-dir data/ \
    --labels-path data/labels.csv \
    --target-cols hyperkalemia_label \
    --saved-dir outputs/ \
    --pretrained-ckpt checkpoint/base_model.pth \
    --epochs 100 \
    --batch-size 256

```

*(Tip: Add the `--linear-prob` flag if you only want to freeze the backbone and train the final classification head.)*

### 2. Run Batch Inference / Evaluation

Evaluate a trained model on a holdout test set or run predictions on new data. By default, this script loads the weights from `checkpoint/checkpoint.pth`.

```bash
python inference.py \
    --data-dir data/ \
    --labels-path data/test_labels.csv \
    --target-cols hyperkalemia_label \
    --ckpt-path checkpoint/checkpoint.pth \
    --output-csv predictions.csv

```

This will output a `.csv` file containing the true labels, predicted probabilities, and original potassium values for downstream analysis.

---

## Important Notes

* **Clinical Disclaimer:** Pocket-K is intended as a **screening-oriented research tool** and is *not* a replacement for laboratory confirmation (e.g., venous blood tests) or professional medical diagnosis.
* **Signal Quality:** Input signal quality matters significantly. Motion artifacts, extreme baseline wander, or severe rhythm abnormalities may degrade model performance.
* **Deployment:** For clinical or edge deployment, additional engineering (signal-quality control algorithms, device calibration, regulatory compliance) is required.

---


