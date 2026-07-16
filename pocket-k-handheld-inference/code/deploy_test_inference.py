import argparse
import json
import pickle
import sys
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from scipy.signal import butter, filtfilt, iirnotch, medfilt


CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from net1d import Net1D  # noqa: E402


PACKAGE_ROOT = CODE_DIR.parent
DEFAULT_CONFIG = PACKAGE_ROOT / "config" / "deployment_config.json"
DEFAULT_INPUT = PACKAGE_ROOT / "test_data" / "sample_metadata.csv"
DEFAULT_OUTPUT = PACKAGE_ROOT / "test_outputs" / "sample_predictions.csv"


def build_model(device: torch.device) -> Net1D:
    model = Net1D(
        in_channels=1,
        base_filters=64,
        ratio=1,
        filter_list=[64, 160, 160, 400, 400, 1024, 1024],
        m_blocks_list=[2, 2, 2, 3, 3, 4, 4],
        kernel_size=16,
        stride=2,
        groups_width=16,
        verbose=False,
        use_bn=False,
        use_do=False,
        n_classes=1,
    )
    return model.to(device)


def load_config(config_path: Path = DEFAULT_CONFIG) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def filter_bandpass(
    signal: np.ndarray,
    fs: int,
    lowcut: float = 0.5,
    highcut: float = 40.0,
) -> np.ndarray:
    b, a = iirnotch(50, 30, fs)
    filtered = np.zeros_like(signal, dtype=np.float32)
    for c in range(signal.shape[0]):
        filtered[c] = filtfilt(b, a, signal[c])

    b, a = butter(N=4, Wn=[lowcut, highcut], btype="bandpass", fs=fs)
    for c in range(filtered.shape[0]):
        filtered[c] = filtfilt(b, a, filtered[c])

    baseline = np.zeros_like(filtered, dtype=np.float32)
    kernel_size = int(0.4 * fs) + 1
    if kernel_size % 2 == 0:
        kernel_size += 1
    for c in range(filtered.shape[0]):
        baseline[c] = medfilt(filtered[c], kernel_size=kernel_size)
    return filtered - baseline


def crop_or_pad(signal: np.ndarray, target_len: int) -> np.ndarray:
    current_len = signal.shape[1]
    if current_len == target_len:
        return signal
    if current_len > target_len:
        start = (current_len - target_len) // 2
        return signal[:, start:start + target_len]
    pad_total = target_len - current_len
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    return np.pad(signal, ((0, 0), (pad_left, pad_right)), mode="constant")


def to_lead_first(data: np.ndarray) -> np.ndarray:
    if data.ndim == 3 and data.shape[0] == 1:
        data = data.squeeze(0)
    if data.ndim != 2:
        raise ValueError(f"Expected 2D ECG array, got shape={data.shape}")
    if data.shape[0] == 12:
        return data
    if data.shape[1] == 12:
        return data.T
    if data.shape[0] == 1:
        return data
    if data.shape[1] == 1:
        return data.T
    raise ValueError(f"Cannot identify ECG lead dimension, shape={data.shape}")


def load_signal(
    path: Path,
    sample_rate: int,
    signal_length: int,
    apply_filter: bool,
    filter_low_hz: float = 0.5,
    filter_high_hz: float = 40.0,
) -> np.ndarray:
    data = np.load(path)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    data = to_lead_first(data)

    signal = data[:1, :]
    if apply_filter:
        signal = filter_bandpass(signal, sample_rate, lowcut=filter_low_hz, highcut=filter_high_hz)
    signal = crop_or_pad(signal, signal_length)
    signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
    return signal.astype(np.float32)


def resolve_package_path(path: str, package_root: Path) -> Path:
    raw_path = Path(path)
    if raw_path.is_absolute():
        return raw_path
    return package_root / raw_path


def resolve_ecg_path(row: pd.Series, ecg_path_col: str) -> Path:
    ecg_path = resolve_package_path(str(row[ecg_path_col]), PACKAGE_ROOT)
    if ecg_path.exists():
        return ecg_path
    if "deployment_ecg_file" in row and pd.notna(row["deployment_ecg_file"]):
        package_ecg_path = PACKAGE_ROOT / "test_data" / "ecg_npy" / str(row["deployment_ecg_file"])
        if package_ecg_path.exists():
            return package_ecg_path
    return ecg_path


def assign_risk_group(
    score: float,
    low_threshold: Optional[float],
    high_threshold: Optional[float],
) -> Optional[str]:
    if low_threshold is None or high_threshold is None:
        return None
    if score < low_threshold:
        return "Low risk"
    if score < high_threshold:
        return "Medium risk"
    return "High risk"


def load_deployment_model(config: dict, device: torch.device) -> Net1D:
    model = build_model(device)
    checkpoint_path = resolve_package_path(config["checkpoint_path"], PACKAGE_ROOT)
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except (TypeError, pickle.UnpicklingError):
        # The packaged checkpoint is trusted; this fallback supports older checkpoint formats.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`")
            checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["state_dict"] if isinstance(checkpoint, dict) and "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def get_preprocessing_config(config: dict) -> dict:
    preprocessing = config.get("preprocessing", {})
    return {
        "filter_low_hz": float(preprocessing.get("bandpass_low_hz", 0.5)),
        "filter_high_hz": float(preprocessing.get("bandpass_high_hz", 40.0)),
    }


def get_threshold_config(config: dict) -> tuple:
    thresholds = config.get("thresholds") or {}
    low_threshold = thresholds.get("low_risk_threshold")
    high_threshold = thresholds.get("high_risk_threshold")
    if low_threshold is None or high_threshold is None:
        return None, None
    return float(low_threshold), float(high_threshold)


def main() -> None:
    parser = argparse.ArgumentParser(description="Deployment smoke-test inference for ECG-K checkpoint 902.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--input-csv", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-csv", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--ecg-path-col", default="deployment_ecg_path")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config = load_config(Path(args.config))

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model = load_deployment_model(config, device)

    df = pd.read_csv(args.input_csv)
    low_threshold, high_threshold = get_threshold_config(config)
    preprocessing = get_preprocessing_config(config)

    pred_probs = []
    risk_groups = []
    with torch.no_grad():
        for _, row in df.iterrows():
            signal = load_signal(
                resolve_ecg_path(row, args.ecg_path_col),
                sample_rate=int(config["sample_rate"]),
                signal_length=int(config["signal_length"]),
                apply_filter=bool(config["apply_filter"]),
                **preprocessing,
            )
            x = torch.from_numpy(signal).unsqueeze(0).to(device)
            prob = torch.sigmoid(model(x)).detach().cpu().numpy().reshape(-1)[0]
            pred_probs.append(float(prob))
            risk_group = assign_risk_group(float(prob), low_threshold, high_threshold)
            if risk_group is not None:
                risk_groups.append(risk_group)

    out = df.copy()
    out["pred_prob"] = pred_probs
    if risk_groups:
        out["risk_group"] = risk_groups

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    print(f"Saved predictions to {output_path}")
    display_cols = [
        col
        for col in [args.ecg_path_col, "demo_sample_id", "pred_prob", "risk_group"]
        if col in out.columns
    ]
    print(out[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()
