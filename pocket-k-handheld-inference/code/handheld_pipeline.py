import argparse
import json
import sys
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
import torch
from scipy.signal import resample


CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from deploy_test_inference import (  # noqa: E402
    DEFAULT_CONFIG,
    PACKAGE_ROOT,
    assign_risk_group,
    crop_or_pad,
    filter_bandpass,
    get_preprocessing_config,
    get_threshold_config,
    load_config,
    load_deployment_model,
    resolve_package_path,
    to_lead_first,
)


InferenceFn = Callable[[np.ndarray], np.ndarray]


def zscore(signal: np.ndarray) -> np.ndarray:
    return ((signal - np.mean(signal)) / (np.std(signal) + 1e-8)).astype(np.float32)


def resample_to_target(
    signal: np.ndarray,
    source_sample_rate: int,
    target_sample_rate: int,
) -> np.ndarray:
    if source_sample_rate <= 0:
        raise ValueError("source_sample_rate must be positive")
    if source_sample_rate == target_sample_rate:
        return signal.astype(np.float32)

    target_len = int(round(signal.shape[1] * target_sample_rate / source_sample_rate))
    if target_len <= 0:
        raise ValueError("resampled signal would be empty")
    return resample(signal, target_len, axis=1).astype(np.float32)


def prepare_handheld_clips(
    ecg_array: np.ndarray,
    source_sample_rate: int,
    target_sample_rate: int = 500,
    measurement_seconds: float = 30.0,
    clip_seconds: float = 10.0,
    apply_filter: bool = True,
    filter_low_hz: float = 0.5,
    filter_high_hz: float = 40.0,
) -> np.ndarray:
    """Convert one handheld measurement into model-ready 10-second clips.

    The public reference implementation starts from an anonymized ECG array.
    BLE acquisition and mobile UI code are intentionally kept outside this
    backend package.
    """
    data = np.nan_to_num(ecg_array, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    data = to_lead_first(data)
    signal = data[:1, :]

    signal = resample_to_target(signal, source_sample_rate, target_sample_rate)
    if apply_filter:
        signal = filter_bandpass(
            signal,
            target_sample_rate,
            lowcut=filter_low_hz,
            highcut=filter_high_hz,
        )

    measurement_len = int(round(measurement_seconds * target_sample_rate))
    clip_len = int(round(clip_seconds * target_sample_rate))
    if measurement_len <= 0 or clip_len <= 0:
        raise ValueError("measurement_seconds and clip_seconds must be positive")
    if measurement_len % clip_len != 0:
        raise ValueError("measurement length must be an exact multiple of clip length")

    signal = crop_or_pad(signal, measurement_len)
    clips = []
    for start in range(0, measurement_len, clip_len):
        clips.append(zscore(signal[:, start:start + clip_len]))
    return np.stack(clips, axis=0).astype(np.float32)


def make_torch_inference_fn(model: torch.nn.Module, device: torch.device) -> InferenceFn:
    def infer(clips: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(clips).to(device)
        with torch.inference_mode():
            probs = torch.sigmoid(model(x)).detach().cpu().numpy().reshape(-1)
        return probs.astype(np.float32)

    return infer


def score_handheld_measurement(
    ecg_array: np.ndarray,
    inference_fn: InferenceFn,
    source_sample_rate: int,
    target_sample_rate: int,
    low_threshold: Optional[float] = None,
    high_threshold: Optional[float] = None,
    measurement_seconds: float = 30.0,
    clip_seconds: float = 10.0,
    apply_filter: bool = True,
    filter_low_hz: float = 0.5,
    filter_high_hz: float = 40.0,
) -> Dict[str, object]:
    clips = prepare_handheld_clips(
        ecg_array,
        source_sample_rate=source_sample_rate,
        target_sample_rate=target_sample_rate,
        measurement_seconds=measurement_seconds,
        clip_seconds=clip_seconds,
        apply_filter=apply_filter,
        filter_low_hz=filter_low_hz,
        filter_high_hz=filter_high_hz,
    )
    clip_probs = np.asarray(inference_fn(clips), dtype=np.float32).reshape(-1)
    if clip_probs.shape[0] != clips.shape[0]:
        raise ValueError("inference_fn must return one score per clip")

    measurement_score = float(np.mean(clip_probs))
    result = {
        "pred_prob": measurement_score,
        "measurement_pred_prob": measurement_score,
        "clip_pred_probs": [float(x) for x in clip_probs],
        "clip_count": int(clips.shape[0]),
        "clip_seconds": float(clip_seconds),
        "measurement_seconds": float(measurement_seconds),
        "aggregation": "mean_clip_score",
        "source_sample_rate": int(source_sample_rate),
        "target_sample_rate": int(target_sample_rate),
    }
    risk_group = assign_risk_group(measurement_score, low_threshold, high_threshold)
    if risk_group is not None:
        result["risk_group"] = risk_group
    return result


def predict_handheld_path(
    ecg_path: Path,
    inference_fn: InferenceFn,
    source_sample_rate: int,
    target_sample_rate: int,
    low_threshold: Optional[float] = None,
    high_threshold: Optional[float] = None,
    measurement_seconds: float = 30.0,
    clip_seconds: float = 10.0,
    apply_filter: bool = True,
    filter_low_hz: float = 0.5,
    filter_high_hz: float = 40.0,
) -> Dict[str, object]:
    ecg_array = np.load(ecg_path)
    return score_handheld_measurement(
        ecg_array,
        inference_fn=inference_fn,
        source_sample_rate=source_sample_rate,
        target_sample_rate=target_sample_rate,
        low_threshold=low_threshold,
        high_threshold=high_threshold,
        measurement_seconds=measurement_seconds,
        clip_seconds=clip_seconds,
        apply_filter=apply_filter,
        filter_low_hz=filter_low_hz,
        filter_high_hz=filter_high_hz,
    )


def get_handheld_config(config: dict) -> dict:
    handheld = config.get("handheld_pipeline", {})
    preprocessing = get_preprocessing_config(config)
    return {
        "target_sample_rate": int(config.get("sample_rate", 500)),
        "measurement_seconds": float(handheld.get("measurement_seconds", 30.0)),
        "clip_seconds": float(handheld.get("clip_seconds", 10.0)),
        "apply_filter": bool(config.get("apply_filter", True)),
        **preprocessing,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Handheld 30-second measurement inference demo.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--input-npy", default=str(PACKAGE_ROOT / "test_data" / "ecg_npy" / "demo_handheld_30s.npy"))
    parser.add_argument("--source-sample-rate", type=int, default=500)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model = load_deployment_model(config, device)
    inference_fn = make_torch_inference_fn(model, device)
    low_threshold, high_threshold = get_threshold_config(config)

    result = predict_handheld_path(
        resolve_package_path(args.input_npy, PACKAGE_ROOT),
        inference_fn=inference_fn,
        source_sample_rate=args.source_sample_rate,
        low_threshold=low_threshold,
        high_threshold=high_threshold,
        **get_handheld_config(config),
    )
    result["model_name"] = config["model_name"]
    result["input_npy"] = str(args.input_npy)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
