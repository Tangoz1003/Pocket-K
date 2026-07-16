import csv
from pathlib import Path

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEMO_DIR = PACKAGE_ROOT / "test_data" / "ecg_npy"
METADATA_PATH = PACKAGE_ROOT / "test_data" / "sample_metadata.csv"


def synthetic_lead(
    seconds: float,
    sample_rate: int,
    heart_rate: float,
    amplitude: float,
    noise_scale: float,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(round(seconds * sample_rate))
    t = np.arange(n, dtype=np.float32) / sample_rate
    signal = 0.025 * np.sin(2 * np.pi * 0.25 * t)
    signal += 0.010 * np.sin(2 * np.pi * 0.05 * t)

    rr = 60.0 / heart_rate
    beat_times = np.arange(0.4, seconds + rr, rr)
    for beat in beat_times:
        qrs = np.exp(-0.5 * ((t - beat) / 0.018) ** 2)
        q = np.exp(-0.5 * ((t - (beat - 0.025)) / 0.010) ** 2)
        s = np.exp(-0.5 * ((t - (beat + 0.030)) / 0.012) ** 2)
        p = np.exp(-0.5 * ((t - (beat - 0.160)) / 0.040) ** 2)
        tw = np.exp(-0.5 * ((t - (beat + 0.240)) / 0.070) ** 2)
        signal += amplitude * (1.00 * qrs - 0.22 * q - 0.28 * s + 0.08 * p + 0.24 * tw)

    signal += rng.normal(0.0, noise_scale, size=n).astype(np.float32)
    return signal.astype(np.float32)


def synthetic_multilead(
    seconds: float,
    sample_rate: int,
    heart_rate: float,
    amplitude: float,
    noise_scale: float,
    seed: int,
) -> np.ndarray:
    lead_i = synthetic_lead(seconds, sample_rate, heart_rate, amplitude, noise_scale, seed)
    weights = np.array([1.00, 0.82, -0.45, 0.28, -0.30, 0.55, 0.70, 0.92, 1.05, 0.88, 0.62, 0.42], dtype=np.float32)
    offsets = np.linspace(-0.015, 0.015, num=12, dtype=np.float32)
    leads = []
    for idx, weight in enumerate(weights):
        drift = offsets[idx] * np.sin(2 * np.pi * 0.12 * np.arange(lead_i.size) / sample_rate)
        leads.append((weight * lead_i + drift).astype(np.float32))
    return np.stack(leads, axis=1)


def write_demo_metadata() -> None:
    rows = [
        ["test_data/ecg_npy/demo_ecg_000.npy", "demo_ecg_000.npy", "demo_sample_000", "synthetic demo input"],
        ["test_data/ecg_npy/demo_ecg_001.npy", "demo_ecg_001.npy", "demo_sample_001", "synthetic demo input"],
        ["test_data/ecg_npy/demo_ecg_002.npy", "demo_ecg_002.npy", "demo_sample_002", "synthetic demo input"],
        ["test_data/ecg_npy/demo_ecg_003.npy", "demo_ecg_003.npy", "demo_sample_003", "synthetic demo input"],
    ]
    with open(METADATA_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "deployment_ecg_path",
            "deployment_ecg_file",
            "demo_sample_id",
            "demo_note",
        ])
        writer.writerows(rows)


def main() -> None:
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    profiles = [
        (66.0, 0.85, 0.006, 9020),
        (74.0, 0.75, 0.008, 9021),
        (82.0, 0.95, 0.010, 9022),
        (58.0, 1.05, 0.007, 9023),
    ]
    for idx, (heart_rate, amplitude, noise_scale, seed) in enumerate(profiles):
        data = synthetic_multilead(
            seconds=10.0,
            sample_rate=500,
            heart_rate=heart_rate,
            amplitude=amplitude,
            noise_scale=noise_scale,
            seed=seed,
        )
        np.save(DEMO_DIR / f"demo_ecg_{idx:03d}.npy", data.astype(np.float32))

    handheld = synthetic_multilead(
        seconds=30.0,
        sample_rate=500,
        heart_rate=72.0,
        amplitude=0.90,
        noise_scale=0.008,
        seed=9030,
    )
    np.save(DEMO_DIR / "demo_handheld_30s.npy", handheld.astype(np.float32))
    write_demo_metadata()


if __name__ == "__main__":
    main()
