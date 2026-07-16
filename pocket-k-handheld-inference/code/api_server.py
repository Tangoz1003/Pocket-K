import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel


CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from deploy_test_inference import (  # noqa: E402
    PACKAGE_ROOT,
    get_preprocessing_config,
    get_threshold_config,
    load_signal,
    load_deployment_model,
    resolve_package_path,
    assign_risk_group,
)
from handheld_pipeline import (  # noqa: E402
    get_handheld_config,
    make_torch_inference_fn,
    predict_handheld_path as predict_handheld_measurement_path,
)


CONFIG_PATH = PACKAGE_ROOT / "config" / "deployment_config.json"

app = FastAPI(title="Pocket-K Handheld Inference API")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = json.load(f)

low_threshold, high_threshold = get_threshold_config(config)

model = load_deployment_model(config, device)
torch_inference_fn = make_torch_inference_fn(model, device)
preprocessing = get_preprocessing_config(config)


class PredictPathRequest(BaseModel):
    ecg_path: str
    sample_id: Optional[str] = None


class HandheldPredictPathRequest(BaseModel):
    ecg_path: str
    sample_id: Optional[str] = None
    source_sample_rate: int = 500


def predict_array(ecg_array: np.ndarray) -> dict:
    temp_path = PACKAGE_ROOT / "test_outputs" / "_api_tmp_input.npy"
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(temp_path, ecg_array)
    try:
        signal = load_signal(
            temp_path,
            sample_rate=int(config["sample_rate"]),
            signal_length=int(config["signal_length"]),
            apply_filter=bool(config["apply_filter"]),
            **preprocessing,
        )
    finally:
        temp_path.unlink(missing_ok=True)

    x = torch.from_numpy(signal).unsqueeze(0).to(device)
    with torch.inference_mode():
        pred_prob = torch.sigmoid(model(x)).detach().cpu().numpy().reshape(-1)[0]

    pred_prob = float(pred_prob)
    risk_group = assign_risk_group(pred_prob, low_threshold, high_threshold)
    result = {
        "pred_prob": pred_prob,
        "model_name": config["model_name"],
        "device": str(device),
    }
    if risk_group is not None:
        result["risk_group"] = risk_group
    return result


def predict_path(ecg_path: Path) -> dict:
    signal = load_signal(
        ecg_path,
        sample_rate=int(config["sample_rate"]),
        signal_length=int(config["signal_length"]),
        apply_filter=bool(config["apply_filter"]),
        **preprocessing,
    )
    x = torch.from_numpy(signal).unsqueeze(0).to(device)
    with torch.inference_mode():
        pred_prob = torch.sigmoid(model(x)).detach().cpu().numpy().reshape(-1)[0]

    pred_prob = float(pred_prob)
    risk_group = assign_risk_group(pred_prob, low_threshold, high_threshold)
    result = {
        "pred_prob": pred_prob,
        "model_name": config["model_name"],
        "device": str(device),
    }
    if risk_group is not None:
        result["risk_group"] = risk_group
    return result


def predict_handheld_path(ecg_path: Path, source_sample_rate: int) -> dict:
    result = predict_handheld_measurement_path(
        ecg_path,
        inference_fn=torch_inference_fn,
        source_sample_rate=source_sample_rate,
        low_threshold=low_threshold,
        high_threshold=high_threshold,
        **get_handheld_config(config),
    )
    result["model_name"] = config["model_name"]
    result["device"] = str(device)
    return result


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "device": str(device),
        "model_name": config["model_name"],
        "thresholds_configured": low_threshold is not None and high_threshold is not None,
    }


@app.post("/api/predict")
def predict(request: PredictPathRequest):
    ecg_path = resolve_package_path(request.ecg_path, PACKAGE_ROOT)
    if not ecg_path.exists():
        return {"error": f"ECG file not found: {ecg_path}"}

    result = predict_path(ecg_path)
    if request.sample_id is not None:
        result["sample_id"] = request.sample_id
    return result


@app.post("/api/predict_upload")
async def predict_upload(file: UploadFile = File(...)):
    data = await file.read()
    temp_path = PACKAGE_ROOT / "test_outputs" / f"_upload_{file.filename}"
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path.write_bytes(data)
    try:
        result = predict_path(temp_path)
    finally:
        temp_path.unlink(missing_ok=True)
    result["filename"] = file.filename
    return result


@app.post("/api/predict_handheld")
def predict_handheld(request: HandheldPredictPathRequest):
    ecg_path = resolve_package_path(request.ecg_path, PACKAGE_ROOT)
    if not ecg_path.exists():
        return {"error": f"ECG file not found: {ecg_path}"}

    result = predict_handheld_path(ecg_path, request.source_sample_rate)
    if request.sample_id is not None:
        result["sample_id"] = request.sample_id
    return result


@app.post("/api/predict_handheld_upload")
async def predict_handheld_upload(
    file: UploadFile = File(...),
    source_sample_rate: int = 500,
):
    data = await file.read()
    temp_path = PACKAGE_ROOT / "test_outputs" / f"_handheld_upload_{file.filename}"
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path.write_bytes(data)
    try:
        result = predict_handheld_path(temp_path, source_sample_rate)
    finally:
        temp_path.unlink(missing_ok=True)
    result["filename"] = file.filename
    return result
