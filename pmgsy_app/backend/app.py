"""
app.py  –  Flask backend for PMGSY Scheme Classifier
Endpoints:
    POST /predict      – ML prediction + LLM explanation
    GET  /model-info   – model metadata (accuracy, feature importances)
    GET  /health       – liveness probe
    GET  /             – serves frontend/index.html (avoids file:// CORS issues)
"""

import os
import json
import logging
import requests
import numpy as np
import joblib
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR    = os.path.join(BASE_DIR, "..", "models")
FRONTEND_DIR = os.path.join(BASE_DIR, "..", "frontend")

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")
CORS(app)

# ── IBM watsonx credentials ──────────────────────────────────────────────────
WX_API_KEY    = os.getenv("WX_API_KEY",    "LYb4Yb_AusIOhtkqq6N2cQ4SryDad8TopwAFFjCvnsrZ")
WX_PROJECT_ID = os.getenv("WX_PROJECT_ID", "3e4960c0-aa26-4bec-bd22-f2b08a4427f8")
WX_MODEL_ID   = os.getenv("WX_MODEL_ID",   "ibm/granite-4-h-small")
WX_CHAT_URL   = "https://us-south.ml.cloud.ibm.com/ml/v1/text/chat?version=2023-05-29"
IAM_URL       = "https://iam.cloud.ibm.com/identity/token"

# ── Scheme descriptions for the LLM context ─────────────────────────────────
SCHEME_CONTEXT = {
    "PMGSY-I":    "PMGSY Phase I (2000–present): first-generation rural connectivity scheme targeting unconnected habitations with all-weather roads.",
    "PMGSY-II":   "PMGSY Phase II (2013–present): focuses on upgradation of existing rural roads connecting higher-order roads.",
    "PMGSY-III":  "PMGSY Phase III (2019–present): consolidation of existing rural road network through upgradation of Through Routes and Major Rural Links.",
    "RCPLWEA":    "Roads & Bridges in Left Wing Extremism Affected Areas: special sub-scheme under PMGSY targeting LWE-affected districts for strategic road connectivity.",
    "PM-JANMAN":  "PM-JANMAN (2023–present): targets road connectivity for Particularly Vulnerable Tribal Groups (PVTGs) in forest and remote tribal habitations.",
}

# ── Load ML artifacts at startup ─────────────────────────────────────────────
def load_artifacts():
    clf = joblib.load(os.path.join(MODEL_DIR, "pmgsy_classifier.pkl"))
    le  = joblib.load(os.path.join(MODEL_DIR, "label_encoder.pkl"))
    with open(os.path.join(MODEL_DIR, "model_meta.json")) as f:
        meta = json.load(f)
    log.info("ML artifacts loaded.  Classes: %s", meta["classes"])
    return clf, le, meta

try:
    CLF, LE, META = load_artifacts()
except Exception as exc:
    log.error("Could not load ML artifacts: %s", exc)
    log.error("Run  python backend/train_model.py  first.")
    CLF, LE, META = None, None, {}


# ── IBM IAM token cache ───────────────────────────────────────────────────────
_iam_token_cache: dict = {}

def get_iam_token() -> str:
    import time
    now = time.time()
    if _iam_token_cache.get("token") and now < _iam_token_cache.get("expires", 0) - 60:
        return _iam_token_cache["token"]
    resp = requests.post(
        IAM_URL,
        data={
            "grant_type":    "urn:ibm:params:oauth:grant-type:apikey",
            "apikey":        WX_API_KEY,
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=20,
    )
    resp.raise_for_status()
    data = resp.json()
    _iam_token_cache["token"]   = data["access_token"]
    _iam_token_cache["expires"] = now + data.get("expires_in", 3600)
    return _iam_token_cache["token"]


def call_granite(prompt: str) -> str:
    """Send a prompt to IBM Granite and return the assistant reply."""
    token = get_iam_token()
    # NOTE: watsonx.ai chat API requires max_tokens & temperature at the TOP
    # level of the payload — NOT nested under a "parameters" key.
    payload = {
        "model_id":   WX_MODEL_ID,
        "project_id": WX_PROJECT_ID,
        "messages": [
            {
                "role":    "system",
                "content": (
                    "You are a domain expert on India's Pradhan Mantri Gram Sadak Yojana (PMGSY) "
                    "rural road and bridge construction program. Provide clear, factual, concise "
                    "explanations in plain English (2-4 sentences). Avoid jargon."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "max_tokens":  300,
        "temperature": 0.3,
    }
    resp = requests.post(
        WX_CHAT_URL,
        json=payload,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type":  "application/json",
        },
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


def build_explanation_prompt(inputs: dict, predicted_scheme: str, probabilities: dict) -> str:
    ctx = SCHEME_CONTEXT.get(predicted_scheme, predicted_scheme)
    top2 = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)[:2]
    top2_str = ", ".join(f"{k} ({v:.0%})" for k, v in top2)
    return (
        f"A rural infrastructure project has the following characteristics:\n"
        f"- Road works sanctioned: {inputs.get('NO_OF_ROAD_WORK_SANCTIONED', 0)}, "
        f"total length {inputs.get('LENGTH_OF_ROAD_WORK_SANCTIONED', 0):.2f} km\n"
        f"- Bridges sanctioned: {inputs.get('NO_OF_BRIDGES_SANCTIONED', 0)}\n"
        f"- Sanctioned cost: ₹{inputs.get('COST_OF_WORKS_SANCTIONED', 0):.2f} Cr\n"
        f"- Roads completed: {inputs.get('NO_OF_ROAD_WORKS_COMPLETED', 0)} "
        f"({inputs.get('LENGTH_OF_ROAD_WORK_COMPLETED', 0):.2f} km), "
        f"Bridges completed: {inputs.get('NO_OF_BRIDGES_COMPLETED', 0)}\n"
        f"- Expenditure: ₹{inputs.get('EXPENDITURE_OCCURED', 0):.2f} Cr\n"
        f"- Balance roads: {inputs.get('NO_OF_ROAD_WORKS_BALANCE', 0)} "
        f"({inputs.get('LENGTH_OF_ROAD_WORK_BALANCE', 0):.2f} km), "
        f"Bridges balance: {inputs.get('NO_OF_BRIDGES_BALANCE', 0)}\n\n"
        f"The ML model classified this project under '{predicted_scheme}' "
        f"(top predictions: {top2_str}).\n\n"
        f"Scheme context: {ctx}\n\n"
        f"In 2–4 sentences, explain WHY this project most likely belongs to '{predicted_scheme}' "
        f"based on its physical and financial profile, and what that scheme aims to achieve."
    )


def engineer_features(d: dict) -> list:
    """Reproduce the feature engineering from train_model.py."""
    road_s  = float(d.get("NO_OF_ROAD_WORK_SANCTIONED", 0))
    len_s   = float(d.get("LENGTH_OF_ROAD_WORK_SANCTIONED", 0))
    br_s    = float(d.get("NO_OF_BRIDGES_SANCTIONED", 0))
    cost_s  = float(d.get("COST_OF_WORKS_SANCTIONED", 0))
    road_c  = float(d.get("NO_OF_ROAD_WORKS_COMPLETED", 0))
    len_c   = float(d.get("LENGTH_OF_ROAD_WORK_COMPLETED", 0))
    br_c    = float(d.get("NO_OF_BRIDGES_COMPLETED", 0))
    exp     = float(d.get("EXPENDITURE_OCCURED", 0))
    road_b  = float(d.get("NO_OF_ROAD_WORKS_BALANCE", 0))
    len_b   = float(d.get("LENGTH_OF_ROAD_WORK_BALANCE", 0))
    br_b    = float(d.get("NO_OF_BRIDGES_BALANCE", 0))

    eps = 1e-9
    completion_rate  = road_c  / (road_s  + eps)
    expenditure_rate = exp     / (cost_s  + eps)
    cost_per_km      = cost_s  / (len_s   + eps)
    bridge_ratio     = br_s    / (road_s  + eps)
    balance_ratio    = road_b  / (road_s  + eps)

    return [
        road_s, len_s, br_s, cost_s,
        road_c, len_c, br_c, exp,
        road_b, len_b, br_b,
        completion_rate, expenditure_rate, cost_per_km, bridge_ratio, balance_ratio,
    ]


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the frontend so it runs on http://localhost:5000 (no file:// CORS issues)."""
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/health")
def health():
    model_loaded = CLF is not None
    return jsonify({"status": "ok", "model_loaded": model_loaded}), 200


@app.route("/model-info")
def model_info():
    if not META:
        return jsonify({"error": "Model not loaded"}), 503
    return jsonify(META), 200


@app.route("/predict", methods=["POST"])
def predict():
    if CLF is None:
        return jsonify({"error": "Model not loaded. Run train_model.py first."}), 503

    data = request.get_json(force=True)

    # Build feature vector
    try:
        features = np.array([engineer_features(data)])
    except Exception as exc:
        return jsonify({"error": f"Feature engineering failed: {exc}"}), 400

    # ML prediction
    pred_idx   = int(CLF.predict(features)[0])
    pred_proba = CLF.predict_proba(features)[0]
    pred_scheme = LE.inverse_transform([pred_idx])[0]

    probabilities = {
        cls: round(float(prob), 4)
        for cls, prob in zip(LE.classes_, pred_proba)
    }

    # LLM explanation
    explanation = ""
    llm_error   = None
    try:
        prompt = build_explanation_prompt(data, pred_scheme, probabilities)
        explanation = call_granite(prompt)
    except Exception as exc:
        log.warning("LLM call failed: %s", exc)
        llm_error = str(exc)
        explanation = (
            f"This project was classified as '{pred_scheme}'. "
            + SCHEME_CONTEXT.get(pred_scheme, "")
        )

    return jsonify({
        "predicted_scheme": pred_scheme,
        "probabilities":    probabilities,
        "explanation":      explanation,
        "llm_error":        llm_error,
    }), 200


if __name__ == "__main__":
    # use_reloader=False prevents the double-start issue on Windows
    app.run(debug=True, host="0.0.0.0", port=5000, use_reloader=False)
