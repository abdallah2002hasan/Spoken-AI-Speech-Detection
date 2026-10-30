import argparse
import joblib
from features import extract


def predict(bundle_path, audio_path):
    bundle = joblib.load(bundle_path)
    x = extract(audio_path, bundle["feature"], bundle["config"])[None, :]
    p = float(bundle["model"].predict_proba(x)[0, 1])
    return {"label": "AI-GENERATED" if p >= 0.5 else "REAL", "ai_probability": p}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True)
    ap.add_argument("--audio", required=True); args = ap.parse_args()
    result = predict(args.model, args.audio)
    print(f"Prediction: {result['label']} | AI probability: {result['ai_probability']:.4f}")

