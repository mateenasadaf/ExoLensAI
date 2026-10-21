# ExoLens AI — Streamlit dashboard setup

## 1. Export artifacts from your existing Colab notebook

Run this cell after the model, `features`, `candidates`, `candidate_results`, and `knowledge_base` variables exist:

```python
import json
import joblib
from pathlib import Path

ARTIFACTS = Path("artifacts")
ARTIFACTS.mkdir(exist_ok=True)

joblib.dump(rf_model, ARTIFACTS / "rf_model.joblib")

with open(ARTIFACTS / "features.json", "w", encoding="utf-8") as f:
    json.dump(features, f, indent=2)

# Combine the candidate measurements with the prediction outputs.
candidate_export = candidates.merge(
    candidate_results[["kepid", "predicted_probability", "predicted_class"]],
    on="kepid",
    how="left"
)
candidate_export.to_csv(
    ARTIFACTS / "candidate_observations.csv",
    index=False
)

with open(ARTIFACTS / "knowledge_base.json", "w", encoding="utf-8") as f:
    json.dump(knowledge_base, f, indent=2)

print("Artifacts created:")
for path in ARTIFACTS.iterdir():
    print(path, path.stat().st_size, "bytes")
```

Download the `artifacts` files from Colab, preserving the folder structure.

## 2. Local project structure

```text
ExoLensAI/
├── app.py
├── requirements.txt
└── artifacts/
    ├── rf_model.joblib
    ├── features.json
    ├── candidate_observations.csv
    └── knowledge_base.json
```

Put `app.py` and `requirements.txt` beside the `artifacts` folder.

## 3. Run locally

```bash
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\activate
```

Install and run:
```bash
pip install -r requirements.txt
streamlit run app.py
```

Streamlit will print a local URL, typically `http://localhost:8501`.

## Important
The app uses the trained Random Forest pipeline from the notebook. Keep the model and preprocessing pipeline together. The dashboard describes the output as a model prediction, not scientific confirmation.
