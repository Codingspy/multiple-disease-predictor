# Multiple Disease Predictor — MVP-1

MVP-1 includes:
- Heart Disease Predictor (demo model trained on synthetic but structured data)
- Symptom Checker (alpha) via rule-based mapping.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt

# Train and save artifacts
cd training/heart
python train_heart.py
cd ../../

# Run app
streamlit run app/app.py
