# Real vs AI-Generated Speech Detection

Course project implementation for Spoken Language Processing (Summer 2026).

## What is included

- Leakage-aware data indexing and group-based splits.
- MFCC, LFCC, and spectral/time-domain feature extraction.
- SVM and Random Forest classifiers.
- Accuracy, precision, recall, F1, ROC-AUC, EER, confusion matrices, and ROC plots.
- Unseen-generator generalization experiment.
- Command-line prediction and Streamlit live demo.

## Dataset layout

WaveFake distributions differ slightly. The loader accepts either:

```text
data/wavefake/
  real/<speaker_or_source>/*.wav
  fake/<generator>/<speaker_or_source>/*.wav
```

or folders whose names contain `real`, `bonafide`, `fake`, `generated`, or known
generator names. If automatic inference is uncertain, create `metadata.csv`:

```csv
path,label,generator,speaker
real/p01/a.wav,0,real,p01
fake/hifiGAN/p01/a.wav,1,hifiGAN,p01
```

`label=0` means real/bonafide and `label=1` means AI-generated/spoof.

## Run

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt

python run_experiments.py --config config.yaml
python predict.py --model outputs/best_model.joblib --audio sample.wav
streamlit run app.py
```

For a quick code check without downloading a dataset:

```bash
python make_smoke_dataset.py
python run_experiments.py --config config.yaml --dataset-root data/smoke
```

The smoke dataset is only a software test and must never be reported as a real
scientific result. Submit the metrics generated from WaveFake or ASVspoof.

## Reproducibility and leakage control

- All random operations use seed 42.
- Standardization is fitted only on the training fold through an sklearn Pipeline.
- Hyperparameters are selected using grouped cross-validation.
- Files from the same inferred speaker/source group do not cross train/test boundaries.
- The generalization experiment removes one fake generator from training and uses it only for testing.

## Before submission

1. Replace `TEAM_MEMBER_*` in the report and slides.
2. Run on the real dataset.
3. Copy the generated values from `outputs/summary.csv` into the report.
4. Update the contribution table honestly.
5. Keep the AI-use disclosure and all references.

