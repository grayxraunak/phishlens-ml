# PhishLens 🔎
### Explainable URL risk triage with classical machine learning

PhishLens is a small, reproducible ML project that extracts interpretable lexical features from a URL, compares two classical classifiers, and reports held-out evaluation metrics. It does **not** open or fetch submitted URLs.

> **Important limitation:** the included CSV is a small, illustrative teaching dataset. Metrics from it do not estimate real-world phishing-detection performance. Do not use this project as a security product or rely on its predictions to decide whether a link is safe.

## What it demonstrates

- Feature engineering from raw URL strings
- A reproducible, stratified train/test split
- Baseline comparison: Logistic Regression vs Random Forest
- Evaluation with precision, recall, F1, accuracy and a confusion matrix
- Lightweight, human-readable warning signals for each submitted URL
- A CLI interface with no network requests

## Feature set

The model uses URL length, hostname length, path length, dots, hyphens, `@` symbol, IPv4 host indicator, HTTPS scheme, subdomain count, suspicious keyword count, digit count and query length.

These features are intentionally transparent. They are not sufficient to reliably identify modern phishing attacks.

## Quick start

Python 3.10+ is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate       # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
python main.py
```

Analyse a URL string without visiting it:

```bash
python main.py --url "http://secure-login-example.test/account/verify"
```

The `.test` domain is reserved for testing. The script analyses the string only and never makes a network request.

## Dataset format

`data/sample_urls.csv` has two columns:

- `url`: URL string
- `label`: `0` for benign-like and `1` for phishing-like

The bundled examples are synthetic and illustrative, not a representative real-world corpus. To conduct meaningful research, replace them with a properly sourced, licensed dataset, preserve its provenance, deduplicate samples, and use time-aware or domain-aware splitting to reduce leakage.

## Methodology

1. Extract URL lexical features using Python's standard library.
2. Split the labelled examples into training and held-out test subsets with stratification and a fixed random seed.
3. Train Logistic Regression and Random Forest on the same split.
4. Compare precision, recall and F1. Select the baseline with the highest held-out F1 for the optional CLI prediction.
5. Display the confusion matrix and a classification report.

Because the sample dataset is deliberately small and constructed for teaching, the reported numbers are only a code-path demonstration. They are not evidence that the model generalises to real phishing URLs. In particular, a probability from this model is not a calibrated real-world risk estimate.

## Project structure

```text
phishlens/
├── data/
│   └── sample_urls.csv
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Limitations and responsible use

- Educational prototype only; not suitable for production security decisions.
- Dataset is synthetic, small and potentially easy to separate by superficial patterns.
- No DNS, domain age, certificate, redirect, page-content or reputation checks.
- HTTPS does not guarantee that a website is legitimate.
- A model label of `benign-like` does not mean a URL is safe.
- Do not test using live suspicious links by opening them. The code itself does not visit URLs.

## Next steps

- Replace the sample data with a documented public dataset.
- Add cross-validation and domain-aware/time-aware evaluation.
- Tune the decision threshold based on false-positive and false-negative costs.
- Add calibration and explainability analysis.
- Build a small Streamlit interface only after validating the dataset and evaluation design.

## Author

Built as a learning project to practise Python, feature engineering, model comparison and responsible ML evaluation.
