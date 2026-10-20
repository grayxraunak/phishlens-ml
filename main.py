"""PhishLens: explainable URL risk triage demo.

This is an educational prototype. Its bundled dataset is small and illustrative;
do not use its predictions to make real security decisions.
"""
import argparse
import re
from urllib.parse import urlparse

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

DATA_PATH = "data/sample_urls.csv"

SUSPICIOUS_WORDS = {
    "login", "verify", "update", "secure", "account", "signin",
    "password", "wallet", "confirm", "billing", "support"
}

def extract_features(raw_url: str) -> dict:
    """Extract transparent lexical URL features without visiting the URL."""
    value = str(raw_url).strip()
    candidate = value if "://" in value else "http://" + value
    parsed = urlparse(candidate)
    host = (parsed.hostname or "").lower()
    path = (parsed.path or "").lower()
    full = value.lower()
    tokens = re.findall(r"[a-z]+", full)

    return {
        "url_length": len(value),
        "hostname_length": len(host),
        "path_length": len(path),
        "dot_count": full.count("."),
        "hyphen_count": full.count("-"),
        "at_symbol": int("@" in value),
        "ip_address_host": int(bool(re.fullmatch(r"(?:\d{1,3}\.){3}\d{1,3}", host))),
        "https_scheme": int(parsed.scheme.lower() == "https"),
        "subdomain_count": max(0, len(host.split(".")) - 2),
        "suspicious_word_count": sum(word in tokens for word in SUSPICIOUS_WORDS),
        "digit_count": sum(ch.isdigit() for ch in value),
        "query_length": len(parsed.query),
    }

FEATURES = list(extract_features("https://example.com"))

def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    required = {"url", "label"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {sorted(required)}")
    X = pd.DataFrame([extract_features(url) for url in df["url"]])
    y = df["label"].astype(int)
    return X, y

def build_models():
    return {
        "Logistic Regression": Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42))
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=42, max_depth=8
        ),
    }

def evaluate(data_path=DATA_PATH):
    X, y = load_data(data_path)
    if y.nunique() != 2 or y.value_counts().min() < 2:
        raise ValueError("Dataset needs at least two examples of each label.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    rows, fitted = [], {}
    for name, model in build_models().items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        rows.append({
            "model": name,
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred, zero_division=0),
            "f1": f1_score(y_test, pred, zero_division=0),
        })
        fitted[name] = model
    results = pd.DataFrame(rows).sort_values("f1", ascending=False)
    winner_name = results.iloc[0]["model"]
    winner = fitted[winner_name]
    pred = winner.predict(X_test)
    return results, winner, X_test, y_test, pred, winner_name

def explain(url):
    f = extract_features(url)
    signals = []
    if f["https_scheme"] == 0:
        signals.append("URL does not use HTTPS")
    if f["ip_address_host"]:
        signals.append("host is an IPv4 address")
    if f["at_symbol"]:
        signals.append("contains an @ symbol")
    if f["suspicious_word_count"]:
        signals.append(f'{f["suspicious_word_count"]} suspicious keyword(s) detected')
    if f["subdomain_count"] >= 3:
        signals.append("many subdomain levels")
    if f["url_length"] > 75:
        signals.append("unusually long URL")
    if f["hyphen_count"] >= 3:
        signals.append("multiple hyphens")
    return signals or ["No obvious lexical warning signals found"]

def main():
    parser = argparse.ArgumentParser(description="PhishLens URL risk triage demo")
    parser.add_argument("--url", help="URL to analyse (never visited)")
    parser.add_argument("--data", default=DATA_PATH, help="CSV with url,label columns")
    args = parser.parse_args()

    results, model, X_test, y_test, pred, winner = evaluate(args.data)
    print("\nPhishLens | model evaluation")
    print(results.to_string(index=False, formatters={
        "accuracy": "{:.3f}".format, "precision": "{:.3f}".format,
        "recall": "{:.3f}".format, "f1": "{:.3f}".format
    }))
    print(f"\nSelected baseline by held-out F1: {winner}")
    print("\nHeld-out classification report:")
    print(classification_report(y_test, pred, target_names=["benign (0)", "phishing-like (1)"], zero_division=0))
    print("Confusion matrix [rows=true, columns=predicted]:")
    print(confusion_matrix(y_test, pred))

    if args.url:
        features = pd.DataFrame([extract_features(args.url)])
        label = int(model.predict(features)[0])
        probabilities = model.predict_proba(features)[0]
        confidence = float(max(probabilities))
        print("\nURL triage result")
        print(f"URL: {args.url}")
        print(f"Model label: {'PHISHING-LIKE' if label else 'BENIGN-LIKE'}")
        print(f"Model probability for predicted class: {confidence:.1%}")
        print("Signals to review:")
        for signal in explain(args.url):
            print(f"  - {signal}")
        print("\nWarning: this is a small educational demo, not a security verdict.")

if __name__ == "__main__":
    main()
