"""Classify SMS messages as spam or ham with the trained model.

Usage:
    python predict.py "WINNER!! You have been selected to receive a £900 prize. Call 09061701461"
    python predict.py --file messages.txt      # one message per line
"""
import argparse
import sys

import joblib

# the model's preprocessing function lives in the notebook; define it here too so the pickle can load
import re


def normalise(t):
    t = t.lower()
    t = re.sub(r"https?://\S+|www\.\S+", " URL ", t)
    t = re.sub(r"£\s?\d+|\$\s?\d+|\d+\s?p\b", " MONEY ", t)
    t = re.sub(r"\b\d{5,}\b", " LONGNUM ", t)
    return t


def main():
    parser = argparse.ArgumentParser(description="SMS spam classifier")
    parser.add_argument("message", nargs="?", help="message text")
    parser.add_argument("--file", help="text file with one message per line")
    parser.add_argument("--model", default="models/spam_model.joblib")
    args = parser.parse_args()

    if args.file:
        with open(args.file, encoding="utf-8") as f:
            messages = [line.strip() for line in f if line.strip()]
    elif args.message:
        messages = [args.message]
    else:
        parser.print_help()
        sys.exit(1)

    bundle = joblib.load(args.model)
    model, threshold = bundle["model"], bundle["threshold"]
    for msg, p in zip(messages, model.predict_proba(messages)[:, 1]):
        label = "SPAM" if p >= threshold else "ham"
        print(f"{label:<5} ({p:.2f})  {msg[:80]}")


if __name__ == "__main__":
    main()
