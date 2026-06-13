"""Compute ITA (Individual Typology Angle) from sample forehead regions.

Usage:
    python scripts/fitzpatrick_audit.py --input PATH/TO/IMAGES --out ita_report.json

This is a best-effort audit; for robust studies manual labeling is recommended.
"""
import argparse
import json
import os
from pathlib import Path
from typing import List

try:
    import cv2
    import numpy as np
except Exception:
    cv2 = None
    np = None


def compute_ita(img) -> float:
    # expects BGR image (cv2)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0].astype(float)
    b = lab[:, :, 2].astype(float)
    L_mean = L.mean()
    b_mean = b.mean()
    ita = (180 / 3.141592653589793) * cv2.phase(np.array([L_mean - 50]), np.array([b_mean]), angleInDegrees=False)[0]
    # The above uses cv2.phase in radians; convert to degrees
    # Fallback if cv2.phase not suitable:
    try:
        ita = (180.0 / 3.141592653589793) * np.arctan((L_mean - 50.0) / (b_mean + 1e-6))
    except Exception:
        ita = 0.0
    return float(ita)


def sample_forehead(img):
    h, w = img.shape[:2]
    # take a small rectangle near top-center (10%-25% height)
    y1 = int(h * 0.08)
    y2 = int(h * 0.25)
    x1 = int(w * 0.35)
    x2 = int(w * 0.65)
    return img[y1:y2, x1:x2]


def audit_images(folder: str) -> dict:
    p = Path(folder)
    results = {}
    for f in sorted(p.rglob('*.jpg')) + sorted(p.rglob('*.png')):
        img = cv2.imread(str(f))
        if img is None:
            continue
        fore = sample_forehead(img)
        ita = compute_ita(fore)
        results[str(f)] = ita
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--out', default='ita_report.json')
    args = ap.parse_args()
    report = audit_images(args.input)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    print('Wrote ITA report to', args.out)


if __name__ == '__main__':
    main()
