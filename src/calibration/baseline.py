"""Simple baseline collection, storage, and loading.
Writes per-user baseline JSON into `data/baselines/`. If `cryptography` is
available, uses `Fernet` to encrypt stored blobs; otherwise stores plain JSON
and logs a warning.
"""
import json
import os
from datetime import datetime
from typing import Dict, Tuple

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'baselines'))

os.makedirs(BASE_DIR, exist_ok=True)

try:
    from cryptography.fernet import Fernet
    HAS_FERNET = True
except Exception:
    HAS_FERNET = False


def store_baseline(user_id: str, baseline: Dict, key: bytes = None) -> str:
    """Store `baseline` for `user_id`. If `key` provided and Fernet available,
    the data is encrypted.
    """
    filename = os.path.join(BASE_DIR, f"{user_id}.baseline")
    payload = json.dumps({
        'mean': baseline.get('mean'),
        'std': baseline.get('std'),
        'timestamp': baseline.get('timestamp', datetime.utcnow().isoformat())
    }).encode('utf-8')

    if HAS_FERNET and key:
        f = Fernet(key)
        blob = f.encrypt(payload)
        with open(filename, 'wb') as fobj:
            fobj.write(blob)
    else:
        # Fallback: write plaintext JSON (user should enable cryptography)
        with open(filename + '.json', 'w', encoding='utf-8') as fobj:
            fobj.write(payload.decode('utf-8'))

    return filename


def load_baseline(user_id: str, key: bytes = None) -> Dict:
    """Load baseline for `user_id`. Handles decryption if encrypted."""
    filename = os.path.join(BASE_DIR, f"{user_id}.baseline")

    # Try encrypted first
    if os.path.exists(filename):
        with open(filename, 'rb') as fobj:
            blob = fobj.read()
        if HAS_FERNET and key:
            f = Fernet(key)
            payload = f.decrypt(blob)
            return json.loads(payload.decode('utf-8'))
        else:
            raise ValueError("Baseline is encrypted but no key/Fernet provided")

    # Try plaintext fallback
    plaintext_filename = filename + '.json'
    if os.path.exists(plaintext_filename):
        with open(plaintext_filename, 'r', encoding='utf-8') as fobj:
            return json.load(fobj)

    return None
