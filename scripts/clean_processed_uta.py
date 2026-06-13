"""Assess and clean data/processed/uta-rldd

Usage:
  python scripts/clean_processed_uta.py --dir data/processed/uta-rldd --backup data/processed/uta-rldd_collisions_backup --move

- Finds .npy files with numeric-only stems (e.g. "0.npy", "5.npy") which are likely collisions
- Moves them to the backup folder (if --move)
- Validates remaining .npy files by loading and checking shape/NaN/Inf
"""
import argparse
from pathlib import Path
import numpy as np
import shutil
import datetime


def is_numeric_stem(p: Path):
    return p.stem.isdigit()


def main():
    p = Path(args.dir)
    if not p.exists():
        print('Directory not found:', p)
        return
    npys = sorted(p.rglob('*.npy'))
    print('Found', len(npys), '.npy files in', p)

    numeric = [f for f in npys if is_numeric_stem(f)]
    print('Numeric-stem files (likely collisions):', len(numeric))
    for f in numeric[:20]:
        print('  ', f.name)

    if args.move and numeric:
        backup = Path(args.backup)
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        backup = backup.with_name(backup.name + '_' + ts)
        backup.mkdir(parents=True, exist_ok=True)
        for f in numeric:
            dest = backup / f.name
            try:
                shutil.move(str(f), str(dest))
            except Exception as e:
                print('Failed move', f, e)
        print('Moved', len(numeric), 'files to', backup)

    # validate remaining
    npys_after = sorted(p.rglob('*.npy'))
    issues = []
    for f in npys_after:
        try:
            arr = np.load(str(f))
            if not isinstance(arr, np.ndarray):
                issues.append((f, 'not ndarray'))
            else:
                if arr.size == 0:
                    issues.append((f, 'empty'))
                if np.isnan(arr).any():
                    issues.append((f, 'has NaN'))
                if np.isinf(arr).any():
                    issues.append((f, 'has Inf'))
                if arr.ndim != 2 or arr.shape[1] != 20:
                    issues.append((f, f'shape={arr.shape}'))
        except Exception as e:
            issues.append((f, f'load-error:{e}'))

    print('Post-clean files:', len(npys_after))
    if issues:
        print('Validation issues found:', len(issues))
        for f, reason in issues[:50]:
            print('  ', f.name, reason)
    else:
        print('No validation issues found')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', required=True)
    parser.add_argument('--backup', required=False, default='data/processed/uta-rldd_collisions_backup')
    parser.add_argument('--move', action='store_true', help='Move numeric collision files to backup')
    args = parser.parse_args()
    main()
