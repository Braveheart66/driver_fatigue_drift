"""Fix common temporary suffix issues in processed .npy files.
Usage:
  python scripts/fix_tmp_suffixes.py --dir data/processed/uta-rldd --backup data/processed/uta-rldd_fix_backup --apply

By default the script runs in dry-run mode and reports actions. Use --apply to perform moves/renames.
"""
import argparse
from pathlib import Path
import datetime
import shutil
import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--dir', required=True)
parser.add_argument('--backup', required=False, default='data/processed/uta-rldd_fix_backup')
parser.add_argument('--apply', action='store_true', help='Actually perform renames/moves')
args = parser.parse_args()

p = Path(args.dir)
if not p.exists():
    print('Directory not found:', p)
    raise SystemExit(1)

candidates = sorted([f for f in p.rglob('*.npy') if f.name.endswith('.tmp.npy') or f.name.endswith('.npy.tmp') or f.name.endswith('.tmp')])
print('Found', len(candidates), 'tmp-like .npy files')
for f in candidates[:50]:
    print(' ', f.relative_to(p))

if not candidates:
    print('No tmp-like files found.'); raise SystemExit(0)

# Prepare backup
backup = Path(args.backup)
ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
backup = backup.with_name(backup.name + '_' + ts)
print('Backup (if needed):', backup)

if args.apply:
    backup.mkdir(parents=True, exist_ok=True)

moved = 0
renamed = 0
errors = []
for f in candidates:
    name = f.name
    # Handle common double-suffix patterns
    if name.endswith('.npy.tmp.npy'):
        target_name = name.replace('.npy.tmp.npy', '.npy')
    elif name.endswith('.tmp.npy'):
        target_name = name.replace('.tmp.npy', '.npy')
    elif name.endswith('.npy.tmp'):
        target_name = name.replace('.npy.tmp', '.npy')
    elif name.endswith('.tmp') and not name.endswith('.npy'):
        target_name = name[:-4] + '.npy'
    else:
        target_name = name.replace('.tmp', '')

    target = f.with_name(target_name)
    if target.exists():
        # conflict -> move to backup
        print('Conflict: target exists for', f.name, '-> moving to backup')
        if args.apply:
            try:
                dest = backup / f.name
                shutil.move(str(f), str(dest))
                moved += 1
            except Exception as e:
                errors.append((f, str(e)))
        continue
    else:
        print('Will rename', f.name, '->', target.name)
        if args.apply:
            try:
                os.replace(str(f), str(target))
                renamed += 1
                # quick validation: try loading
                try:
                    arr = np.load(str(target))
                    if not isinstance(arr, np.ndarray) or arr.ndim != 2:
                        print('  validation failed for', target.name, '-> moving to backup')
                        shutil.move(str(target), str(backup / target.name))
                        moved += 1
                except Exception as e:
                    print('  load failed for', target.name, '-> moving to backup', e)
                    shutil.move(str(target), str(backup / target.name))
                    moved += 1
            except Exception as e:
                errors.append((f, str(e)))

print('Renamed:', renamed, 'Moved to backup:', moved, 'Errors:', len(errors))
for f, e in errors[:20]:
    print('ERR', f, e)
