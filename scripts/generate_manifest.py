"""Generate a simple manifest.csv for a processed directory.
Columns: session_id,npy_path
"""
import argparse
from pathlib import Path
import csv

parser = argparse.ArgumentParser()
parser.add_argument('--dir', required=True)
parser.add_argument('--out', required=False)
args = parser.parse_args()

p = Path(args.dir)
if not p.exists():
    print('Directory not found:', p); raise SystemExit(1)

out_path = Path(args.out) if args.out else p / 'manifest.csv'
files = sorted(p.rglob('*.npy'))
print('Found', len(files), 'npy files; writing manifest to', out_path)

with open(out_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['session_id', 'npy_path'])
    for fp in files:
        writer.writerow([fp.stem, str(fp.relative_to(p))])

print('Wrote', out_path)
