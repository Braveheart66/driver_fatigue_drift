"""Create subject-level splits for a dataset directory.

Directory layout expected (flexible):
 - dataset_root/
     - subject_001/
         - session1.mp4
         - session2.mp4
     - subject_002/
         - ...

Output: writes `splits.json` with lists of subjects for train/val/test.
"""
import argparse
import json
import random
from pathlib import Path


def subject_level_split(dataset_root: str, seed: int = 42, train_pct=0.7, val_pct=0.15, test_pct=0.15):
    p = Path(dataset_root)
    subjects = [d.name for d in p.iterdir() if d.is_dir()]
    random.Random(seed).shuffle(subjects)
    n = len(subjects)
    n_train = int(n * train_pct)
    n_val = int(n * val_pct)
    train = subjects[:n_train]
    val = subjects[n_train:n_train + n_val]
    test = subjects[n_train + n_val:]
    return {'train': train, 'val': val, 'test': test}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--out', default='splits.json')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    splits = subject_level_split(args.input, seed=args.seed)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(splits, f, indent=2)
    print('Wrote splits to', args.out)


if __name__ == '__main__':
    main()
