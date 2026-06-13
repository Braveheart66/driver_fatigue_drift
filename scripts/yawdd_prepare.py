"""Prepare YawDD dataset: unzip, process videos into 5s feature windows, create manifest.

Usage:
  python scripts/yawdd_prepare.py --zip path/to/YawDD.zip --out-dir data/raw/yawdd

Behavior:
 - Extracts zip to `--out-dir` (default `data/raw/yawdd`)
 - Finds video files and processes each via `scripts/process_video.py` logic
 - Writes per-session .npy files to `data/processed/yawdd/`
 - Generates `data/processed/yawdd/manifest.csv` with columns: session_id,npy_path,label
   - label: 1 if a companion label file indicates yawn, 0 if label indicates no yawn, -1 if unknown
"""
import argparse
import os
import zipfile
from pathlib import Path
import csv
import tempfile
import shutil


def find_label_for_file(video_path: Path):
    """Try to find a companion label file for `video_path`.
    Common patterns: same basename with .txt, .csv, or a labels/ folder.
    Returns: 1 (yawn), 0 (no yawn), or -1 (unknown)
    """
    base = video_path.with_suffix('')
    for ext in ('.txt', '.csv'):
        candidate = base.with_suffix(ext)
        if candidate.exists():
            # heuristic: look for keywords 'yawn' or '1' in the file
            try:
                text = candidate.read_text(encoding='utf-8', errors='ignore').lower()
                if 'yawn' in text or '\t1' in text or ',1' in text or '\n1' in text:
                    return 1
                if '\t0' in text or ',0' in text or '\n0' in text:
                    return 0
            except Exception:
                return -1

    # look for label folders
    labels_dir = video_path.parent / 'labels'
    if labels_dir.exists() and labels_dir.is_dir():
        for f in labels_dir.iterdir():
            if f.suffix.lower() in ('.txt', '.csv'):
                try:
                    text = f.read_text(encoding='utf-8', errors='ignore').lower()
                    if video_path.name in text and 'yawn' in text:
                        return 1
                except Exception:
                    pass

    return -1


def process_extracted_folder(folder: Path, out_processed: Path):
    # import the processing function
    from scripts.process_video import process_video

    out_processed.mkdir(parents=True, exist_ok=True)
    manifest = []

    video_exts = ('.mp4', '.avi', '.mov', '.mkv')
    for vid in sorted(folder.rglob('*')):
        if vid.is_file() and vid.suffix.lower() in video_exts:
            # create a unique session id from the relative path under folder
            rel = vid.relative_to(folder).with_suffix('')
            session_id = "_".join(rel.parts)
            session_id = session_id.replace(' ', '_').replace('.', '_')
            out_npy = out_processed / f"{session_id}.npy"
            try:
                process_video(vid, str(out_npy))
            except Exception as e:
                print('Failed processing', vid, e)
                continue
            label = find_label_for_file(vid)
            manifest.append((session_id, str(out_npy), label))

    # write manifest
    manifest_path = out_processed / 'manifest.csv'
    with open(manifest_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['session_id', 'npy_path', 'label'])
        for row in manifest:
            writer.writerow(row)

    print('Wrote manifest to', manifest_path)
    return manifest_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--zip', required=True)
    parser.add_argument('--out-dir', default='data/raw/yawdd')
    parser.add_argument('--processed-dir', default='data/processed/yawdd')
    args = parser.parse_args()

    zip_path = Path(args.zip)
    out_dir = Path(args.out_dir)
    processed_dir = Path(args.processed_dir)

    if not zip_path.exists():
        raise FileNotFoundError(str(zip_path))

    # Extract to out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(str(zip_path), 'r') as z:
        z.extractall(path=str(out_dir))

    # Process extracted folder
    process_extracted_folder(out_dir, processed_dir)


if __name__ == '__main__':
    main()
