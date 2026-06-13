#!/usr/bin/env python3
"""Process an already-extracted dataset folder into per-session .npy features and manifest.
This runner uses a process pool so heavy imports (e.g., MediaPipe) are initialized once
per worker process, avoiding per-video Python startup overhead.
"""
import argparse
from pathlib import Path
import sys
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from scripts.yawdd_prepare import process_extracted_folder

# Import the worker wrapper from process_video so the heavy init code is colocated.
from scripts.process_video import run_process_video_task, init_global_extractor


def _sanitize_session_id(p: Path) -> str:
    s = "_".join(p.parts)
    s = s.replace(' ', '_')
    s = s.replace('.', '_')
    s = s.replace('/', '_')
    s = s.replace('\\', '_')
    return s


parser = argparse.ArgumentParser()
parser.add_argument('--dir', required=True)
parser.add_argument('--processed-dir', default='data/processed/yawdd')
parser.add_argument('--timeout', type=int, default=None, help='Per-video processing timeout in seconds')
parser.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 1) - 1),
                    help='Number of parallel worker processes to run')
parser.add_argument('--skip-existing', action='store_true', help='Skip processing if output .npy already exists')
args = parser.parse_args()

in_dir = Path(args.dir)
out_dir = Path(args.processed_dir)
if not in_dir.exists():
    raise FileNotFoundError(str(in_dir))


def _worker_initializer():
    # called once per process in the pool to warm-up heavy objects
    init_global_extractor()


def main():
    # collect video files first
    video_exts = ('.mp4', '.avi', '.mov', '.mkv')
    vids = [p for p in sorted(in_dir.rglob('*')) if p.is_file() and p.suffix.lower() in video_exts]
    if vids:
        out_dir.mkdir(parents=True, exist_ok=True)
        futures = []
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_worker_initializer) as ex:
            for vid in vids:
                rel = vid.relative_to(in_dir).with_suffix('')
                session_id = _sanitize_session_id(rel)
                out_npy = out_dir / f"{session_id}.npy"
                if args.skip_existing and out_npy.exists():
                    print(f'Skipping existing {out_npy}')
                    continue
                futures.append(ex.submit(run_process_video_task, str(vid), str(out_npy)))

            for fut in as_completed(futures):
                video_path, out_npy, ok, out = fut.result()
                if ok:
                    print(f'Wrote {out_npy}')
                else:
                    print(f'Failed processing {video_path}: {out}')

        manifest_path = process_extracted_folder(in_dir, out_dir)
        print('Completed processing extracted dataset into', out_dir)
    else:
        process_extracted_folder(in_dir, out_dir)
        print('Completed processing extracted dataset into', out_dir)


if __name__ == '__main__':
    main()
