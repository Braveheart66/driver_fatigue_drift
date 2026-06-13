"""Run a small subset ingestion using the worker pool to measure throughput.
Usage:
  python scripts/run_subset_ingest.py --src data/raw/uta-rldd --processed data/processed/uta-rldd --n 2 --workers 2
"""
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from scripts.process_video import run_process_video_task, init_global_extractor

parser = argparse.ArgumentParser()
parser.add_argument('--src', required=True)
parser.add_argument('--processed', required=True)
parser.add_argument('--n', type=int, default=2)
parser.add_argument('--workers', type=int, default=2)
parser.add_argument('--timeout', type=int, default=600)
args = parser.parse_args()

src = Path(args.src)
processed = Path(args.processed)
processed.mkdir(parents=True, exist_ok=True)

video_exts = ('.mp4', '.avi', '.mov', '.mkv')
def sanitize(p: Path, root: Path):
    rel = p.relative_to(root).with_suffix('')
    s = "_".join(rel.parts).replace(' ', '_').replace('.', '_').replace('/', '_').replace('\\', '_')
    return s
def _initializer():
    init_global_extractor()


def main():
    vids = [p for p in sorted(src.rglob('*')) if p.is_file() and p.suffix.lower() in video_exts]
    if not vids:
        print('No videos found in', src)
        raise SystemExit(1)
    subset = vids[:args.n]
    print('Processing', len(subset), 'videos')

    # simple sanitizer
    def sanitize(p: Path, root: Path):
        rel = p.relative_to(root).with_suffix('')
        s = "_".join(rel.parts).replace(' ', '_').replace('.', '_').replace('/', '_').replace('\\', '_')
        return s

    with ProcessPoolExecutor(max_workers=args.workers, initializer=_initializer) as ex:
        futures = []
        for v in subset:
            sid = sanitize(v, src)
            out = processed / f"{sid}.npy"
            futures.append(ex.submit(run_process_video_task, str(v), str(out)))

        for fut in as_completed(futures):
            video_path, out_npy, ok, out = fut.result()
            if ok:
                print('Wrote', out_npy)
            else:
                print('Failed', video_path)
                print(out)

    print('Subset ingest complete')


if __name__ == '__main__':
    main()
