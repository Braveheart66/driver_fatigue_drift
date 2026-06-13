import numpy as np
import glob, os

def check(path):
    files = glob.glob(os.path.join(path, "*.npy"))
    print(f"Checking {path}: {len(files)} files")
    for f in files[:5]:
        a = np.load(f)
        has_nan = bool(np.isnan(a).any())
        has_inf = bool(np.isinf(a).any())
        amin = float(np.nanmin(a)) if a.size else None
        amax = float(np.nanmax(a)) if a.size else None
        print(f"{os.path.basename(f)}: shape={a.shape}, nan={has_nan}, inf={has_inf}, dtype={a.dtype}, min={amin}, max={amax}")

for p in ["data/processed/yawdd", "data/processed/uta-rldd"]:
    if os.path.exists(p):
        check(p)
    else:
        print(f"Path missing: {p}")
