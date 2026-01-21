# scripts/split_spects.py
import os, random, shutil, pathlib
random.seed(42)

ROOT = "dataset/spectrograms/train"
OUT  = "dataset/spectrograms"
VAL_PCT, TEST_PCT = 0.15, 0.15

for cls in ["real","fake"]:
    files = [f for f in pathlib.Path(ROOT, cls).glob("*.png")]
    random.shuffle(files)
    n = len(files); n_val = int(n*VAL_PCT); n_test = int(n*TEST_PCT)
    val = files[:n_val]; test = files[n_val:n_val+n_test]; train = files[n_val+n_test:]

    for split, lst in [("val",val),("test",test)]:
        os.makedirs(f"{OUT}/{split}/{cls}", exist_ok=True)
        for p in lst:
            shutil.move(str(p), f"{OUT}/{split}/{cls}/{p.name}")
print("Done.")
