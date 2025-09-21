#!/usr/bin/env python3
import os, shutil, json, random, pathlib
from collections import defaultdict

random.seed(0)

SRC_TRAIN = pathlib.Path("../data/training")
SRC_TEST = pathlib.Path("../data/testing")

DST_ROOT = pathlib.Path("../data/images")
DST_TRAIN = DST_ROOT / "train"
DST_VAL = DST_ROOT / "val"
DST_TEST = DST_ROOT / "test"
SPLITS = pathlib.Path("../data/splits")
for p in [DST_TRAIN, DST_VAL, DST_TEST, SPLITS]:
    p.mkdir(parents=True, exist_ok=True)

mapping = {
    "Apple Braeburn 1": "Braeburn",
    "Apple Crimson Snow 1": "Crimson Snow",
    "Apple Granny Smith 1": "Granny Smith",
    "Apple Pink Lady 1": "Pink Lady",
    "Apple Golden 1": "Golden Delicious",
    "Apple Golden 2": "Golden Delicious",
    "Apple Golden 3": "Golden Delicious",
    "Apple Red 1": "Red Delicious",
    "Apple Red 2": "Red Delicious",
    "Apple Red 3": "Red Delicious",
    "Apple Red Delicious 1": "Red Delicious",
    "Apple Red Yellow 1": "Red Yellow",
    "Apple Red Yellow 2": "Red Yellow",
}

def copy_class_split(src_dir: pathlib.Path, dst_dir: pathlib.Path, mapper: dict):
    counts = defaultdict(int)
    for clsdir in sorted([d for d in src_dir.iterdir() if d.is_dir()]):
        name = clsdir.name
        if not name.startswith("Apple"):
            continue
        if name not in mapper:
            print(f"[warn] skipping unmapped class: {name}")
            continue
        friendly = mapper[name]
        out = dst_dir / friendly
        out.mkdir(parents=True, exist_ok=True)
        for img in clsdir.iterdir():
            if img.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            shutil.copy2(img, out / img.name)
            counts[friendly] += 1
    return counts

print("Copying TRAIN...")
train_counts = copy_class_split(SRC_TRAIN, DST_TRAIN, mapping)
print("Copying TEST...")
test_counts  = copy_class_split(SRC_TEST,  DST_TEST,  mapping)

# 15% validation split from training set
VAL_FRAC = 0.15
for cls in sorted(train_counts):
    src = DST_TRAIN / cls
    imgs = [p for p in src.iterdir() if p.is_file()]
    random.shuffle(imgs)
    k = max(1, int(len(imgs) * VAL_FRAC))
    val_take = imgs[:k]
    (DST_VAL / cls).mkdir(parents=True, exist_ok=True)
    for p in val_take:
        shutil.move(p, DST_VAL / cls / p.name)

def write_split(root: pathlib.Path, out_path: pathlib.Path):
    rows = []
    for clsdir in sorted([d for d in root.iterdir() if d.is_dir()]):
        for img in sorted(clsdir.iterdir()):
            if img.suffix.lower() in {".jpg",".jpeg",".png"}:
                rows.append(f"{img}\t{clsdir.name}\n")
    out_path.write_text("".join(rows), encoding="utf-8")

SPLITS.mkdir(exist_ok=True, parents=True)
write_split(DST_TRAIN, SPLITS / "train.txt")
write_split(DST_VAL,   SPLITS / "val.txt")
write_split(DST_TEST,  SPLITS / "test.txt")

classes = sorted({v for v in mapping.values()})
with open("data/classes.json","w") as f:
    json.dump({"classes": classes, "mapping": mapping}, f, indent=2)

print("\nDone.")
print("Classes:", classes)
print("Train counts:", dict(train_counts))
print("Test counts:", dict(test_counts))
print("Wrote: data/splits/train.txt, val.txt, test.txt and data/classes.json")