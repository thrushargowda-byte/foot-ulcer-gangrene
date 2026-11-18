import os
import hashlib
import cv2
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE, "dataset")
IMG_SIZE = 224
CLASSES = {"normal": 0, os.path.join("DFU", "Original Images"): 1, "gangrene": 2}


def load_data():
    images, labels, groups, ids = [], [], [], {}
    for class_name, label in CLASSES.items():
        folder = os.path.join(DATASET_DIR, class_name)
        if not os.path.exists(folder):
            print("WARNING: folder not found ->", folder)
            continue
        files = sorted(os.listdir(folder))
        print(f"Loading {len(files)} files from '{class_name}'...")
        for name in files:
            path = os.path.join(folder, name)
            img = cv2.imread(path)
            if img is None:
                print("  Skipping unreadable file:", name)
                continue
            img = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (IMG_SIZE, IMG_SIZE))
            images.append(img)
            labels.append(label)
            with open(path, "rb") as f:
                groups.append(ids.setdefault(hashlib.md5(f.read()).hexdigest(), len(ids)))
    return np.array(images, dtype="float32"), np.array(labels), np.array(groups)


def grouped_split(y, groups, seed=42):
    """70/15/15 per class; identical images always land in the same split."""
    rng = np.random.RandomState(seed)
    split = np.zeros(len(y), dtype=int)  # 0 train, 1 val, 2 test
    for c in np.unique(y):
        members = np.where(y == c)[0]
        gs = np.unique(groups[members])
        rng.shuffle(gs)
        n = round(0.15 * len(members))
        got = [0, 0]
        for g in gs:
            idx = members[groups[members] == g]
            if got[0] < n:
                split[idx] = 2; got[0] += len(idx)
            elif got[1] < n:
                split[idx] = 1; got[1] += len(idx)
    return split


if __name__ == "__main__":
    X, y, g = load_data()
    print(f"\nTotal images loaded: {len(X)}")
    print("Label distribution:", np.bincount(y))
    split = grouped_split(y, g)
    os.makedirs(os.path.join(BASE, "data"), exist_ok=True)
    for name, k in (("train", 0), ("val", 1), ("test", 2)):
        np.save(os.path.join(BASE, "data", f"X_{name}.npy"), X[split == k])
        np.save(os.path.join(BASE, "data", f"y_{name}.npy"), y[split == k])
        print(f"{name}: {(split == k).sum()} images, per class {np.bincount(y[split == k], minlength=3)}")
    print("Duplicate groups shared between train and test:", len(set(g[split == 0]) & set(g[split == 2])))
    print("\nPreprocessing complete. Saved to data/ folder.")