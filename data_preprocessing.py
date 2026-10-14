import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# ---- CONFIG ----
DATASET_DIR = r"C:\Users\thrus\OneDrive\Desktop\Dataset"
IMG_SIZE = 224
CLASSES = {
    "normal": 0,
    os.path.join("DFU", "Original Images"): 1,
    "gangrene": 2
}

def load_data():
    images = []
    labels = []

    for class_name, label in CLASSES.items():
        class_folder = os.path.join(DATASET_DIR, class_name)
        if not os.path.exists(class_folder):
            print(f"WARNING: folder not found -> {class_folder}")
            continue

        files = os.listdir(class_folder)
        print(f"Loading {len(files)} images from '{class_name}'...")

        for filename in files:
            filepath = os.path.join(class_folder, filename)
            img = cv2.imread(filepath)

            if img is None:
                print(f"  Skipping unreadable file: {filename}")
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            images.append(img)
            labels.append(label)

    images = np.array(images, dtype="float32")  # raw 0-255, preprocess_input applied at training time
    labels = np.array(labels)

    return images, labels


if __name__ == "__main__":
    X, y = load_data()
    print(f"\nTotal images loaded: {len(X)}")
    print(f"Image shape: {X.shape}")
    print(f"Label distribution: {np.bincount(y)}")

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    print(f"\nTrain: {X_train.shape[0]} images")
    print(f"Val:   {X_val.shape[0]} images")
    print(f"Test:  {X_test.shape[0]} images")

    os.makedirs("data", exist_ok=True)
    np.save("data/X_train.npy", X_train)
    np.save("data/y_train.npy", y_train)
    np.save("data/X_val.npy", X_val)
    np.save("data/y_val.npy", y_val)
    np.save("data/X_test.npy", X_test)
    np.save("data/y_test.npy", y_test)

    print("\nPreprocessing complete. Saved to data/ folder.")