"""
Retrain (or just test) the Foot Ulcer / Gangrene model so it works with app.py.

  python train_model.py --data C:/path/to/dataset --eval-only    # test your CURRENT model first
  python train_model.py --data C:/path/to/dataset                 # retrain and save model/final_model.keras

Dataset: any folder tree whose folders say which class the images are. The class is found from the
folder names on the image's path:  "gangrene" -> Gangrene,  "ulcer" / "dfu" -> Foot Ulcer (DFU),
"normal" / "healthy" / "control" -> Normal.  Pre-split layouts (train/valid/test) are fine: every
image is collected and re-split here (70% train, 15% validation, 15% test).

The output order is FIXED to what app.py expects:  0 = Normal, 1 = Foot Ulcer (DFU), 2 = Gangrene.
Images are prepared exactly like app.py does (resize 224x224, ResNet50 preprocess_input).
"""
import argparse
import os
import random
import shutil
import sys

CLASS_NAMES = ["Normal", "Foot Ulcer (DFU)", "Gangrene"]   # same order as app.py
IMG_SIZE = 224
EXTS = (".jpg", ".jpeg", ".png")


def class_of(path):
    """Work out the class from the folder names on the path (deepest folder first)."""
    parts = [p.lower() for p in os.path.normpath(path).split(os.sep)[:-1]][::-1]
    for p in parts:
        if "gangrene" in p or "gangrine" in p:
            return 2
        if "ulcer" in p or "dfu" in p:
            return 1
        if "normal" in p or "healthy" in p or "control" in p:
            return 0
    return None


def collect(data_dir):
    items, skipped = [], 0
    for root, _, files in os.walk(data_dir):
        for f in files:
            if f.lower().endswith(EXTS):
                p = os.path.join(root, f)
                c = class_of(p)
                if c is None:
                    skipped += 1
                else:
                    items.append((p, c))
    return items, skipped


def split(items, seed=42, val=0.15, test=0.15):
    """Stratified split: every class is split on its own so Gangrene is in all three sets."""
    rnd = random.Random(seed)
    tr, va, te = [], [], []
    for c in range(3):
        grp = [it for it in items if it[1] == c]
        rnd.shuffle(grp)
        n_te, n_va = int(len(grp) * test), int(len(grp) * val)
        te += grp[:n_te]; va += grp[n_te:n_te + n_va]; tr += grp[n_te + n_va:]
    for s in (tr, va, te):
        rnd.shuffle(s)
    return tr, va, te


def counts(items):
    return [sum(1 for _, c in items if c == k) for k in range(3)]


def make_ds(items, train, batch=16):
    import tensorflow as tf
    from tensorflow.keras.applications.resnet50 import preprocess_input
    paths = tf.constant([p for p, _ in items])
    labels = tf.constant([c for _, c in items], dtype=tf.int32)
    rot = tf.keras.layers.RandomRotation(0.08)
    zoom = tf.keras.layers.RandomZoom(0.15)

    def load(path, label):
        img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
        img.set_shape((IMG_SIZE, IMG_SIZE, 3))
        return img, label

    def augment(img, label):
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        img = tf.image.random_brightness(img, 35.0)
        img = tf.image.random_contrast(img, 0.75, 1.25)
        img = tf.image.random_saturation(img, 0.75, 1.25)
        img = zoom(rot(tf.expand_dims(img, 0), training=True), training=True)[0]
        return tf.clip_by_value(img, 0.0, 255.0), label

    def prep(img, label):
        return preprocess_input(img), tf.one_hot(label, 3)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if train:
        ds = ds.shuffle(len(items), seed=42)
    ds = ds.map(load, num_parallel_calls=tf.data.AUTOTUNE)
    if train:
        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.map(prep, num_parallel_calls=tf.data.AUTOTUNE).batch(batch).prefetch(tf.data.AUTOTUNE)


def report(model, items, title):
    import numpy as np
    ds = make_ds(items, train=False)
    probs = model.predict(ds, verbose=0)
    pred = probs.argmax(1)
    true = np.array([c for _, c in items])
    cm = np.zeros((3, 3), int)
    for t, p in zip(true, pred):
        cm[t, p] += 1
    print("\n=== %s (%d images) ===" % (title, len(items)))
    print("Overall accuracy: %.1f%%" % (100.0 * (pred == true).mean()))
    print("\nConfusion matrix (rows = real class, columns = what the model said):")
    print("%-18s" % "" + "".join("%-18s" % n for n in CLASS_NAMES))
    for i, n in enumerate(CLASS_NAMES):
        print("%-18s" % n + "".join("%-18d" % v for v in cm[i]))
    print("\nPer class:")
    for i, n in enumerate(CLASS_NAMES):
        rec = cm[i, i] / max(cm[i].sum(), 1)
        prec = cm[i, i] / max(cm[:, i].sum(), 1)
        print("  %-18s recall %5.1f%%   precision %5.1f%%" % (n, 100 * rec, 100 * prec))
    print("(recall = how many real cases of that class the model finds)")
    return cm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dataset folder")
    ap.add_argument("--eval-only", action="store_true", help="only test the current model, do not train")
    ap.add_argument("--model", default=os.path.join("model", "final_model.keras"))
    ap.add_argument("--epochs-head", type=int, default=8)
    ap.add_argument("--epochs-fine", type=int, default=25)
    ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args()

    items, skipped = collect(a.data)
    if not items:
        sys.exit("No images found. Check the --data folder and that class folders are named like "
                 "'gangrene', 'ulcer'/'dfu' and 'normal'.")
    tr, va, te = split(items)
    print("Images found per class (Normal / Foot Ulcer / Gangrene):", counts(items))
    print("Train %s | Validation %s | Test %s" % (counts(tr), counts(va), counts(te)))
    if skipped:
        print("Skipped %d images whose folder name did not say which class they are." % skipped)
    if min(counts(items)) == 0:
        sys.exit("One class has no images at all, so it cannot be trained.")
    if counts(items)[2] < 0.6 * max(counts(items)):
        print("WARNING: Gangrene has far fewer images than the other classes. Class weights will help, "
              "but adding more gangrene photos is the best fix.")

    import numpy as np
    import tensorflow as tf
    tf.keras.utils.set_random_seed(42)

    if a.eval_only:
        model = tf.keras.models.load_model(a.model)
        report(model, te, "CURRENT model on the TEST split")
        print("\nIf Gangrene recall is low, retrain with:  python train_model.py --data \"%s\"" % a.data)
        return

    from tensorflow.keras.applications import ResNet50
    base = ResNet50(include_top=False, weights="imagenet", input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base.trainable = False
    x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    x = tf.keras.layers.Dropout(0.4)(x)
    out = tf.keras.layers.Dense(3, activation="softmax")(x)
    model = tf.keras.Model(base.input, out)

    n = counts(tr)
    weights = {i: len(tr) / (3.0 * n[i]) for i in range(3)}     # rarer class = bigger weight
    print("Class weights:", {CLASS_NAMES[i]: round(w, 2) for i, w in weights.items()})

    train_ds, val_ds = make_ds(tr, True, a.batch), make_ds(va, False, a.batch)
    cbs = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
           tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.4, patience=3, min_lr=1e-7)]

    print("\n--- Phase 1: training the new top layer ---")
    model.compile(tf.keras.optimizers.Adam(1e-3), "categorical_crossentropy", metrics=["accuracy"])
    model.fit(train_ds, validation_data=val_ds, epochs=a.epochs_head, class_weight=weights, callbacks=cbs)

    print("\n--- Phase 2: fine-tuning the last ResNet layers ---")
    base.trainable = True
    for layer in base.layers[:-40]:
        layer.trainable = False
    for layer in base.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False                              # keeps small-dataset training stable
    model.compile(tf.keras.optimizers.Adam(1e-5), "categorical_crossentropy", metrics=["accuracy"])
    model.fit(train_ds, validation_data=val_ds, epochs=a.epochs_fine, class_weight=weights, callbacks=cbs)

    cm = report(model, te, "NEW model on the TEST split (never used for training)")

    os.makedirs(os.path.dirname(a.model) or ".", exist_ok=True)
    if os.path.isfile(a.model):
        shutil.copy2(a.model, a.model.replace(".keras", "_old.keras"))
        print("\nYour previous model was saved as", a.model.replace(".keras", "_old.keras"))
    model.save(a.model)
    print("New model saved to", a.model)
    print("Restart the app (python app.py) to use it.")


if __name__ == "__main__":
    main()
