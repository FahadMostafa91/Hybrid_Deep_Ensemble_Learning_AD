# hybrid_ensemble_ad_mci.py
# TensorFlow/Keras + scikit-learn reference implementation

import os, math, random, pathlib, itertools
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers as L, models as M, optimizers as O, callbacks as C
from tensorflow.keras.applications import ResNet50, NASNetMobile, MobileNetV2
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_pp
from tensorflow.keras.applications.nasnet import preprocess_input as nasnet_pp
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_pp
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.utils import shuffle as skshuffle

# ------------------------------------------------------------
# 0) Config
# ------------------------------------------------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 24
LR = 2e-5
EPOCHS_FROZEN = 5        # quick head training
EPOCHS_FINETUNE = 15     # gentle end-to-end
DROPOUT_RATE = 0.5
RANDOM_SEED = 1337
N_FOLDS = 5              # for OOF stacking
VAL_SPLIT = 0.20         # within train for early stopping
TEST_SPLIT = 0.20
# Set your data root where GM/WM slices are organized per class (AD vs MCI or MCI vs NC)
DATA_ROOT = "/path/to/slices"  # expected structure: DATA_ROOT/{class_name}/*.png

tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

# ------------------------------------------------------------
# 1) Data pipeline (replace loader with your GM/WM slice fetch)
#    We assume binary labels: 1 = AD (positive), 0 = MCI (negative), or as you choose.
# ------------------------------------------------------------
def list_images(root):
    classes = sorted([d.name for d in pathlib.Path(root).iterdir() if d.is_dir()])
    paths, labels = [], []
    for c in classes:
        for p in pathlib.Path(root, c).glob("*.png"):
            paths.append(str(p))
            labels.append(classes.index(c))
    return np.array(paths), np.array(labels), classes

def preprocess_factory(backbone):
    if backbone == "resnet": return resnet_pp
    if backbone == "nasnet": return nasnet_pp
    if backbone == "mobilenet": return mobilenet_pp
    raise ValueError("unknown backbone")

def decode_fn(path, label, backbone):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = preprocess_factory(backbone)(img)
    return img, tf.cast(label, tf.int32)

def make_dataset(paths, labels, backbone, training, batch_size=BATCH_SIZE):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(paths), seed=RANDOM_SEED, reshuffle_each_iteration=True)
        # light augmentations consistent with paper (crop/flip/brightness/contrast)
        def aug(img, lab):
            img = tf.image.random_flip_left_right(img, seed=RANDOM_SEED)
            img = tf.image.random_brightness(img, max_delta=0.05)
            img = tf.image.random_contrast(img, lower=0.95, upper=1.05)
            return img, lab
        ds = ds.map(lambda p,l: decode_fn(p,l,backbone), num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.map(aug, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        ds = ds.map(lambda p,l: decode_fn(p,l,backbone), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# ------------------------------------------------------------
# 2) Base learners: TL + Fine-tuning (ResNet50, NASNetMobile, MobileNetV2)
#    Two-phase training: feature freezing then fine-tuning. (Sec. 3.2.1) :contentReference[oaicite:2]{index=2}
# ------------------------------------------------------------
def build_head(x, dropout=DROPOUT_RATE):
    x = L.GlobalAveragePooling2D()(x)
    x = L.Dropout(dropout)(x)
    x = L.Dense(256, activation="relu")(x)
    x = L.Dropout(dropout)(x)
    out = L.Dense(1, activation="sigmoid")(x)
    return out

def build_model(backbone="resnet"):
    if backbone == "resnet":
        base = ResNet50(include_top=False, weights="imagenet", input_shape=IMG_SIZE + (3,))
    elif backbone == "nasnet":
        base = NASNetMobile(include_top=False, weights="imagenet", input_shape=IMG_SIZE + (3,))
    elif backbone == "mobilenet":
        base = MobileNetV2(include_top=False, weights="imagenet", input_shape=IMG_SIZE + (3,))
    else:
        raise ValueError("unknown backbone")

    inp = L.Input(shape=IMG_SIZE + (3,))
    x = base(inp, training=False)
    out = build_head(x)
    model = M.Model(inp, out)

    # Phase 1: freeze base
    for l in base.layers: l.trainable = False
    model.compile(optimizer=O.Adam(LR), loss="binary_crossentropy", metrics=["accuracy"])
    return model, base

def finetune(model, base, lr=LR):
    # unfreeze selected top blocks for gentle adaptation (Sec. 3.2.1) :contentReference[oaicite:3]{index=3}
    n_unfreeze = max(20, len(base.layers)//5)  # top ~20% or at least 20 layers
    for l in base.layers[:-n_unfreeze]: l.trainable = False
    for l in base.layers[-n_unfreeze:]: l.trainable = True
    model.compile(optimizer=O.Adam(lr), loss="binary_crossentropy", metrics=["accuracy"])
    return model

def train_one_backbone(backbone_name, x_tr, y_tr, x_va, y_va):
    ds_tr = make_dataset(x_tr, y_tr, backbone_name, training=True)
    ds_va = make_dataset(x_va, y_va, backbone_name, training=False)

    model, base = build_model(backbone_name)
    es = C.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    rlrop = C.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6)

    # Phase 1
    model.fit(ds_tr, validation_data=ds_va, epochs=EPOCHS_FROZEN, callbacks=[es, rlrop], verbose=2)
    # Phase 2
    model = finetune(model, base, LR)
    es2 = C.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)
    model.fit(ds_tr, validation_data=ds_va, epochs=EPOCHS_FINETUNE, callbacks=[es2, rlrop], verbose=2)
    return model

# ------------------------------------------------------------
# 3) Weighted averaging: learn nonnegative weights on simplex (Sec. 3.2.2) :contentReference[oaicite:4]{index=4}
#    We optimize α = softmax(β) to minimize BCE on validation predictions.
# ------------------------------------------------------------
class SimplexWeightLearner(tf.keras.Model):
    def __init__(self, K):
        super().__init__()
        # β initialized equally
        self.logits = tf.Variable(tf.zeros([K], dtype=tf.float32))

    @property
    def weights_simplex(self):
        return tf.nn.softmax(self.logits)

    def call(self, P):  # P: (N, K) base probabilities
        w = self.weights_simplex  # (K,)
        return tf.clip_by_value(tf.tensordot(P, w, axes=1), 1e-6, 1-1e-6)

def learn_alpha(P_val, y_val, steps=1000, lr=0.05):
    P = tf.constant(P_val.astype("float32"))
    y = tf.constant(y_val.astype("float32"))
    learner = SimplexWeightLearner(P.shape[1])
    opt = tf.keras.optimizers.Adam(lr)
    for _ in range(steps):
        with tf.GradientTape() as tape:
            p_hat = learner(P)
            loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y, p_hat))
        grads = tape.gradient(loss, learner.trainable_variables)
        opt.apply_gradients(zip(grads, learner.trainable_variables))
    return learner.weights_simplex.numpy()

# ------------------------------------------------------------
# 4) Stacking (Sec. 3.2.3): OOF meta-features -> Logistic Regression meta-learner :contentReference[oaicite:5]{index=5}
# ------------------------------------------------------------
def oof_meta_features(models, paths, labels):
    """Generate K base-prob OOF predictions."""
    K = len(models)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    P_oof = np.zeros((len(paths), K), dtype=np.float32)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(paths, labels), 1):
        for k, (name, mdl) in enumerate(models.items()):
            ds_va = make_dataset(paths[va_idx], labels[va_idx], name, training=False)
            P = []
            for xb, _ in ds_va:
                P.append(mdl.predict(xb, verbose=0))
            P = np.vstack(P).squeeze()
            P_oof[va_idx, k] = P
        print(f"OOF fold {fold} done.")
    return P_oof

# ------------------------------------------------------------
# 5) Grad-CAM (Sec. 3.4; formula on page 10; examples on page 14) :contentReference[oaicite:6]{index=6}
# ------------------------------------------------------------
def grad_cam(model, img_tensor, last_conv_layer_name=None):
    """
    img_tensor: preprocessed (1, H, W, 3) tensor for the given backbone
    """
    # auto-detect last conv if not given
    if last_conv_layer_name is None:
        last_conv_layer = None
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer = layer
                break
        if last_conv_layer is None:
            raise ValueError("No Conv2D layer found.")
        last_conv_layer_name = last_conv_layer.name

    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_tensor)
        class_channel = preds[:, 0]  # positive class score (sigmoid)

    grads = tape.gradient(class_channel, conv_out)      # ∂y/∂A
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # α_c_k
    conv_out = conv_out[0]
    heatmap = tf.reduce_sum(tf.multiply(conv_out, pooled_grads), axis=-1)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-9)

    heatmap = tf.image.resize(heatmap[..., None], IMG_SIZE)
    heatmap = heatmap.numpy().squeeze()
    return heatmap  # [0,1], upsampled to IMG_SIZE

# ------------------------------------------------------------
# 6) Train + Evaluate end-to-end (60/20/20 split) (Sec. 3.5) :contentReference[oaicite:7]{index=7}
# ------------------------------------------------------------
def main():
    paths, labels, classes = list_images(DATA_ROOT)
    paths, labels = skshuffle(paths, labels, random_state=RANDOM_SEED)

    X_temp, X_test, y_temp, y_test = train_test_split(
        paths, labels, test_size=TEST_SPLIT, random_state=RANDOM_SEED, stratify=labels
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=VAL_SPLIT, random_state=RANDOM_SEED, stratify=y_temp
    )

    # Train each backbone
    models = {}
    for name in ["resnet", "nasnet", "mobilenet"]:
        print(f"\n=== Training {name} ===")
        mdl = train_one_backbone(name, X_train, y_train, X_val, y_val)
        models[name] = mdl

    # Validation predictions for alpha learning (weighted averaging)
    P_val = []
    for name, mdl in models.items():
        ds_val = make_dataset(X_val, y_val, name, training=False)
        preds = []
        for xb, _ in ds_val:
            preds.append(mdl.predict(xb, verbose=0))
        P_val.append(np.vstack(preds).squeeze())
    P_val = np.stack(P_val, axis=1)  # (N_val, K)
    alpha = learn_alpha(P_val, y_val, steps=400, lr=0.05)
    print("Learned simplex weights α (resnet, nasnet, mobilenet):", alpha)

    # Test predictions (weighted averaging)
    P_test = []
    for name, mdl in models.items():
        ds_test = make_dataset(X_test, y_test, name, training=False)
        preds = []
        for xb, _ in ds_test:
            preds.append(mdl.predict(xb, verbose=0))
        P_test.append(np.vstack(preds).squeeze())
    P_test = np.stack(P_test, axis=1)  # (N_test, K)

    p_hat_weighted = np.clip(P_test @ alpha, 1e-6, 1-1e-6)
    y_pred_weighted = (p_hat_weighted > 0.5).astype(int)
    print("[Weighted Avg] ACC=%.3f AUC=%.3f" % (
        accuracy_score(y_test, y_pred_weighted),
        roc_auc_score(y_test, p_hat_weighted)
    ))

    # Stacking: OOF meta-features -> Logistic Regression
    print("\n=== Stacking via OOF + Logistic Regression ===")
    # Train OOF on the entire (train+val) set to avoid leakage
    X_all = np.concatenate([X_train, X_val], axis=0)
    y_all = np.concatenate([y_train, y_val], axis=0)
    P_oof = oof_meta_features(models, X_all, y_all)

    meta = LogisticRegression(max_iter=1000)
    meta.fit(P_oof, y_all)

    # Meta predictions on test
    P_test_meta = P_test  # base probs as meta features
    p_hat_meta = meta.predict_proba(P_test_meta)[:, 1]
    y_pred_meta = (p_hat_meta > 0.5).astype(int)
    print("[Stacking] ACC=%.3f AUC=%.3f" % (
        accuracy_score(y_test, y_pred_meta),
        roc_auc_score(y_test, p_hat_meta)
    ))

    # Optional: blend weighted + stacking (simple average)
    p_hat_hybrid = 0.5 * p_hat_weighted + 0.5 * p_hat_meta
    y_pred_hybrid = (p_hat_hybrid > 0.5).astype(int)
    print("[Hybrid (Weighted+Stacking)] ACC=%.3f AUC=%.3f" % (
        accuracy_score(y_test, y_pred_hybrid),
        roc_auc_score(y_test, p_hat_hybrid)
    ))

    # Example Grad-CAM on one test image with ResNet
    example_path = X_test[0]
    example_label = y_test[0]
    # Prepare one image for the chosen backbone
    raw = tf.io.decode_png(tf.io.read_file(example_path), channels=3)
    raw = tf.image.resize(raw, IMG_SIZE)
    img_resnet = resnet_pp(tf.cast(raw, tf.float32))[None, ...]
    heatmap = grad_cam(models["resnet"], img_resnet)  # [H,W] in [0,1]
    np.save("gradcam_example.npy", heatmap)  # save heatmap array for further overlay/plotting
    print("Saved Grad-CAM heatmap to gradcam_example.npy")

if __name__ == "__main__":
    main()
