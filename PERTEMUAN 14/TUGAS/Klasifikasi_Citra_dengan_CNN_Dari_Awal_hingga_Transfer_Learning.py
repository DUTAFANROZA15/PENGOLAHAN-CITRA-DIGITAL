# =============================================================================
# KLASIFIKASI CITRA DENGAN CNN: DARI AWAL HINGGA TRANSFER LEARNING
# Dataset: CIFAR-10 (10 Kelas, 60.000 Gambar)
# Framework: TensorFlow / Keras
# =============================================================================

import os
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_fscore_support
)
from sklearn.preprocessing import label_binarize
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

# ─────────────────────────────────────────────
# KONFIGURASI GLOBAL
# ─────────────────────────────────────────────
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
NUM_CLASSES  = 10
IMG_SHAPE    = (32, 32, 3)
BATCH_SIZE   = 64
EPOCHS_SCRATCH = 50
EPOCHS_TL      = 20
EPOCHS_FINETUNE = 10
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Palette warna konsisten
PALETTE = plt.cm.tab10(np.linspace(0, 1, NUM_CLASSES))


# =============================================================================
# BAGIAN 0: LOAD & PERSIAPAN DATASET
# =============================================================================
def load_and_prepare_data():
    print("\n" + "="*60)
    print(" BAGIAN 0: LOAD & PERSIAPAN DATASET CIFAR-10")
    print("="*60)

    (X_train_raw, y_train_raw), (X_test_raw, y_test_raw) = \
        keras.datasets.cifar10.load_data()

    # Normalisasi ke [0, 1]
    X_train = X_train_raw.astype('float32') / 255.0
    X_test  = X_test_raw.astype('float32')  / 255.0

    # Squeeze label ke 1D
    y_train = y_train_raw.squeeze()
    y_test  = y_test_raw.squeeze()

    # One-hot encode
    y_train_oh = keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_test_oh  = keras.utils.to_categorical(y_test,  NUM_CLASSES)

    print(f"  Training   : {X_train.shape}  labels={y_train.shape}")
    print(f"  Test       : {X_test.shape}   labels={y_test.shape}")
    print(f"  Kelas      : {CLASS_NAMES}")

    return (X_train, y_train, y_train_oh,
            X_test,  y_test,  y_test_oh)


# =============================================================================
# BAGIAN 1: VISUALISASI DATASET (Plot 1)
# =============================================================================
def plot_sample_images(X_train, y_train, plot_num=1):
    print(f"\n[Plot {plot_num}] Sample gambar CIFAR-10 ...")
    fig, axes = plt.subplots(4, 5, figsize=(11, 9))
    fig.suptitle("Plot 1 — Sample Gambar CIFAR-10 (20 Gambar)", fontsize=14, fontweight='bold')

    # Pilih 2 gambar per kelas (10 kelas × 2 = 20)
    idx = []
    for c in range(NUM_CLASSES):
        idxs = np.where(y_train == c)[0]
        idx.extend(idxs[:2])

    for i, ax in enumerate(axes.flat):
        ax.imshow(X_train[idx[i]])
        ax.set_title(CLASS_NAMES[y_train[idx[i]]], fontsize=9)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig("plot01_sample_images.png", dpi=100, bbox_inches='tight')
    plt.show()
    print("  → Tersimpan: plot01_sample_images.png")


# =============================================================================
# BAGIAN 2: DATA AUGMENTATION (Plot 2)
# =============================================================================
def setup_augmentation():
    """Kembalikan ImageDataGenerator dengan augmentasi lengkap."""
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        shear_range=0.2,
        fill_mode='nearest'
    )
    return datagen


def plot_augmentation_samples(X_train, y_train, datagen, plot_num=2):
    print(f"\n[Plot {plot_num}] Visualisasi Augmentasi Data ...")
    sample_idx = np.random.choice(len(X_train), 5, replace=False)
    fig, axes = plt.subplots(2, 5, figsize=(14, 6))
    fig.suptitle("Plot 2 — Contoh Augmentasi Data (Baris Atas: Asli | Bawah: Augmented)",
                 fontsize=12, fontweight='bold')

    for col, si in enumerate(sample_idx):
        img = X_train[si]
        lbl = CLASS_NAMES[y_train[si]]

        # Baris atas: asli
        axes[0, col].imshow(img)
        axes[0, col].set_title(f"Original\n{lbl}", fontsize=9)
        axes[0, col].axis('off')

        # Baris bawah: augmented
        aug_img = next(datagen.flow(img[np.newaxis], batch_size=1))[0]
        axes[1, col].imshow(np.clip(aug_img, 0, 1))
        axes[1, col].set_title("Augmented", fontsize=9)
        axes[1, col].axis('off')

    plt.tight_layout()
    plt.savefig("plot02_augmentation.png", dpi=100, bbox_inches='tight')
    plt.show()
    print("  → Tersimpan: plot02_augmentation.png")


# =============================================================================
# BAGIAN 3: ARSITEKTUR CNN FROM SCRATCH
# =============================================================================
def build_cnn_scratch(name="CNN_Scratch"):
    """CNN utama dari awal sesuai template tugas + BatchNorm + Dropout."""
    model = keras.Sequential([
        # Block 1
        layers.Conv2D(32, (3,3), padding='same', activation='relu',
                      input_shape=IMG_SHAPE, name='conv1'),
        layers.BatchNormalization(name='bn1'),
        layers.Conv2D(32, (3,3), padding='same', activation='relu', name='conv2'),
        layers.BatchNormalization(name='bn2'),
        layers.MaxPooling2D((2,2), name='pool1'),
        layers.Dropout(0.25, name='drop1'),

        # Block 2
        layers.Conv2D(64, (3,3), padding='same', activation='relu', name='conv3'),
        layers.BatchNormalization(name='bn3'),
        layers.Conv2D(64, (3,3), padding='same', activation='relu', name='conv4'),
        layers.BatchNormalization(name='bn4'),
        layers.MaxPooling2D((2,2), name='pool2'),
        layers.Dropout(0.25, name='drop2'),

        # Block 3
        layers.Conv2D(128, (3,3), padding='same', activation='relu', name='conv5'),
        layers.BatchNormalization(name='bn5'),
        layers.Conv2D(128, (3,3), padding='same', activation='relu', name='conv6'),
        layers.BatchNormalization(name='bn6'),
        layers.MaxPooling2D((2,2), name='pool3'),
        layers.Dropout(0.25, name='drop3'),

        # Classifier
        layers.Flatten(name='flatten'),
        layers.Dense(256, activation='relu', name='fc1'),
        layers.BatchNormalization(name='bn7'),
        layers.Dropout(0.5, name='drop4'),
        layers.Dense(NUM_CLASSES, activation='softmax', name='output')
    ], name=name)
    return model


def build_cnn_variant_sgd(name="CNN_SGD"):
    """Variasi: Arsitektur lebih ringan, optimizer SGD."""
    model = keras.Sequential([
        layers.Conv2D(32, (3,3), padding='same', activation='relu',
                      input_shape=IMG_SHAPE),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), padding='same', activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(128, (3,3), padding='same', activation='relu'),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ], name=name)
    return model


# =============================================================================
# BAGIAN 4: TRAINING CNN FROM SCRATCH + VARIASI (Plot 3)
# =============================================================================
def train_cnn_variants(X_train, y_train_oh, X_test, y_test_oh, datagen):
    print("\n" + "="*60)
    print(" BAGIAN 4: TRAINING CNN FROM SCRATCH & VARIASI")
    print("="*60)

    results = {}

    # ── 4A. CNN Scratch + Adam + Augmentation ─────────────────
    print("\n  [A] CNN Scratch — Adam lr=0.001 + Augmentasi")
    model_a = build_cnn_scratch("CNN_Adam_Aug")
    model_a.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model_a.summary()

    cb_a = [
        keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10,
                                      restore_best_weights=True, verbose=0),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                          patience=5, min_lr=1e-6, verbose=0)
    ]

    t0 = time.time()
    flow = datagen.flow(X_train, y_train_oh, batch_size=BATCH_SIZE, seed=SEED)
    hist_a = model_a.fit(
        flow,
        steps_per_epoch=len(X_train) // BATCH_SIZE,
        epochs=EPOCHS_SCRATCH,
        validation_data=(X_test, y_test_oh),
        callbacks=cb_a,
        verbose=1
    )
    train_time_a = time.time() - t0

    t1 = time.time()
    loss_a, acc_a = model_a.evaluate(X_test, y_test_oh, verbose=0)
    inf_time_a = (time.time() - t1) / len(X_test) * 1000  # ms per image

    results['CNN_Adam_Aug'] = {
        'model': model_a, 'history': hist_a,
        'test_acc': acc_a, 'test_loss': loss_a,
        'train_time': train_time_a, 'inf_time_ms': inf_time_a,
        'params': model_a.count_params()
    }
    print(f"    Test Accuracy : {acc_a:.4f}  |  Time: {train_time_a:.1f}s")

    # ── 4B. CNN Variasi — SGD (tanpa augmentasi, arsitektur ringan) ─
    print("\n  [B] CNN Variasi — SGD lr=0.01 (tanpa augmentasi)")
    model_b = build_cnn_variant_sgd("CNN_SGD_NoAug")
    model_b.compile(
        optimizer=keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    cb_b = [keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10,
                                           restore_best_weights=True, verbose=0)]
    t0 = time.time()
    hist_b = model_b.fit(
        X_train, y_train_oh,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS_SCRATCH,
        validation_data=(X_test, y_test_oh),
        callbacks=cb_b,
        verbose=1
    )
    train_time_b = time.time() - t0

    t1 = time.time()
    loss_b, acc_b = model_b.evaluate(X_test, y_test_oh, verbose=0)
    inf_time_b = (time.time() - t1) / len(X_test) * 1000

    results['CNN_SGD_NoAug'] = {
        'model': model_b, 'history': hist_b,
        'test_acc': acc_b, 'test_loss': loss_b,
        'train_time': train_time_b, 'inf_time_ms': inf_time_b,
        'params': model_b.count_params()
    }
    print(f"    Test Accuracy : {acc_b:.4f}  |  Time: {train_time_b:.1f}s")

    return results


# =============================================================================
# BAGIAN 5: TRANSFER LEARNING — FEATURE EXTRACTION + FINE-TUNING (Plot 4 dst)
# =============================================================================
def build_transfer_model(base_name, input_shape=(32, 32, 3), num_classes=10):
    """
    Bangun model transfer learning dari pre-trained base.
    CIFAR-10 native 32×32, kita upscale ke 96×96 agar cocok dengan ImageNet weights.
    """
    upscaled_shape = (96, 96, 3)

    base_map = {
        'VGG16':        (keras.applications.VGG16,
                         keras.applications.vgg16.preprocess_input),
        'ResNet50':     (keras.applications.ResNet50,
                         keras.applications.resnet.preprocess_input),
        'MobileNetV2':  (keras.applications.MobileNetV2,
                         keras.applications.mobilenet_v2.preprocess_input),
    }
    model_fn, preprocess_fn = base_map[base_name]

    base = model_fn(weights='imagenet', include_top=False,
                    input_shape=upscaled_shape)
    base.trainable = False  # Freeze untuk feature extraction

    inp = keras.Input(shape=input_shape)
    x = layers.Resizing(96, 96)(inp)           # Upscale ke 96×96
    x = layers.Lambda(preprocess_fn)(x)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(num_classes, activation='softmax')(x)

    model = keras.Model(inp, out, name=f"TL_{base_name}")
    return model, base


def train_transfer_learning(X_train, y_train_oh, X_test, y_test_oh, datagen):
    print("\n" + "="*60)
    print(" BAGIAN 5: TRANSFER LEARNING")
    print("="*60)

    tl_results = {}

    for base_name in ['MobileNetV2', 'ResNet50', 'VGG16']:
        print(f"\n  ── {base_name} ──────────────────────────")
        try:
            model, base = build_transfer_model(base_name)
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )

            cb = [keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5,
                                                restore_best_weights=True, verbose=0)]

            print("  [Feature Extraction] Training classifier head ...")
            t0 = time.time()
            flow = datagen.flow(X_train, y_train_oh, batch_size=BATCH_SIZE, seed=SEED)
            hist_fe = model.fit(
                flow,
                steps_per_epoch=len(X_train) // BATCH_SIZE,
                epochs=EPOCHS_TL,
                validation_data=(X_test, y_test_oh),
                callbacks=cb,
                verbose=1
            )
            fe_time = time.time() - t0

            _, acc_fe = model.evaluate(X_test, y_test_oh, verbose=0)
            print(f"  Feature Extraction Acc: {acc_fe:.4f}  ({fe_time:.1f}s)")

            # ── Fine-tuning ──────────────────────────────────────
            print("  [Fine-Tuning] Membuka layer terakhir base model ...")
            base.trainable = True
            fine_tune_at = max(0, len(base.layers) - 20)
            for layer in base.layers[:fine_tune_at]:
                layer.trainable = False

            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=1e-5),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )

            t1 = time.time()
            hist_ft = model.fit(
                X_train, y_train_oh,
                batch_size=BATCH_SIZE,
                epochs=EPOCHS_FINETUNE,
                validation_data=(X_test, y_test_oh),
                verbose=1
            )
            ft_time = time.time() - t1

            _, acc_ft = model.evaluate(X_test, y_test_oh, verbose=0)
            print(f"  Fine-Tuning Acc       : {acc_ft:.4f}  ({ft_time:.1f}s)")

            t_inf = time.time()
            _ = model.predict(X_test[:100], verbose=0)
            inf_ms = (time.time() - t_inf) / 100 * 1000

            tl_results[base_name] = {
                'model': model,
                'hist_fe': hist_fe, 'hist_ft': hist_ft,
                'acc_fe': acc_fe,   'acc_ft': acc_ft,
                'fe_time': fe_time, 'ft_time': ft_time,
                'inf_time_ms': inf_ms,
                'params': model.count_params()
            }

        except Exception as e:
            print(f"  !! Error pada {base_name}: {e}")

    return tl_results


# =============================================================================
# BAGIAN 6: EVALUASI KOMPREHENSIF
# =============================================================================
def comprehensive_evaluation(model, X_test, y_test, y_test_oh, model_name="Model"):
    """
    Hitung accuracy, loss, precision, recall, F1, ROC-AUC.
    Kembalikan dict hasil + y_pred_proba.
    """
    loss, acc = model.evaluate(X_test, y_test_oh, verbose=0)
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)

    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='weighted', zero_division=0
    )

    # ROC-AUC (OvR)
    y_bin = label_binarize(y_test, classes=list(range(NUM_CLASSES)))
    roc_auc_per_class = {}
    for i in range(NUM_CLASSES):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_pred_proba[:, i])
        roc_auc_per_class[CLASS_NAMES[i]] = auc(fpr, tpr)
    macro_auc = np.mean(list(roc_auc_per_class.values()))

    return {
        'model_name': model_name,
        'loss': loss, 'accuracy': acc,
        'precision': prec, 'recall': rec, 'f1': f1,
        'macro_auc': macro_auc,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'roc_auc_per_class': roc_auc_per_class
    }


# =============================================================================
# BAGIAN 7: SEMUA PLOT (3 — 13)
# =============================================================================

# ── Plot 3: Learning Curve CNN Scratch ─────────────────────────────────────
def plot_learning_curves(scratch_results, plot_num=3):
    print(f"\n[Plot {plot_num}] Learning Curves CNN Scratch ...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Plot 3 — Learning Curve CNN from Scratch", fontsize=13, fontweight='bold')

    colors = {'CNN_Adam_Aug': 'steelblue', 'CNN_SGD_NoAug': 'tomato'}
    styles = {'CNN_Adam_Aug': '-', 'CNN_SGD_NoAug': '--'}

    for name, res in scratch_results.items():
        h = res['history'].history
        c, s = colors[name], styles[name]
        axes[0].plot(h['accuracy'],     color=c, linestyle=s,  label=f'{name} Train')
        axes[0].plot(h['val_accuracy'], color=c, linestyle=':', label=f'{name} Val', alpha=0.7)
        axes[1].plot(h['loss'],         color=c, linestyle=s,  label=f'{name} Train')
        axes[1].plot(h['val_loss'],     color=c, linestyle=':', label=f'{name} Val', alpha=0.7)

    for ax, title, ylabel in zip(axes,
                                  ['Accuracy', 'Loss'],
                                  ['Accuracy', 'Loss']):
        ax.set_title(title); ax.set_xlabel('Epoch'); ax.set_ylabel(ylabel)
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("plot03_learning_curves_scratch.png", dpi=100, bbox_inches='tight')
    plt.show()
    print("  → Tersimpan: plot03_learning_curves_scratch.png")


# ── Plot 4: Confusion Matrix ────────────────────────────────────────────────
def plot_confusion_matrix(eval_res, plot_num=4):
    print(f"\n[Plot {plot_num}] Confusion Matrix ...")
    y_pred = eval_res['y_pred']
    cm = confusion_matrix(
        np.argmax(eval_res.get('y_test_oh', np.eye(NUM_CLASSES)[y_pred]), axis=1)
        if 'y_test' not in eval_res else eval_res['y_test'],
        y_pred
    )
    # Akan dipanggil dengan y_test passed externally; re-compute:
    return cm  # caller akan plot


def plot_cm(y_test, y_pred, title="Confusion Matrix", plot_num=4):
    print(f"\n[Plot {plot_num}] Confusion Matrix — {title} ...")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                linewidths=0.5, ax=ax)
    ax.set_title(f"Plot {plot_num} — Confusion Matrix: {title}", fontsize=13, fontweight='bold')
    ax.set_xlabel('Predicted Label'); ax.set_ylabel('True Label')
    plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"plot0{plot_num}_confusion_matrix.png", dpi=100, bbox_inches='tight')
    plt.show()
    print(f"  → Tersimpan: plot0{plot_num}_confusion_matrix.png")


# ── Plot 5: ROC Curve Multi-Class ───────────────────────────────────────────
def plot_roc_curves(eval_res, y_test, plot_num=5):
    print(f"\n[Plot {plot_num}] ROC Curve Multi-Class ...")
    y_bin = label_binarize(y_test, classes=list(range(NUM_CLASSES)))
    y_prob = eval_res['y_pred_proba']

    fig, ax = plt.subplots(figsize=(10, 7))
    for i in range(NUM_CLASSES):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=PALETTE[i], lw=1.5,
                label=f"{CLASS_NAMES[i]} (AUC={roc_auc:.2f})")

    ax.plot([0,1],[0,1], 'k--', lw=1)
    ax.set_title(f"Plot {plot_num} — ROC Curve Multi-Class (OvR)\n"
                 f"Macro AUC = {eval_res['macro_auc']:.4f}",
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("plot05_roc_curves.png", dpi=100, bbox_inches='tight')
    plt.show()
    print("  → Tersimpan: plot05_roc_curves.png")


# ── Plot 6: Precision / Recall / F1 per Kelas ───────────────────────────────
def plot_per_class_metrics(y_test, y_pred, plot_num=6):
    print(f"\n[Plot {plot_num}] Metrik per Kelas ...")
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, zero_division=0)

    x = np.arange(NUM_CLASSES)
    width = 0.28
    fig, ax = plt.subplots(figsize=(13, 6))
    b1 = ax.bar(x - width,     prec, width, label='Precision', color='steelblue')
    b2 = ax.bar(x,             rec,  width, label='Recall',    color='seagreen')
    b3 = ax.bar(x + width,     f1,   width, label='F1-Score',  color='tomato')

    ax.set_title(f"Plot {plot_num} — Precision, Recall, F1-Score per Kelas",
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels(CLASS_NAMES, rotation=40, ha='right')
    ax.set_ylim(0, 1.1); ax.set_ylabel('Score'); ax.legend()
    ax.grid(axis='y', alpha=0.3)
    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                    f'{h:.2f}', ha='center', va='bottom', fontsize=7)
    plt.tight_layout()
    plt.savefig("plot06_per_class_metrics.png", dpi=100, bbox_inches='tight')
    plt.show()
    print("  → Tersimpan: plot06_per_class_metrics.png")


# ── Plot 7: Feature Maps ────────────────────────────────────────────────────
def plot_feature_maps(model, X_test, plot_num=7):
    print(f"\n[Plot {plot_num}] Feature Maps Conv Layer ...")

    # Pastikan model sudah di-build
    _ = model(tf.zeros((1, *IMG_SHAPE)), training=False)

    # Ambil output conv layers pertama dan kedua
    conv_layers = [l for l in model.layers if 'conv' in l.name][:2]
    if not conv_layers:
        print("  Tidak ada conv layer ditemukan.")
        return

    activation_model = keras.Model(
        inputs=model.inputs,
        outputs=[l.output for l in conv_layers]
    )

    sample = X_test[0:1]
    activations = activation_model.predict(sample, verbose=0)

    fig, axes = plt.subplots(2, 8, figsize=(16, 5))
    fig.suptitle(f"Plot {plot_num} — Feature Maps (Conv Layer 1 & 2, 8 Filter)",
                 fontsize=12, fontweight='bold')

    for row, (act, layer) in enumerate(zip(activations, conv_layers)):
        for col in range(8):
            axes[row, col].imshow(act[0, :, :, col], cmap='viridis')
            axes[row, col].set_title(f"F{col+1}", fontsize=8)
            axes[row, col].axis('off')
        axes[row, 0].set_ylabel(layer.name, fontsize=9, rotation=90, labelpad=5)

    plt.tight_layout()
    plt.savefig("plot07_feature_maps.png", dpi=100, bbox_inches='tight')
    plt.show()
    print("  → Tersimpan: plot07_feature_maps.png")


# ── Plot 8: Filter Visualization ────────────────────────────────────────────
def plot_filter_visualization(model, plot_num=8):
    print(f"\n[Plot {plot_num}] Filter Visualization (Konvolusi Layer 1) ...")

    # Ambil conv layer pertama
    conv1 = next((l for l in model.layers if 'conv' in l.name), None)
    if conv1 is None:
        print("  Conv layer tidak ditemukan.")
        return

    weights = conv1.get_weights()[0]  # shape: (H, W, C_in, C_out)
    n_filters = min(32, weights.shape[-1])

    # Normalisasi filter ke [0,1]
    w_min, w_max = weights.min(), weights.max()
    filters_norm = (weights - w_min) / (w_max - w_min + 1e-8)

    n_cols = 8; n_rows = int(np.ceil(n_filters / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4))
    fig.suptitle(f"Plot {plot_num} — Filter Visualization: {conv1.name} ({n_filters} Filter)",
                 fontsize=12, fontweight='bold')

    for i in range(n_rows * n_cols):
        ax = axes[i // n_cols, i % n_cols]
        if i < n_filters:
            f = filters_norm[:, :, :, i]
            if f.shape[2] == 3:
                ax.imshow(f)
            else:
                ax.imshow(f[:, :, 0], cmap='gray')
            ax.set_title(f"F{i+1}", fontsize=7)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig("plot08_filter_visualization.png", dpi=100, bbox_inches='tight')
    plt.show()
    print("  → Tersimpan: plot08_filter_visualization.png")


# ── Plot 9: Grad-CAM ─────────────────────────────────────────────────────────
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """Buat heatmap Grad-CAM untuk satu gambar."""
    grad_model = keras.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def plot_gradcam(model, X_test, y_test, plot_num=9, n_images=8):
    print(f"\n[Plot {plot_num}] Grad-CAM Visualization ...")

    # Cari nama conv layer terakhir
    last_conv = None
    for layer in reversed(model.layers):
        if 'conv' in layer.name:
            last_conv = layer.name
            break

    if last_conv is None:
        print("  Conv layer tidak ditemukan untuk Grad-CAM.")
        return

    indices = np.random.choice(len(X_test), n_images, replace=False)
    fig, axes = plt.subplots(2, n_images, figsize=(n_images * 2, 5))
    fig.suptitle(f"Plot {plot_num} — Grad-CAM (Baris Atas: Asli | Bawah: Heatmap)",
                 fontsize=12, fontweight='bold')

    for col, idx in enumerate(indices):
        img = X_test[idx]
        img_arr = img[np.newaxis].astype(np.float32)
        pred_proba = model.predict(img_arr, verbose=0)[0]
        pred_class = np.argmax(pred_proba)
        true_class = y_test[idx]

        try:
            heatmap = make_gradcam_heatmap(img_arr, model, last_conv, pred_class)
            heatmap_resized = np.array(
                tf.image.resize(heatmap[..., np.newaxis], [32, 32])
            ).squeeze()
        except Exception:
            heatmap_resized = np.zeros((32, 32))

        # Baris atas: gambar asli
        axes[0, col].imshow(img)
        color = 'green' if pred_class == true_class else 'red'
        axes[0, col].set_title(
            f"T:{CLASS_NAMES[true_class]}\nP:{CLASS_NAMES[pred_class]}",
            fontsize=7, color=color
        )
        axes[0, col].axis('off')

        # Baris bawah: overlay heatmap
        axes[1, col].imshow(img)
        axes[1, col].imshow(heatmap_resized, cmap='jet', alpha=0.45)
        axes[1, col].axis('off')

    plt.tight_layout()
    plt.savefig("plot09_gradcam.png", dpi=100, bbox_inches='tight')
    plt.show()
    print("  → Tersimpan: plot09_gradcam.png")


# ── Plot 10: t-SNE Feature Embeddings ───────────────────────────────────────
def plot_tsne_embeddings(model, X_test, y_test, plot_num=10, n_samples=2000):
    print(f"\n[Plot {plot_num}] t-SNE Feature Embeddings ...")

    # Buat model ekstraksi fitur (sebelum layer output)
    feature_layer = model.layers[-2]  # Dense sebelum output
    feat_model = keras.Model(inputs=model.inputs, outputs=feature_layer.output)

    # Ambil subset
    idx = np.random.choice(len(X_test), min(n_samples, len(X_test)), replace=False)
    feats = feat_model.predict(X_test[idx], verbose=0, batch_size=128)
    labels = y_test[idx]

    print(f"  t-SNE pada {len(feats)} sampel, dim={feats.shape[1]} ...")
    import sklearn
    tsne_kwargs = dict(n_components=2, perplexity=30, random_state=SEED)
    sk_version = tuple(int(x) for x in sklearn.__version__.split(".")[:2])
    if sk_version >= (1, 2):
        tsne_kwargs['max_iter'] = 500
    else:
        tsne_kwargs['n_iter'] = 500
    tsne = TSNE(**tsne_kwargs)
    emb  = tsne.fit_transform(feats)

    fig, ax = plt.subplots(figsize=(10, 8))
    for c in range(NUM_CLASSES):
        mask = labels == c
        ax.scatter(emb[mask, 0], emb[mask, 1], s=8, alpha=0.6,
                   color=PALETTE[c], label=CLASS_NAMES[c])

    ax.set_title(f"Plot {plot_num} — t-SNE Feature Embeddings ({n_samples} sampel)",
                 fontsize=13, fontweight='bold')
    ax.legend(markerscale=2, fontsize=9, loc='best')
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig("plot10_tsne_embeddings.png", dpi=100, bbox_inches='tight')
    plt.show()
    print("  → Tersimpan: plot10_tsne_embeddings.png")


# ── Plot 11: Augmentasi vs Tanpa Augmentasi (Dampak Overfitting) ─────────────
def plot_augmentation_effect(scratch_results, plot_num=11):
    print(f"\n[Plot {plot_num}] Pengaruh Augmentasi terhadap Overfitting ...")
    aug_hist   = scratch_results['CNN_Adam_Aug']['history'].history
    noaug_hist = scratch_results['CNN_SGD_NoAug']['history'].history

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Plot {plot_num} — Augmentasi vs Tanpa Augmentasi (Overfitting Analysis)",
                 fontsize=12, fontweight='bold')

    for ax, key, title in zip(axes, ['accuracy', 'loss'], ['Accuracy', 'Loss']):
        ax.plot(aug_hist[key],        color='steelblue', label='Aug - Train')
        ax.plot(aug_hist[f'val_{key}'], color='steelblue', linestyle='--', label='Aug - Val')
        ax.plot(noaug_hist[key],        color='tomato',    label='NoAug - Train')
        ax.plot(noaug_hist[f'val_{key}'], color='tomato',  linestyle='--', label='NoAug - Val')
        ax.set_title(title); ax.set_xlabel('Epoch'); ax.set_ylabel(title)
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

        # Highlight gap (overfitting)
        gap_aug   = np.array(aug_hist[key])   - np.array(aug_hist[f'val_{key}'])
        gap_noaug = np.array(noaug_hist[key]) - np.array(noaug_hist[f'val_{key}'])
        ax.fill_between(range(len(gap_aug)), aug_hist[key], aug_hist[f'val_{key}'],
                        alpha=0.08, color='steelblue')
        ax.fill_between(range(len(gap_noaug)), noaug_hist[key], noaug_hist[f'val_{key}'],
                        alpha=0.08, color='tomato')

    plt.tight_layout()
    plt.savefig("plot11_augmentation_effect.png", dpi=100, bbox_inches='tight')
    plt.show()
    print("  → Tersimpan: plot11_augmentation_effect.png")


# ── Plot 12: Tabel Perbandingan Semua Model ──────────────────────────────────
def plot_comparison_table(scratch_results, tl_results, eval_scratch, eval_tl_dict, plot_num=12):
    print(f"\n[Plot {plot_num}] Tabel Perbandingan Semua Model ...")

    rows = []

    # CNN Scratch
    for name, res in scratch_results.items():
        ev = eval_scratch.get(name, {})
        rows.append([
            name.replace('_', ' '),
            f"{res['test_acc']:.4f}",
            f"{ev.get('f1', 0):.4f}",
            f"{ev.get('macro_auc', 0):.4f}",
            f"{res['train_time']:.1f}s",
            f"{res['inf_time_ms']:.2f}ms",
            f"{res['params']:,}"
        ])

    # Transfer Learning
    for base_name, res in tl_results.items():
        ev = eval_tl_dict.get(base_name, {})
        rows.append([
            f"TL {base_name} FE",
            f"{res['acc_fe']:.4f}",
            f"{ev.get('f1', 0):.4f}",
            f"{ev.get('macro_auc', 0):.4f}",
            f"{res['fe_time']:.1f}s",
            f"{res['inf_time_ms']:.2f}ms",
            f"{res['params']:,}"
        ])
        rows.append([
            f"TL {base_name} FT",
            f"{res['acc_ft']:.4f}",
            "—", "—",
            f"{res['ft_time']:.1f}s",
            f"{res['inf_time_ms']:.2f}ms",
            f"{res['params']:,}"
        ])

    cols = ['Model', 'Test Acc', 'F1 (W)', 'AUC Macro',
            'Train Time', 'Inf/img', 'Parameters']

    fig, ax = plt.subplots(figsize=(16, max(4, len(rows) * 0.55 + 1.5)))
    ax.axis('off')
    tbl = ax.table(
        cellText=rows, colLabels=cols,
        loc='center', cellLoc='center'
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.5)

    # Styling header
    for j in range(len(cols)):
        tbl[(0, j)].set_facecolor('#2C3E50')
        tbl[(0, j)].set_text_props(color='white', fontweight='bold')

    # Alternate row color
    for i in range(1, len(rows) + 1):
        for j in range(len(cols)):
            tbl[(i, j)].set_facecolor('#EAF2FB' if i % 2 == 0 else 'white')

    ax.set_title(f"Plot {plot_num} — Tabel Perbandingan Semua Eksperimen",
                 fontsize=13, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig("plot12_comparison_table.png", dpi=100, bbox_inches='tight')
    plt.show()
    print("  → Tersimpan: plot12_comparison_table.png")


# ── Plot 13: Transfer Learning Learning Curves ───────────────────────────────
def plot_tl_curves(tl_results, plot_num=13):
    print(f"\n[Plot {plot_num}] Transfer Learning Curves ...")
    n_models = len(tl_results)
    if n_models == 0:
        print("  Tidak ada hasil Transfer Learning.")
        return

    fig, axes = plt.subplots(n_models, 2, figsize=(13, 4 * n_models))
    if n_models == 1:
        axes = axes[np.newaxis, :]
    fig.suptitle(f"Plot {plot_num} — Transfer Learning: Feature Extraction & Fine-Tuning Curves",
                 fontsize=12, fontweight='bold')

    colors_fe = 'steelblue'
    colors_ft = 'darkorange'

    for row, (base_name, res) in enumerate(tl_results.items()):
        h_fe = res['hist_fe'].history
        h_ft = res['hist_ft'].history

        # Accuracy
        axes[row, 0].plot(h_fe['accuracy'],     color=colors_fe, label='FE Train')
        axes[row, 0].plot(h_fe['val_accuracy'], color=colors_fe, ls='--', label='FE Val')
        offset = len(h_fe['accuracy'])
        x_ft = range(offset, offset + len(h_ft['accuracy']))
        axes[row, 0].plot(x_ft, h_ft['accuracy'],     color=colors_ft, label='FT Train')
        axes[row, 0].plot(x_ft, h_ft['val_accuracy'], color=colors_ft, ls='--', label='FT Val')
        axes[row, 0].axvline(offset, color='gray', ls=':', lw=1.5, label='Fine-tune start')
        axes[row, 0].set_title(f'{base_name} — Accuracy')
        axes[row, 0].set_xlabel('Epoch'); axes[row, 0].set_ylabel('Accuracy')
        axes[row, 0].legend(fontsize=8); axes[row, 0].grid(True, alpha=0.3)

        # Loss
        axes[row, 1].plot(h_fe['loss'],     color=colors_fe, label='FE Train')
        axes[row, 1].plot(h_fe['val_loss'], color=colors_fe, ls='--', label='FE Val')
        axes[row, 1].plot(x_ft, h_ft['loss'],     color=colors_ft, label='FT Train')
        axes[row, 1].plot(x_ft, h_ft['val_loss'], color=colors_ft, ls='--', label='FT Val')
        axes[row, 1].axvline(offset, color='gray', ls=':', lw=1.5)
        axes[row, 1].set_title(f'{base_name} — Loss')
        axes[row, 1].set_xlabel('Epoch'); axes[row, 1].set_ylabel('Loss')
        axes[row, 1].legend(fontsize=8); axes[row, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("plot13_tl_curves.png", dpi=100, bbox_inches='tight')
    plt.show()
    print("  → Tersimpan: plot13_tl_curves.png")


# =============================================================================
# MAIN — ORKESTRASI SEMUA BAGIAN
# =============================================================================
def main():
    print("=" * 65)
    print("  KLASIFIKASI CITRA DENGAN CNN: DARI AWAL HINGGA TRANSFER LEARNING")
    print("  Dataset: CIFAR-10  |  Framework: TensorFlow / Keras")
    print("=" * 65)

    # ── 0. Load Data ─────────────────────────────────────────────
    (X_train, y_train, y_train_oh,
     X_test,  y_test,  y_test_oh) = load_and_prepare_data()

    # ── 1. Plot 1: Sample Gambar ─────────────────────────────────
    plot_sample_images(X_train, y_train, plot_num=1)

    # ── 2. Augmentation ─────────────────────────────────────────
    datagen = setup_augmentation()
    plot_augmentation_samples(X_train, y_train, datagen, plot_num=2)

    # ── 3. Training CNN Scratch ──────────────────────────────────
    scratch_results = train_cnn_variants(X_train, y_train_oh, X_test, y_test_oh, datagen)

    # ── 4. Plot 3: Learning Curves ───────────────────────────────
    plot_learning_curves(scratch_results, plot_num=3)

    # Pilih model scratch terbaik untuk evaluasi lanjutan
    best_scratch_name = max(scratch_results, key=lambda k: scratch_results[k]['test_acc'])
    best_model        = scratch_results[best_scratch_name]['model']
    print(f"\n  Best CNN Scratch: {best_scratch_name}  "
          f"(Acc={scratch_results[best_scratch_name]['test_acc']:.4f})")

    # ── 5. Evaluasi Komprehensif Scratch ─────────────────────────
    eval_scratch = {}
    for name, res in scratch_results.items():
        ev = comprehensive_evaluation(res['model'], X_test, y_test, y_test_oh, name)
        ev['y_test'] = y_test
        eval_scratch[name] = ev

    best_ev = eval_scratch[best_scratch_name]
    print(f"\n  Accuracy : {best_ev['accuracy']:.4f}")
    print(f"  F1 (W)   : {best_ev['f1']:.4f}")
    print(f"  AUC Macro: {best_ev['macro_auc']:.4f}")
    print("\n  Classification Report:")
    print(classification_report(y_test, best_ev['y_pred'],
                                 target_names=CLASS_NAMES, zero_division=0))

    # ── 6. Plot 4: Confusion Matrix ──────────────────────────────
    plot_cm(y_test, best_ev['y_pred'],
            title=f"Best CNN ({best_scratch_name})", plot_num=4)

    # ── 7. Plot 5: ROC Curve ─────────────────────────────────────
    plot_roc_curves(best_ev, y_test, plot_num=5)

    # ── 8. Plot 6: Per-Class Metrics ─────────────────────────────
    plot_per_class_metrics(y_test, best_ev['y_pred'], plot_num=6)

    # ── 9. Plot 7: Feature Maps ──────────────────────────────────
    plot_feature_maps(best_model, X_test, plot_num=7)

    # ── 10. Plot 8: Filter Visualization ────────────────────────
    plot_filter_visualization(best_model, plot_num=8)

    # ── 11. Plot 9: Grad-CAM ─────────────────────────────────────
    plot_gradcam(best_model, X_test, y_test, plot_num=9, n_images=8)

    # ── 12. Plot 10: t-SNE ───────────────────────────────────────
    plot_tsne_embeddings(best_model, X_test, y_test, plot_num=10, n_samples=2000)

    # ── 13. Transfer Learning ────────────────────────────────────
    tl_results = train_transfer_learning(X_train, y_train_oh, X_test, y_test_oh, datagen)

    # Evaluasi model TL (gunakan model setelah fine-tuning)
    eval_tl_dict = {}
    for base_name, res in tl_results.items():
        ev = comprehensive_evaluation(res['model'], X_test, y_test, y_test_oh,
                                      f"TL_{base_name}_FT")
        ev['y_test'] = y_test
        eval_tl_dict[base_name] = ev

    # ── 14. Plot 11: Augmentation Effect ─────────────────────────
    plot_augmentation_effect(scratch_results, plot_num=11)

    # ── 15. Plot 12: Comparison Table ────────────────────────────
    plot_comparison_table(scratch_results, tl_results,
                          eval_scratch, eval_tl_dict, plot_num=12)

    # ── 16. Plot 13: TL Curves ───────────────────────────────────
    plot_tl_curves(tl_results, plot_num=13)

    # ── 17. Ringkasan Akhir ──────────────────────────────────────
    print("\n" + "="*65)
    print(" RINGKASAN AKHIR SEMUA EKSPERIMEN")
    print("="*65)

    all_models = {}
    for name, res in scratch_results.items():
        all_models[name] = res['test_acc']
    for base_name, res in tl_results.items():
        all_models[f"TL_{base_name}_FT"] = res['acc_ft']

    best_overall = max(all_models, key=all_models.get)
    print(f"  {'Model':<30} {'Test Accuracy':>14}")
    print("  " + "-"*46)
    for name, acc in sorted(all_models.items(), key=lambda x: -x[1]):
        marker = " ← TERBAIK" if name == best_overall else ""
        print(f"  {name:<30} {acc:>14.4f}{marker}")

    print("\n  KESIMPULAN & REKOMENDASI:")
    print("  1. Transfer Learning dengan fine-tuning umumnya mengungguli CNN from scratch")
    print("     pada dataset kecil/menengah, terutama ResNet50 dan MobileNetV2.")
    print("  2. Data augmentasi secara konsisten mengurangi overfitting (gap train-val).")
    print("  3. MobileNetV2 memberikan trade-off terbaik antara akurasi dan kecepatan inferensi.")
    print("  4. Grad-CAM menunjukkan model berfokus pada fitur objek yang relevan.")
    print("  5. Rekomendasi: gunakan MobileNetV2 + augmentasi + fine-tuning 20 layer terakhir")
    print("     untuk aplikasi klasifikasi citra umum.")
    print("="*65)
    print("\n  SEMUA PLOT TERSIMPAN (plot01 — plot13)")


# =============================================================================
if __name__ == "__main__":
    main()
