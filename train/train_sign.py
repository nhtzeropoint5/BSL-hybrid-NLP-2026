"""
Train the unified Transformer classifier (letters + words).

Usage
-----
  cd bsl-hybrid
  python train/train_sign.py
  python train/train_sign.py --signer-split    # signer-independent eval
  python train/train_sign.py --no-augment      # skip augmentation

Input  : data/sequences/<sign>/<signer_id>_<index>.npy   shape (30, 258)
Output : models/sign_classifier.h5
"""

import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.sign_model import SIGNS, LETTERS, WORDS, FEATURES, SEQUENCE_LENGTH, MODEL_PATH

DATA_DIR = os.path.join('data', 'sequences')


# ── augmentation ──────────────────────────────────────────────────────────────

def augment_sequence(seq):
    """5 augmented copies of seq (SEQUENCE_LENGTH, FEATURES).

    Keypoint layout per frame (258 values):
      [0:63]    left  hand : 21 lm × (x, y, z)
      [63:126]  right hand : 21 lm × (x, y, z)
      [126:258] pose       : 33 lm × (x, y, z, visibility)

    Pose left/right pairs (landmark indices):
      shoulders 11↔12, elbows 13↔14, wrists 15↔16,
      hips 23↔24, knees 25↔26, ankles 27↔28,
      heels 29↔30, foot index 31↔32,
      eyes 1↔4, ears 5↔6, eye inner 2↔5... (use conservative set below)
    """
    # Pose landmark pairs to swap on mirror flip (left_idx, right_idx)
    POSE_SWAP_PAIRS = [
        (11, 12), (13, 14), (15, 16), (17, 18), (19, 20),
        (21, 22), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32),
        (1, 4), (2, 5), (3, 6), (7, 8),
    ]

    copies = []

    # 1. Mirror flip — swap left/right hand data, swap pose body sides,
    #    negate all X coords. Simulates a left-handed signer.
    flipped = seq.copy()

    # swap hands
    lh = flipped[:, :63].copy()
    rh = flipped[:, 63:126].copy()
    flipped[:, :63]   = rh
    flipped[:, 63:126] = lh

    # swap pose left/right landmark pairs
    pose_start = 126
    for li, ri in POSE_SWAP_PAIRS:
        l_slice = slice(pose_start + li * 4, pose_start + li * 4 + 4)
        r_slice = slice(pose_start + ri * 4, pose_start + ri * 4 + 4)
        tmp                  = flipped[:, l_slice].copy()
        flipped[:, l_slice]  = flipped[:, r_slice]
        flipped[:, r_slice]  = tmp

    # negate X coords: hands (every 3rd from 0) + pose (every 4th from 126)
    flipped[:, 0:126:3]       *= -1   # hand X coords
    flipped[:, 126::4]        *= -1   # pose X coords
    copies.append(flipped)

    # 2. Gaussian noise — small random perturbation to all coordinates
    copies.append(
        (seq + np.random.normal(0, 0.005, seq.shape)).astype(np.float32))

    # 3. Frame dropout — randomly zero out 1–3 frames (simulate occlusion)
    dropped  = seq.copy()
    n_drop   = np.random.randint(1, 4)   # 1, 2, or 3 frames
    drop_idx = np.random.choice(SEQUENCE_LENGTH, n_drop, replace=False)
    dropped[drop_idx] = 0.0
    copies.append(dropped)

    # 4. Coordinate jitter — slight uniform translation of whole skeleton
    copies.append(
        (seq + np.random.uniform(-0.02, 0.02, (1, FEATURES))).astype(np.float32))

    # 5. Time warp — randomly stretch or compress the sequence
    #    Stretch first half (slow start), compress second (fast finish), or vice versa
    half   = SEQUENCE_LENGTH // 2
    shift  = np.random.randint(2, 6)     # random warp strength 2–5 frames
    if np.random.rand() > 0.5:
        first  = np.linspace(0, half - 1, half + shift).astype(int)
        second = np.linspace(half, SEQUENCE_LENGTH - 1, half - shift).astype(int)
    else:
        first  = np.linspace(0, half - 1, half - shift).astype(int)
        second = np.linspace(half, SEQUENCE_LENGTH - 1, half + shift).astype(int)
    idx = np.clip(np.concatenate([first, second]), 0, SEQUENCE_LENGTH - 1)
    copies.append(seq[idx[:SEQUENCE_LENGTH]])

    return copies


def apply_augmentation(X, y):
    X_out, y_out = list(X), list(y)
    for seq, label in zip(X, y):
        for aug in augment_sequence(seq):
            X_out.append(aug)
            y_out.append(label)
    return np.array(X_out, dtype=np.float32), np.array(y_out, dtype=np.int32)


# ── data loading ──────────────────────────────────────────────────────────────

def load_dataset():
    X, y, signers = [], [], []
    letter_count  = 0
    word_count    = 0

    print("Loading sequences:\n")
    print(f"  {'Sign':<16s}  {'Type':<8s}  Samples")
    print(f"  {'-'*40}")

    for label, sign in enumerate(SIGNS):
        sign_dir = os.path.join(DATA_DIR, sign)
        if not os.path.isdir(sign_dir):
            continue

        files = sorted(f for f in os.listdir(sign_dir) if f.endswith('.npy'))
        if not files:
            continue

        loaded = 0
        for fname in files:
            seq = np.load(os.path.join(sign_dir, fname))
            if seq.shape == (SEQUENCE_LENGTH, FEATURES):
                X.append(seq)
                y.append(label)
                signer_id = fname.split('_')[0] if '_' in fname else 'default'
                signers.append(signer_id)
                loaded += 1

        kind = 'LETTER' if sign in LETTERS else 'WORD'
        if loaded:
            print(f"  {sign:<16s}  {kind:<8s}  {loaded}")
            if kind == 'LETTER':
                letter_count += loaded
            else:
                word_count += loaded

    print(f"\n  Letter samples : {letter_count}")
    print(f"  Word samples   : {word_count}")
    print(f"  Total          : {letter_count + word_count}\n")

    return (np.array(X, dtype=np.float32),
            np.array(y, dtype=np.int32),
            np.array(signers))


# ── splits ────────────────────────────────────────────────────────────────────

def signer_independent_split(X, y, signers):
    unique = sorted(set(signers))
    if len(unique) < 3:
        print(f"  [warn] Only {len(unique)} signer(s) — need 3+ for signer-independent split.")
        print("         Falling back to random split.\n")
        return random_split(X, y)

    test_signer   = unique[-1]
    val_signer    = unique[-2]
    train_signers = set(unique[:-2])

    print(f"  Signer-independent split:")
    print(f"    Train : {sorted(train_signers)}")
    print(f"    Val   : {val_signer}")
    print(f"    Test  : {test_signer}\n")

    tr  = (signers != test_signer) & (signers != val_signer)
    val = signers == val_signer
    te  = signers == test_signer

    return X[tr], X[val], X[te], y[tr], y[val], y[te]


def random_split(X, y):
    from sklearn.model_selection import train_test_split
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y)
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.5, random_state=42, stratify=y_tmp)
    return X_tr, X_val, X_te, y_tr, y_val, y_te


# ── model ─────────────────────────────────────────────────────────────────────

def build_transformer(num_classes, d_model=64, num_heads=8, ff_dim=64, num_blocks=2):
    """
    Compact Transformer Encoder (paper architecture):
      Input (30, 258)
      → Linear projection → Positional embedding
      → 2× [LayerNorm → MHA → residual → LayerNorm → FFN(GeLU) → residual]
      → GlobalMaxPool → Dropout → Dense(relu, L2) → Dense(softmax)
    """
    import tensorflow as tf
    from tensorflow.keras import layers, Model

    inputs = tf.keras.Input(shape=(SEQUENCE_LENGTH, FEATURES))

    x         = layers.Dense(d_model)(inputs)
    positions = tf.range(start=0, limit=SEQUENCE_LENGTH)
    pos_emb   = layers.Embedding(input_dim=SEQUENCE_LENGTH, output_dim=d_model)(positions)
    x         = x + pos_emb

    for _ in range(num_blocks):
        attn = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=0.1,
        )(x, x)
        x = layers.LayerNormalization(epsilon=1e-6)(x + attn)

        ff = layers.Dense(ff_dim, activation='gelu')(x)
        ff = layers.Dense(d_model)(ff)
        x  = layers.LayerNormalization(epsilon=1e-6)(x + ff)

    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(
        d_model, activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(0.01),
    )(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--signer-split', action='store_true')
    parser.add_argument('--no-augment',   action='store_true')
    args = parser.parse_args()

    print("=== BSL Hybrid — Transformer Training ===\n")
    print(f"  Classes : {len(SIGNS)} total  ({len(LETTERS)} letters + {len(WORDS)} words)\n")

    X, y, signers = load_dataset()

    if len(X) == 0:
        print("No training data found. Run  python train/collect_data.py  first.")
        return

    unique_classes = len(np.unique(y))
    print(f"Unique signers : {sorted(set(signers))}")
    print(f"Classes found  : {unique_classes} / {len(SIGNS)}\n")

    if unique_classes < 2:
        print("Need at least 2 classes.")
        return

    if args.signer_split:
        X_tr, X_val, X_te, y_tr, y_val, y_te = signer_independent_split(X, y, signers)
    else:
        X_tr, X_val, X_te, y_tr, y_val, y_te = random_split(X, y)

    if not args.no_augment:
        before = len(X_tr)
        X_tr, y_tr = apply_augmentation(X_tr, y_tr)
        print(f"Augmented training set: {before} → {len(X_tr)} samples\n")

    print(f"Train : {len(X_tr)}   Val : {len(X_val)}   Test : {len(X_te)}\n")

    model = build_transformer(len(SIGNS))
    model.summary()
    print()

    os.makedirs('models', exist_ok=True)

    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

    model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=[
            ModelCheckpoint(MODEL_PATH, save_best_only=True, verbose=1),
            EarlyStopping(patience=15, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(factor=0.5, patience=7, min_lr=1e-7, verbose=1),
        ],
    )

    loss, acc = model.evaluate(X_te, y_te, verbose=0)
    split_label = 'signer-independent' if args.signer_split else 'random'
    print(f"\nTest accuracy ({split_label}) : {acc:.2%}")
    print(f"Model saved to               : {MODEL_PATH}")

    # ── confusion matrix ──────────────────────────────────────────────────────
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, classification_report

    y_pred = np.argmax(model.predict(X_te, verbose=0), axis=1)

    # per-class report (precision / recall / F1) — printed to console
    present = sorted(set(y_te))
    present_labels = [SIGNS[i] for i in present]
    print("\n" + classification_report(y_te, y_pred, labels=present, target_names=present_labels))

    # full matrix as PNG
    cm = confusion_matrix(y_te, y_pred, labels=present)
    n  = len(present)
    fig, ax = plt.subplots(figsize=(max(14, n // 4), max(12, n // 4)))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(n)); ax.set_xticklabels(present_labels, rotation=90, fontsize=7)
    ax.set_yticks(range(n)); ax.set_yticklabels(present_labels, fontsize=7)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title(f'Confusion Matrix — test acc {acc:.2%}')
    fig.tight_layout()
    cm_path = os.path.join('models', 'confusion_matrix.png')
    fig.savefig(cm_path, dpi=120)
    plt.close(fig)
    print(f"Confusion matrix saved to    : {cm_path}")

    print("\nNext step:  python main.py")


if __name__ == '__main__':
    main()
