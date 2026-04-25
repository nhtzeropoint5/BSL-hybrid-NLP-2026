"""
BDSL-49 → MediaPipe landmark sequences converter.

Converts BDSL-49 cropped recognition images (JPG) into 30-frame .npy
landmark sequences compatible with the bsl-hybrid training pipeline.

Each image is processed through MediaPipe Hands + Pose to extract a
258-dimensional keypoint vector, which is then replicated 30 times to
form a static sequence. Small Gaussian noise is added to 4 augmented
copies per image to simulate natural hand variation.

Usage
-----
  cd bsl-hybrid

  # If BDSL-49 folders are named by label index (0, 1, 2 ... 48):
  python train/process_bdsl49.py --src path/to/recognition_dataset

  # If folders are named by Bangla character (অ, আ, ক ...):
  python train/process_bdsl49.py --src path/to/recognition_dataset --named

  # Limit augmented copies per image (default 4):
  python train/process_bdsl49.py --src path/to/recognition_dataset --aug 2

Expected input structure (either naming convention):
  <src>/
    0/          ← or  অ/
      img001.jpg
      img002.jpg
      ...
    1/          ← or  আ/
      ...
    ...
    48/         ← or  ঞ/
      ...

Output
------
  data/sequences/<letter>/<index>.npy   shape (30, 258)
"""

import os
import sys
import argparse
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.hand_pipeline import HandPipeline
from src.sign_model    import LETTERS, SEQUENCE_LENGTH, FEATURES

DATA_DIR      = os.path.join('data', 'sequences')
IMG_EXTS      = {'.jpg', '.jpeg', '.png', '.bmp'}
AUGMENT_NOISE = 0.005   # std dev of Gaussian noise for augmentation


def _get_keypoints(hp, img_path):
    """
    Run MediaPipe on one image. Returns (258,) float32 or None if no hand found.
    """
    frame = cv2.imread(img_path)
    if frame is None:
        return None

    # Resize to a standard size for consistent landmark extraction
    frame = cv2.resize(frame, (640, 480))

    keypoints, has_hands, _ = hp.process(frame)

    if not has_hands:
        return None

    return keypoints.astype(np.float32)


def _make_sequence(keypoints, n_aug):
    """
    From a single (258,) keypoint vector produce:
      1 original + n_aug augmented sequences, each shape (30, 258).
    """
    base = np.tile(keypoints, (SEQUENCE_LENGTH, 1))   # (30, 258)
    seqs = [base]
    for _ in range(n_aug):
        noise = np.random.normal(0, AUGMENT_NOISE, base.shape).astype(np.float32)
        seqs.append((base + noise).astype(np.float32))
    return seqs


def _resolve_folders(src, named):
    """
    Returns list of (letter, folder_path) pairs for all 49 BDSL-49 classes.
    named=False: folders expected to be named 0..48 (label index)
    named=True : folders expected to be named by the Bangla character
    """
    pairs = []
    for idx, letter in enumerate(LETTERS):
        if named:
            folder = os.path.join(src, letter)
        else:
            folder = os.path.join(src, str(idx))

        if os.path.isdir(folder):
            pairs.append((letter, folder))
        else:
            print(f"  [warn] Folder not found for label {idx} ('{letter}'): {folder}")
    return pairs


def main():
    parser = argparse.ArgumentParser(
        description='Convert BDSL-49 images to MediaPipe landmark sequences')
    parser.add_argument('--src',   required=True,
                        help='Path to BDSL-49 recognition dataset root')
    parser.add_argument('--named', action='store_true',
                        help='Folders are named by Bangla character instead of label index')
    parser.add_argument('--aug',   type=int, default=4,
                        help='Augmented copies per image (default: 4)')
    args = parser.parse_args()

    src = args.src
    if not os.path.isdir(src):
        print(f"ERROR: Source directory not found: {src}")
        sys.exit(1)

    print("=== BDSL-49 → Landmark Sequence Converter ===\n")
    print(f"  Source  : {src}")
    print(f"  Output  : {DATA_DIR}")
    print(f"  Augment : {args.aug} copies per image")
    print(f"  Naming  : {'Bangla character' if args.named else 'label index (0–48)'}\n")

    hp = HandPipeline()
    os.makedirs(DATA_DIR, exist_ok=True)

    pairs = _resolve_folders(src, args.named)
    if not pairs:
        print("ERROR: No class folders found. Check --src path and --named flag.")
        sys.exit(1)

    total_saved   = 0
    total_skipped = 0

    for letter, folder in pairs:
        out_dir = os.path.join(DATA_DIR, letter)
        os.makedirs(out_dir, exist_ok=True)

        images = sorted(
            f for f in os.listdir(folder)
            if os.path.splitext(f)[1].lower() in IMG_EXTS
        )

        if not images:
            print(f"  '{letter}' — no images found in {folder}, skipping.")
            continue

        saved   = 0
        skipped = 0
        seq_idx = len([f for f in os.listdir(out_dir) if f.endswith('.npy')])

        for fname in images:
            img_path  = os.path.join(folder, fname)
            keypoints = _get_keypoints(hp, img_path)

            if keypoints is None:
                skipped += 1
                continue

            for seq in _make_sequence(keypoints, args.aug):
                save_path = os.path.join(out_dir, f"{seq_idx}.npy")
                np.save(save_path, seq)
                seq_idx += 1
                saved   += 1

        total_saved   += saved
        total_skipped += skipped
        print(f"  '{letter}' — {len(images)} images → {saved} sequences "
              f"({skipped} skipped, no hand detected)")

    print(f"\n=== Done ===")
    print(f"  Total sequences saved : {total_saved}")
    print(f"  Total images skipped  : {total_skipped}")
    print(f"\nNext step:  python train/train_sign.py")


if __name__ == '__main__':
    main()
