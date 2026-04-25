"""
Interactive data collection for the hybrid BSL system.

Collects 30-frame sequences for BOTH letter signs and word signs.
All signs use the same collection procedure — the model learns the
distinction from the data.

Usage
-----
  cd bsl-hybrid
  python train/collect_data.py                        # signer = "default"
  python train/collect_data.py --signer s1            # tag as signer s1
  python train/collect_data.py --only letters         # letters only
  python train/collect_data.py --only words           # words only

Signer-independent evaluation needs data from 3+ signers.
Run once per person with a unique --signer flag.

Output
------
  data/sequences/<sign>/<signer_id>_<index>.npy   shape (30, 258) float32
"""

import os
import sys
import time
import argparse
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.hand_pipeline import HandPipeline
from src.sign_model    import LETTERS, WORDS, SIGNS, SEQUENCE_LENGTH

DATA_DIR       = os.path.join('data', 'sequences')
NUM_SEQUENCES  = 30
COUNTDOWN_SECS = 3


# ── helpers ───────────────────────────────────────────────────────────────────

def _txt(img, text, y, color=(255, 255, 255), scale=0.75, thick=2):
    cv2.putText(img, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick)


def _countdown(cap, hp, sign, secs=COUNTDOWN_SECS):
    for i in range(secs, 0, -1):
        deadline = time.time() + 1.0
        while time.time() < deadline:
            ok, frame = cap.read()
            if not ok:
                return
            frame = cv2.flip(frame, 1)
            _, _, vis = hp.process(frame)
            _txt(vis, f"Get ready: {sign}", 45, (0, 230, 255), 0.9, 2)
            _txt(vis, str(i), 200, (0, 60, 255), 5.0, 6)
            cv2.imshow("BSL Data Collection", vis)
            cv2.waitKey(1)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='BSL hybrid data collection')
    parser.add_argument('--signer', default='default',
                        help='Signer ID (e.g. s1, s2). Use different IDs per person.')
    parser.add_argument('--only', choices=['letters', 'words', 'all'], default='all',
                        help='Collect only letters, only words, or all (default: all)')
    args = parser.parse_args()

    signer_id = args.signer

    if args.only == 'letters':
        signs_to_collect = LETTERS
        label = 'LETTERS ONLY'
    elif args.only == 'words':
        signs_to_collect = WORDS
        label = 'WORDS ONLY'
    else:
        signs_to_collect = SIGNS
        label = 'ALL SIGNS'

    os.makedirs(DATA_DIR, exist_ok=True)
    for sign in signs_to_collect:
        os.makedirs(os.path.join(DATA_DIR, sign), exist_ok=True)

    hp  = HandPipeline()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam.")
        return

    print(f"\n=== BSL Hybrid Data Collection ===")
    print(f"Signer ID          : {signer_id}")
    print(f"Collecting         : {label}")
    print(f"Signs to collect   : {len(signs_to_collect)}")
    print(f"Sequences per sign : {NUM_SEQUENCES}")
    print(f"Frames per sequence: {SEQUENCE_LENGTH}")
    print(f"\nLETTER signs  — hold a static handshape for the letter")
    print(f"WORD signs    — perform the full BSL gesture for the word")
    print(f"\nPress SPACE to start recording. Press Q to quit.\n")

    for s_idx, sign in enumerate(signs_to_collect):
        sign_dir = os.path.join(DATA_DIR, sign)
        existing = len([f for f in os.listdir(sign_dir)
                        if f.endswith('.npy') and f.startswith(f'{signer_id}_')])

        if existing >= NUM_SEQUENCES:
            print(f"[{s_idx+1}/{len(signs_to_collect)}] '{sign}' — {existing} sequences already exist, skipping.")
            continue

        needed = NUM_SEQUENCES - existing
        kind   = 'WORD' if sign in WORDS else 'LETTER'
        print(f"[{s_idx+1}/{len(signs_to_collect)}] {kind}  '{sign}'  — collecting {needed} sequences")

        # wait for SPACE
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            _, _, vis = hp.process(frame)
            _txt(vis, f"{kind}: {sign}  ({s_idx+1}/{len(signs_to_collect)})",
                 38, (0, 240, 240), 1.0, 2)
            if kind == 'WORD':
                _txt(vis, "Perform the full BSL word gesture, then SPACE", 75, (0, 200, 255))
            else:
                _txt(vis, "Hold the letter handshape, then SPACE", 75)
            _txt(vis, f"Collected: {existing}/{NUM_SEQUENCES}", 115, (180, 180, 180), 0.6, 1)
            cv2.imshow("BSL Data Collection", vis)
            k = cv2.waitKey(1) & 0xFF
            if k == ord(' '):
                break
            if k == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                print("\nStopped.")
                return

        _countdown(cap, hp, sign)

        # record sequences
        for seq_i in range(existing, existing + needed):
            sequence       = []
            no_hand_frames = 0

            for f_i in range(SEQUENCE_LENGTH):
                ok, frame = cap.read()
                if not ok:
                    break
                frame = cv2.flip(frame, 1)
                kp, has_hands, vis = hp.process(frame)

                if not has_hands:
                    no_hand_frames += 1

                sequence.append(kp)

                pct = int((f_i / SEQUENCE_LENGTH) * 220)
                cv2.rectangle(vis, (10, 443), (230, 463), (50, 50, 50), -1)
                cv2.rectangle(vis, (10, 443), (10 + pct, 463), (0, 200, 80), -1)
                cv2.rectangle(vis, (10, 443), (230, 463), (180, 180, 180), 1)
                _txt(vis, f"Recording '{sign}'  [{seq_i+1}/{existing+needed}]",
                     30, (0, 255, 80), 0.72, 2)

                if no_hand_frames > SEQUENCE_LENGTH // 2:
                    _txt(vis, "! Hand not detected — reposition", 68, (0, 50, 255), 0.65, 2)

                cv2.imshow("BSL Data Collection", vis)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    return

            save_path = os.path.join(sign_dir, f"{signer_id}_{seq_i}.npy")
            np.save(save_path, np.array(sequence, dtype=np.float32))

            if no_hand_frames > SEQUENCE_LENGTH // 2:
                print(f"  seq {seq_i:03d}  WARNING: hand absent in "
                      f"{no_hand_frames}/{SEQUENCE_LENGTH} frames — re-record recommended.")

        print(f"  '{sign}' done.\n")

    cap.release()
    cv2.destroyAllWindows()
    print("=== Collection complete! ===")
    print("Next step:  python train/train_sign.py")


if __name__ == '__main__':
    main()
