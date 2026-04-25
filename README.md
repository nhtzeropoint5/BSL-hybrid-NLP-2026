# BSL-Hybrid — Bangladeshi Sign Language Translator

A real-time Bangladeshi Sign Language (BSL) to Bangla text translator.
Uses your webcam, MediaPipe landmarks, and a compact Transformer model
to recognize 151 signs (49 letters + 102 words) and compose Bangla text live.

---

## Requirements

- Python **3.11**
- A **webcam**
- Windows 10/11 (tested), or Linux with a display server
- ~2 GB disk space for dependencies

---

## Setup

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd bsl-hybrid
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / Mac
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> If you want higher-quality word correction (optional), also install:
> ```bash
> pip install transformers torch
> ```
> This downloads ~700 MB of BanglaBERT weights on first run.

### 4. Download the trained model

Download `sign_classifier.h5` from the link below and place it in the `models/` folder:

```
[Google Drive link — ask the project lead]
```

Your folder should look like:
```
bsl-hybrid/
  models/
    sign_classifier.h5   ← place here
  src/
  train/
  main.py
  ...
```

### 5. Run the app

```bash
python main.py
```

---

## How to Use

| Action | Result |
|---|---|
| Show a **letter sign** to the camera | Character appears in the COMPOSING box |
| **Lower your hand** for ~1 second | Composed word moves to SENTENCE |
| Show the **space sign** | Word boundary — commits word to sentence |
| Show a **word sign** | Word appears directly in SENTENCE |
| Click **Clear** | Resets sentence and composing box |

### Sign types
- **49 letter signs** — vowels, consonants, digits, and special characters from the BDSL-49 dataset. Use these to spell out any Bangla word letter by letter.
- **102 word signs** — dedicated gestures for common everyday Bangla words (ভালো, সময়, বিজ্ঞান, etc.). Faster than fingerspelling.

### Tips for best accuracy
- Ensure **good lighting** on your hands
- Keep your hands within the camera frame
- Face the camera directly
- Hold each sign steady for at least 1 second before moving to the next

---

## Project Structure

```
bsl-hybrid/
  main.py                        # entry point
  requirements.txt
  src/
    sign_model.py                # vocabulary + Transformer wrapper
    translator.py                # BanglaComposer + hybrid routing
    hand_pipeline.py             # MediaPipe hand + pose extraction
    face_pipeline.py             # FER emotion detection
    word_corrector.py            # Levenshtein + BanglaBERT correction
    ui.py                        # PyQt5 application window
  train/
    collect_data.py              # webcam-based data collection tool
    process_bdsl49.py            # convert BDSL-49 images → .npy sequences
    process_word_dataset.py      # convert 102-word images → .npy sequences
    train_sign.py                # train the Transformer model
  data/
    dataset/Final_Dataset/       # 102-word image dataset
    sequences/                   # generated .npy training sequences
  models/
    sign_classifier.h5           # trained model (download separately)
```

---

## Troubleshooting

**App launches but shows "No model" warning**
→ Make sure `sign_classifier.h5` is in the `models/` folder.

**Webcam not opening**
→ Check that no other app is using your camera. Try changing `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)` in `src/ui.py` line 64 if you have multiple cameras.

**Bangla text not displaying correctly**
→ The app uses the **Nirmala UI** font (built into Windows). On Linux, install a Bangla-compatible font and update the font name in `src/ui.py`.

**Low recognition accuracy**
→ Improve lighting, reposition yourself closer to the camera, and ensure your full upper body is visible for pose landmarks.

**`ImportError` on launch**
→ Make sure your virtual environment is activated and `pip install -r requirements.txt` completed without errors.

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| tensorflow | 2.13.0 | Transformer model |
| mediapipe | 0.10.9 | Hand + pose landmark extraction |
| opencv-python | 4.8.1.78 | Webcam capture + image processing |
| PyQt5 | ≥5.15.9 | Desktop UI |
| fer | 22.5.1 | Facial emotion recognition |
| numpy | 1.24.3 | Array operations |
| protobuf | 3.20.3 | MediaPipe dependency |
| scikit-learn | ≥1.3.0 | Train/test split utilities |

---

## Dataset Credits

- **BDSL-49** — Hasib et al., *Data in Brief* 49 (2023) 109329
- **Final_Dataset (102 words)** — Dhrubo et al., *Scientific Reports* 16:1154 (2026)
