"""
Unified sign model — letters + words in one vocabulary.

LETTERS : 49 signs from the BDSL-49 dataset (in dataset label order)
           Vowels (6) + Consonants (30) + Digits (10) + Special (2) + ঞ (1)
           → fed to BanglaComposer to assemble words letter-by-letter

WORDS   : common BSL lexical signs (dedicated single gestures)
           → output directly as a word, no assembly needed

SIGNS   : LETTERS + WORDS  (all classes the model predicts)

To add new word signs: append to WORDS and re-collect + re-train.
"""

import os
import numpy as np

# ── BDSL-49 letter classes (49 total, matching dataset label order) ───────────

VOWELS = ['অ', 'আ', 'ই', 'উ', 'এ', 'ও']                        # labels 0–5

CONSONANTS = [                                                      # labels 6–35
    'ক', 'খ', 'গ', 'ঘ',
    'চ', 'ছ', 'জ', 'ঝ',
    'ট', 'ঠ', 'ড', 'ঢ',
    'ত', 'থ', 'দ', 'ধ',
    'প', 'ফ', 'ব', 'ভ', 'ম',
    'য', 'র', 'ল', 'ন', 'স', 'হ', 'ড়',
    'ং', 'ঃ',
]

DIGITS = ['০', '১', '২', '৩', '৪', '৫', '৬', '৭', '৮', '৯']    # labels 36–45

HASANTA = '~'     # label 46 — triggers virama (্) in BanglaComposer
SPACE   = 'space' # label 47 — triggers word boundary

LETTERS = VOWELS + CONSONANTS + DIGITS + [HASANTA, SPACE, 'ঞ']   # 49 total

# ── word lexicon (BSL dedicated signs) ───────────────────────────────────────
# Each entry is a whole BSL sign — one fluid gesture, not spelled out.
# Add / remove freely; re-collect data and retrain when you change this.
WORDS = [
    # 102 BSL word signs — from Final_Dataset (github.com/Exile404/Final_Dataset)
    # Order matches alphabetical folder order in data/dataset/Final_Dataset/
    'আজেবাজে',      # Ajebaje      — Nonsense
    'আকাশ',         # Akash        — Sky
    'আলাদা',        # Alada        — Separate
    'আল্লাহ',       # Allah        — God
    'আশা',          # Asha         — Hope
    'বাক্য',        # Bakko        — Sentence
    'ব্যাংক',       # Bank
    'বাড়ি',         # Bari         — House
    'ব্যবসা',       # Bebsa        — Business
    'ব্যাপার',      # Bepar        — Matter
    'ব্যায়াম',     # Beyam        — Exercise
    'ভালো',         # Bhalo        — Good
    'ভ্রমণ',        # Bhromon      — Travel
    'বিবাহ',        # Bibaho       — Marriage
    'বিজ্ঞান',      # Biggan       — Science
    'বিরুদ্ধে',     # Biruddhe     — Against
    'বিষয়',        # Bisoy        — Subject
    'বই',           # Boi          — Book
    'বোকা',         # Boka         — Scold
    'বৃষ্টি',       # Bristi       — Rain
    'ক্যামেরা',     # Camera
    'চা',           # Cha          — Tea
    'চাওয়া',       # Chawa        — Want
    'চূড়ান্ত',     # Churanto     — Conclusion
    'দাম',          # Dam          — Price
    'দাঁড়াও',      # Daraw        — Stop
    'দাওয়াত',      # Dawat        — Invitation
    'ধারণা',        # Dharona      — Idea
    'ধোঁয়া',       # Dhowa        — Smoke
    'দোকানদার',     # Dokandar     — Shopkeeper
    'দল',           # Dol          — Team
    'দোয়া করা',    # Dowa_kora    — Pray for someone
    'দ্রুত',        # Druto        — Quick
    'দুপুর',        # Dupur        — Noon
    'দুর্গন্ধ',     # Durgondho    — Bad Smell
    'ফুল',          # Ful          — Flower
    'গাড়ি',        # Gari         — Car
    'ঘি',           # Ghi          — Clarified Butter
    'ঘড়ি',         # Ghori        — Clock
    'ঘোষণা',        # Ghosito_howa — Announce
    'ঘুমানো',       # Ghumano      — Sleep
    'হাসি',         # Hasi         — Smile
    'হাস্যকর',      # Hassokor     — Funny
    'হাত',          # Hat          — Hand
    'ইনজেকশন',      # Injection
    'জেলখানা',      # Jailkhana    — Prison
    'জিনিস',        # Jinish       — Object
    'যোগাযোগ',      # Jogajog      — Communication
    'কাঁচি',        # Kachi        — Scissors
    'কাপড়',        # Kapor        — Cloth
    'কাশি',         # Kashi        — Cough
    'খাওয়া',       # Khawa        — Eating
    'ক্ষমা',        # Khoma        — Forgive
    'ক্লান্ত',      # Klanto       — Tired
    'কুকুর',        # Kukur        — Dog
    'মাছ',          # Mach         — Fish
    'মাথা',         # Matha        — Head
    'মাথাব্যথা',    # Matha_betha  — Headache
    'মঙ্গল',        # Mongol       — Good Luck
    'ময়লা',         # Moyla        — Waste
    'নাম',          # Name
    'নড়াচড়া',     # Norachora    — Moving
    'ওষুধ',         # Oishodh      — Medicine
    'ওজন',          # Ojon         — Weight
    'অনুসরণ',       # Onushoron    — Follow
    'অপমানজনক',     # Opomanjonok  — Insulting
    'অসুস্থ',       # Osustho      — Sick
    'পেটুক',        # Petuk        — Gluttony
    'ফোন',          # Phone
    'পছন্দ',        # Pochondo     — Choice
    'পরীক্ষা',      # Porikkha     — Exam
    'পরিষ্কার',     # Poriskar     — Clean
    'প্রস্তুত',     # Prostut      — Ready
    'প্রতারণা',     # Protarona    — Betray
    'রাত',          # Raat         — Night
    'রাজধানী',      # Rajdhani     — Capital
    'রাস্তা',       # Rasta        — Road
    'সাবধান',       # Sabdhan      — Caution
    'সাজানো',       # Shajano      — Arrangement
    'শাস্তি',       # Shasti       — Punishment
    'সকাল',         # Shokal       — Morning
    'শক্তি',        # Shokti       — Power
    'সস্তা',        # Shosta       — Cheap
    'শত্রু',        # Shotru       — Enemy
    'সমান',         # Soman        — Equal
    'সমস্যা',       # Somossha     — Problem
    'সময়',         # Somoy        — Time
    'সংবাদ',        # Songbad      — News
    'সংকীর্ণ',      # Sonkirno     — Shrink
    'শর্ত',         # Sorto        — Condition
    'টেবিল',        # Table
    'টাকা',         # Taka         — Money
    'তামাশা',       # Tamasha      — Joke
    'তাপমাত্রা',    # Tapmatra     — Temperature
    'তারিখ',        # Tarikh       — Date
    'তৈরি করা',     # Toiri_kora   — Create something
    'তুমি',         # Tumi         — You
    'উন্নত',        # Unnoto       — Improved
    'উপর',          # Upor         — Up
    'ভাগ্য',        # Vaggo        — Luck
    'ভারী',         # Vari         — Heavy
    'ভুলে যাওয়া',  # Vule_jawa    — Forget Something
]

# ── combined vocabulary ───────────────────────────────────────────────────────
SIGNS = LETTERS + WORDS   # model predicts one of these

LETTER_SET = set(LETTERS)
WORD_SET   = set(WORDS)

FEATURES        = 258   # 126 (hands) + 132 (pose: 33 landmarks × x,y,z,visibility)
SEQUENCE_LENGTH = 30    # frames per prediction window
MODEL_PATH      = os.path.join('models', 'sign_classifier.h5')


class SignModel:
    def __init__(self):
        self.model     = None
        self.is_loaded = False
        self._load()

    def _load(self):
        if not os.path.exists(MODEL_PATH):
            print(
                f"[SignModel] No model at '{MODEL_PATH}'.\n"
                "           Run  python train/process_bdsl49.py  then  python train/train_sign.py"
            )
            return
        try:
            from tensorflow.keras.models import load_model
            self.model     = load_model(MODEL_PATH)
            self.is_loaded = True
            print(f"[SignModel] Loaded from '{MODEL_PATH}'.")
        except Exception as exc:
            print(f"[SignModel] Failed to load: {exc}")

    def predict(self, sequence):
        """
        Args:
            sequence : np.ndarray  shape (SEQUENCE_LENGTH, FEATURES)
        Returns:
            (label, confidence, kind)
            kind is 'letter' or 'word'
        """
        if not self.is_loaded:
            return None, 0.0, None

        inp   = sequence[np.newaxis, ...]
        probs = self.model.predict(inp, verbose=0)[0]
        idx   = int(np.argmax(probs))
        conf  = float(probs[idx])

        if idx >= len(SIGNS):
            return None, 0.0, None

        label = SIGNS[idx]
        kind  = 'letter' if label in LETTER_SET else 'word'
        return label, conf, kind
