"""
Hybrid translator — routes model output to the right handler:

  letter prediction  →  BanglaComposer  →  assembles words letter-by-letter
  word prediction    →  appended directly to sentence

BanglaComposer supports the BDSL-49 character set:
  Vowels    : অ আ ই উ এ ও
  Consonants: ক–ড়, ং, ঃ, ঞ
  Digits    : ০–৯  (appended directly)
  ~         : hasanta/virama — appends ্ after a consonant
  space     : word boundary — flushes composer into sentence
"""

from collections import deque
import numpy as np

from src.sign_model import (
    SEQUENCE_LENGTH, VOWELS, CONSONANTS, DIGITS,
    HASANTA, SPACE, LETTER_SET, WORD_SET,
)
from src.word_corrector import WordCorrector

CONFIDENCE_THRESHOLD = 0.85   # minimum confidence to accept any sign
COOLDOWN_FRAMES      = 20     # frames to skip after each accepted sign
NO_HAND_RESET        = 15     # no-hand frames before cooldown resets
WORD_BOUNDARY_FRAMES = 30     # no-hand frames that trigger a word boundary

VOWEL_SET     = set(VOWELS)
CONSONANT_SET = set(CONSONANTS)
DIGIT_SET     = set(DIGITS)

# ং and ঃ are post-consonantal diacritics — appended directly, no matra logic
DIACRITIC_SET = {'ং', 'ঃ'}

# Consonants that can take vowel matras (excludes ং ঃ, includes ঞ)
MATRA_CONSONANT_SET = (CONSONANT_SET - DIACRITIC_SET) | {'ঞ'}

# Vowel → matra mapping (অ = inherent vowel, no matra needed)
VOWEL_TO_MATRA = {
    'আ': '\u09BE',   # া
    'ই': '\u09BF',   # ি
    'উ': '\u09C1',   # ু
    'এ': '\u09C7',   # ে
    'ও': '\u09CB',   # ো
}


class BanglaComposer:
    """
    Assembles a stream of signed letters into Bangla Unicode text.

    Rules:
      consonant (ক–ড়, ঞ)    → append, mark last-was-consonant
      vowel after consonant   → append matra (অ = inherent, nothing added)
      vowel at start / after  → append full vowel form
      ~ after consonant       → append ্ (hasanta/virama)
      ং / ঃ                  → append directly
      digit (০–৯)            → flush current word, append digit as standalone
    """

    def __init__(self):
        self._buf                = ''
        self._last_was_consonant = False

    def push(self, letter):
        if letter == HASANTA:                        # ~
            if self._last_was_consonant:
                self._buf += '্'
            self._last_was_consonant = False

        elif letter in DIACRITIC_SET:               # ং ঃ
            self._buf += letter
            self._last_was_consonant = False

        elif letter in MATRA_CONSONANT_SET:          # ক–ড়, ঞ
            self._buf += letter
            self._last_was_consonant = True

        elif letter in VOWEL_SET:                    # অ আ ই উ এ ও
            if letter == 'অ':
                if not self._last_was_consonant:
                    self._buf += 'অ'
                # else: inherent vowel, nothing added
            elif self._last_was_consonant:
                matra = VOWEL_TO_MATRA.get(letter)
                self._buf += matra if matra else letter
            else:
                self._buf += letter
            self._last_was_consonant = False

        elif letter in DIGIT_SET:                    # ০–৯
            self._buf += letter
            self._last_was_consonant = False

    def flush(self):
        """Return composed word and reset buffer."""
        word = self._buf
        self._buf                = ''
        self._last_was_consonant = False
        return word

    def peek(self):
        return self._buf

    def clear(self):
        self._buf                = ''
        self._last_was_consonant = False


class Translator:
    """
    Hybrid translator.

    - Maintains a 30-frame rolling buffer fed to the model each frame.
    - Letters go to BanglaComposer; a pause or 'space' sign flushes the word.
    - Word signs are appended directly to the sentence.
    """

    def __init__(self, use_lm=True):
        self._buffer      = deque(maxlen=SEQUENCE_LENGTH)
        self._composer    = BanglaComposer()
        self._corrector   = WordCorrector(use_lm=use_lm)
        self._sentence    = []          # list of accepted words
        self._cooldown    = 0
        self._no_hand_cnt = 0
        self._last_sign   = None

    # ── public API ────────────────────────────────────────────────────────────

    def predict(self, keypoints, has_hands, model):
        """
        Args:
            keypoints : np.ndarray  shape (FEATURES,)
            has_hands : bool
            model     : SignModel
        Returns:
            (label, confidence, kind)  or  (None, 0.0, None)
        """
        if not has_hands:
            self._no_hand_cnt += 1
            self._buffer.clear()

            if self._no_hand_cnt >= WORD_BOUNDARY_FRAMES:
                self._flush_composer()
                self._last_sign = None
                self._cooldown  = 0
            elif self._no_hand_cnt >= NO_HAND_RESET:
                self._cooldown  = 0
                self._last_sign = None
            return None, 0.0, None

        self._no_hand_cnt = 0
        self._buffer.append(keypoints)

        if self._cooldown > 0:
            self._cooldown -= 1
            return None, 0.0, None

        if len(self._buffer) < SEQUENCE_LENGTH:
            return None, 0.0, None

        sequence = np.array(self._buffer, dtype=np.float32)
        label, conf, kind = model.predict(sequence)

        if label is None or conf < CONFIDENCE_THRESHOLD:
            return None, 0.0, None

        if label == self._last_sign:
            return None, 0.0, None

        self._last_sign = label
        self._cooldown  = COOLDOWN_FRAMES

        if kind == 'letter':
            if label == SPACE:
                # 'space' sign explicitly signals a word boundary
                self._flush_composer()
            else:
                self._composer.push(label)

        else:  # word sign
            self._flush_composer()
            self._sentence.append(label)

        return label, conf, kind

    def get_sentence(self):
        """Full sentence: committed words + any in-progress composed word."""
        parts   = list(self._sentence)
        partial = self._composer.peek()
        if partial:
            parts.append(partial)
        return ' '.join(parts)

    def get_composing(self):
        """Return the letter buffer currently being composed (for UI display)."""
        return self._composer.peek()

    def clear(self):
        self._buffer.clear()
        self._composer.clear()
        self._sentence.clear()
        self._last_sign   = None
        self._cooldown    = 0
        self._no_hand_cnt = 0

    # ── internal ──────────────────────────────────────────────────────────────

    def _flush_composer(self):
        word = self._composer.flush()
        if word:
            corrected, changed = self._corrector.correct(word)
            if changed:
                print(f"[WordCorrector] '{word}' → '{corrected}'")
            self._sentence.append(corrected)
