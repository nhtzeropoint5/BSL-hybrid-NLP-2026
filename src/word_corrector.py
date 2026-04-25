"""
Word corrector for fingerspelled Bangla words.

When BanglaComposer flushes a fingerspelled word, this module checks
whether it is a valid Bangla word and corrects likely misrecognitions.

Two-tier approach
-----------------
Tier 1 — Edit distance (always available, no downloads)
    Finds the closest word in the built-in Bangla lexicon using
    Levenshtein distance. Corrects only if distance ≤ MAX_EDIT_DIST.
    Example: "আমী" → "আমি"  (distance 1, ী→ি)

Tier 2 — BanglaBERT (optional, better quality)
    If `transformers` is installed, uses sagorsarker/bangla-bert-base
    (a BERT model pretrained on Bengali text) via the fill-mask pipeline
    to score candidate corrections.
    Install: pip install transformers torch
    First run downloads ~700 MB model weights (cached after that).

Usage
-----
    corrector = WordCorrector()           # loads BanglaBERT if available
    corrector = WordCorrector(use_lm=False)  # edit distance only

    word, changed = corrector.correct("আমী")
    # → ("আমি", True)

    word, changed = corrector.correct("পানি")
    # → ("পানি", False)   already valid
"""

MAX_EDIT_DIST = 2   # only correct if closest word is within this distance

# ── Bangla lexicon ────────────────────────────────────────────────────────────
# Common Bangla words. Extend freely — the larger the lexicon,
# the better the correction quality.
BANGLA_LEXICON = {
    # pronouns
    'আমি', 'তুমি', 'সে', 'আমরা', 'তোমরা', 'তারা', 'আপনি',
    'এটা', 'ওটা', 'সেটা', 'এখানে', 'ওখানে', 'সেখানে',
    # common verbs (root forms)
    'করা', 'করি', 'করো', 'করে', 'করেছি', 'করব',
    'যাওয়া', 'যাই', 'যাও', 'যায়', 'যাব',
    'আসা', 'আসি', 'আসো', 'আসে', 'আসব',
    'দেখা', 'দেখি', 'দেখো', 'দেখে',
    'খাওয়া', 'খাই', 'খাও', 'খায়',
    'বলা', 'বলি', 'বলো', 'বলে',
    'শোনা', 'শুনি', 'শোনো', 'শোনে',
    'পড়া', 'পড়ি', 'পড়ো', 'পড়ে',
    'লেখা', 'লিখি', 'লেখো', 'লেখে',
    'জানা', 'জানি', 'জানো', 'জানে',
    'বোঝা', 'বুঝি', 'বোঝো', 'বোঝে',
    'চাওয়া', 'চাই', 'চাও', 'চায়',
    'পাওয়া', 'পাই', 'পাও', 'পায়',
    'দেওয়া', 'দিই', 'দাও', 'দেয়',
    'নেওয়া', 'নিই', 'নাও', 'নেয়',
    'হওয়া', 'হই', 'হও', 'হয়', 'হবে',
    'থাকা', 'থাকি', 'থাকো', 'থাকে',
    'ঘুমানো', 'ঘুমাই', 'ঘুমাও', 'ঘুমায়',
    'হাঁটা', 'হাঁটি', 'হাঁটো', 'হাঁটে',
    'দৌড়ানো', 'দৌড়াই',
    # common adjectives
    'ভালো', 'খারাপ', 'বড়', 'ছোট', 'সুন্দর',
    'কঠিন', 'সহজ', 'নতুন', 'পুরনো', 'দ্রুত', 'ধীর',
    'গরম', 'ঠান্ডা', 'ভারী', 'হালকা', 'উচ্চ', 'নিচু',
    'সঠিক', 'ভুল', 'সত্য', 'মিথ্যা',
    # common nouns
    'বাড়ি', 'স্কুল', 'হাসপাতাল', 'বাজার', 'রাস্তা', 'শহর',
    'ডাক্তার', 'শিক্ষক', 'ছাত্র', 'বন্ধু', 'শত্রু',
    'পরিবার', 'মা', 'বাবা', 'ভাই', 'বোন', 'ছেলে', 'মেয়ে',
    'পানি', 'খাবার', 'ভাত', 'রুটি', 'দুধ', 'চা', 'ফল',
    'হাত', 'পা', 'মাথা', 'চোখ', 'কান', 'নাক', 'মুখ',
    'দিন', 'রাত', 'সকাল', 'বিকেল', 'সন্ধ্যা',
    'আজ', 'কাল', 'গতকাল', 'এখন', 'পরে', 'আগে',
    'টাকা', 'কাজ', 'সময়', 'জায়গা', 'নাম', 'কথা',
    # greetings / common expressions
    'হ্যাঁ', 'না', 'ধন্যবাদ', 'দয়া', 'করে', 'দয়া করে',
    'ঠিক', 'আছে', 'ঠিক আছে', 'অবশ্যই', 'হয়তো',
    'সাহায্য', 'জরুরি', 'সমস্যা',
    # question words
    'কী', 'কে', 'কোথায়', 'কখন', 'কেন', 'কীভাবে', 'কতটা',
    # feelings
    'খুশি', 'দুঃখ', 'রাগ', 'ভয়', 'ভালোবাসা', 'ভালোবাসি',
    'অসুস্থ', 'সুস্থ', 'ক্লান্ত', 'ক্ষুধার্ত', 'তৃষ্ণার্ত',
    # misc
    'সে', 'এই', 'ওই', 'সব', 'অনেক', 'কিছু', 'কেউ', 'কোনো',
    'আর', 'কিন্তু', 'তবে', 'যদি', 'তাহলে', 'কারণ',
}


# ── edit distance ─────────────────────────────────────────────────────────────

def levenshtein(a, b):
    """Standard Levenshtein edit distance between two Unicode strings."""
    m, n = len(a), len(b)
    if m < n:
        a, b, m, n = b, a, n, m
    row = list(range(n + 1))
    for i, ca in enumerate(a, 1):
        prev = i
        for j, cb in enumerate(b, 1):
            curr = min(row[j] + 1, prev + 1, row[j - 1] + (ca != cb))
            row[j - 1] = prev
            prev = curr
        row[n] = prev
    return row[n]


# ── corrector ─────────────────────────────────────────────────────────────────

class WordCorrector:
    """
    Corrects a fingerspelled Bangla word using the lexicon + optional BanglaBERT.

    Parameters
    ----------
    use_lm : bool
        Attempt to load BanglaBERT for higher-quality correction.
        Falls back silently to edit distance if not available.
    """

    def __init__(self, use_lm=True):
        self._lexicon = BANGLA_LEXICON
        self._lm      = None
        if use_lm:
            self._try_load_banglaBERT()

    def _try_load_banglaBERT(self):
        try:
            from transformers import pipeline
            print("[WordCorrector] Loading BanglaBERT (first run downloads ~700 MB)…")
            self._lm = pipeline(
                'fill-mask',
                model='sagorsarker/bangla-bert-base',
            )
            print("[WordCorrector] BanglaBERT loaded.")
        except Exception as e:
            print(f"[WordCorrector] BanglaBERT unavailable ({e}). "
                  "Using edit-distance correction.")

    def correct(self, word):
        """
        Args:
            word : str — assembled Bangla word from BanglaComposer

        Returns:
            (corrected_word, was_changed) : (str, bool)
        """
        if not word:
            return word, False

        # Already a valid word — no correction needed
        if word in self._lexicon:
            return word, False

        if self._lm is not None:
            return self._correct_with_lm(word)
        else:
            return self._correct_with_edit_distance(word)

    # ── tier 1: edit distance ─────────────────────────────────────────────────

    def _correct_with_edit_distance(self, word):
        best_word = None
        best_dist = MAX_EDIT_DIST + 1

        for candidate in self._lexicon:
            d = levenshtein(word, candidate)
            if d < best_dist:
                best_dist = d
                best_word = candidate

        if best_word and best_dist <= MAX_EDIT_DIST:
            return best_word, True

        return word, False   # no close match found

    # ── tier 2: BanglaBERT fill-mask ─────────────────────────────────────────

    def _correct_with_lm(self, word):
        """
        Use BanglaBERT to score candidate corrections.

        Strategy: collect edit-distance candidates (dist ≤ MAX_EDIT_DIST),
        then score each by asking BanglaBERT to predict it in a neutral
        sentence context. Return the highest-scoring valid candidate.
        """
        candidates = [
            (levenshtein(word, c), c)
            for c in self._lexicon
            if levenshtein(word, c) <= MAX_EDIT_DIST
        ]

        if not candidates:
            return word, False

        # If only one candidate within range, return it directly
        if len(candidates) == 1:
            return candidates[0][1], True

        # Score candidates with BanglaBERT using a simple sentence template.
        # "[MASK]" is where we want the model to predict the word.
        best_word  = None
        best_score = -1.0

        for _, candidate in candidates:
            prompt = f"সে {self._lm.tokenizer.mask_token} বলল।"
            try:
                results = self._lm(prompt, targets=[candidate])
                score   = results[0]['score'] if results else 0.0
            except Exception:
                score = 0.0

            if score > best_score:
                best_score = score
                best_word  = candidate

        if best_word:
            return best_word, True

        # fallback to closest edit distance
        candidates.sort()
        return candidates[0][1], True
