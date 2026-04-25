"""
Final_Dataset (102 words) → MediaPipe landmark sequences converter.

Reads JPG images from data/dataset/Final_Dataset/<EnglishFolder>/,
runs MediaPipe Hands + Pose on each image, replicates the result to a
30-frame sequence, and saves .npy files to data/sequences/<BanglaLabel>/.

Usage
-----
  cd bsl-hybrid
  python train/process_word_dataset.py

  # Limit augmented copies per image (default 4):
  python train/process_word_dataset.py --aug 2

Output
------
  data/sequences/<BanglaLabel>/<index>.npy   shape (30, 258)
"""

import os
import sys
import argparse
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.hand_pipeline import HandPipeline
from src.sign_model    import WORDS, SEQUENCE_LENGTH

DATASET_DIR   = os.path.join('data', 'dataset', 'Final_Dataset')
OUT_DIR       = os.path.join('data', 'sequences')
IMG_EXTS      = {'.jpg', '.jpeg', '.png', '.bmp'}
AUGMENT_NOISE = 0.005

# ── Folder name → Bangla label mapping (alphabetical folder order) ────────────
FOLDER_TO_BANGLA = {
    'Ajebaje(Nonsense)':           'আজেবাজে',
    'Akash(Sky)':                  'আকাশ',
    'Alada(Separate)':             'আলাদা',
    'Allah(God)':                  'আল্লাহ',
    'Asha(Hope)':                  'আশা',
    'Bakko(Sentence)':             'বাক্য',
    'Bank':                        'ব্যাংক',
    'Bari(House)':                 'বাড়ি',
    'Bebsa(Business)':             'ব্যবসা',
    'Bepar(Matter)':               'ব্যাপার',
    'Beyam(Exercise)':             'ব্যায়াম',
    'Bhalo(Good)':                 'ভালো',
    'Bhromon(Travel)':             'ভ্রমণ',
    'Bibaho(Marriage)':            'বিবাহ',
    'Biggan(Science)':             'বিজ্ঞান',
    'Biruddhe(Against)':           'বিরুদ্ধে',
    'Bisoy(Subject)':              'বিষয়',
    'Boi(Book)':                   'বই',
    'Boka(Scold)':                 'বোকা',
    'Bristi(Rain)':                'বৃষ্টি',
    'Camera':                      'ক্যামেরা',
    'Cha(Tea)':                    'চা',
    'Chawa(Want)':                 'চাওয়া',
    'Churanto(Conclusion)':        'চূড়ান্ত',
    'Dam(Price)':                  'দাম',
    'Daraw(Stop)':                 'দাঁড়াও',
    'Dawat(Invitation)':           'দাওয়াত',
    'Dharona(Idea)':               'ধারণা',
    'Dhowa(Smoke)':                'ধোঁয়া',
    'Dokandar(Shopkepper)':        'দোকানদার',
    'Dol(Team)':                   'দল',
    'Dowa_kora(Pray for someone)': 'দোয়া করা',
    'Druto(Quick)':                'দ্রুত',
    'Dupur(Noon)':                 'দুপুর',
    'Durgondho(Bad Smell)':        'দুর্গন্ধ',
    'Ful(Flower)':                 'ফুল',
    'Gari(Car)':                   'গাড়ি',
    'Ghi(Clarified Butter)':       'ঘি',
    'Ghori(Clock)':                'ঘড়ি',
    'Ghosito_howa(Announce)':      'ঘোষণা',
    'Ghumano(Sleep)':              'ঘুমানো',
    'Hasi(Smile)':                 'হাসি',
    'Hassokor(Funny)':             'হাস্যকর',
    'Hat(Hand)':                   'হাত',
    'Injection':                   'ইনজেকশন',
    'Jailkhana(Prison)':           'জেলখানা',
    'Jinish(Object)':              'জিনিস',
    'Jogajog(Communication)':      'যোগাযোগ',
    'Kachi(Scissors)':             'কাঁচি',
    'Kapor(Cloth)':                'কাপড়',
    'Kashi(Cough)':                'কাশি',
    'Khawa(Eating)':               'খাওয়া',
    'Khoma(Forgive)':              'ক্ষমা',
    'Klanto(Tired)':               'ক্লান্ত',
    'Kukur(Dog)':                  'কুকুর',
    'Mach(Fish)':                  'মাছ',
    'Matha(Head)':                 'মাথা',
    'Matha_betha(Headache)':       'মাথাব্যথা',
    'Mongol(Good Luck)':           'মঙ্গল',
    'Moyla(Waste)':                'ময়লা',
    'Name':                        'নাম',
    'Norachora(Moving)':           'নড়াচড়া',
    'Oishodh(Medicine)':           'ওষুধ',
    'Ojon(Weight)':                'ওজন',
    'Onushoron(Follow)':           'অনুসরণ',
    'Opomanjonok(Insulting)':      'অপমানজনক',
    'Osustho(Sick)':               'অসুস্থ',
    'Petuk(Gluttony)':             'পেটুক',
    'Phone':                       'ফোন',
    'Pochondo(Choice)':            'পছন্দ',
    'Porikkha(Exam)':              'পরীক্ষা',
    'Poriskar(Clean)':             'পরিষ্কার',
    'Prostut(Ready)':              'প্রস্তুত',
    'Protarona(Betray)':           'প্রতারণা',
    'Raat(Night)':                 'রাত',
    'Rajdhani(Capital)':           'রাজধানী',
    'Rasta(Road)':                 'রাস্তা',
    'Sabdhan(Caution)':            'সাবধান',
    'Shajano(Arrangement)':        'সাজানো',
    'Shasti(Punishment)':          'শাস্তি',
    'Shokal(Morning)':             'সকাল',
    'Shokti(Power)':               'শক্তি',
    'Shosta(Cheap)':               'সস্তা',
    'Shotru(Enemy)':               'শত্রু',
    'Soman(Equal)':                'সমান',
    'Somossha(Problem)':           'সমস্যা',
    'Somoy(Time)':                 'সময়',
    'Songbad(News)':               'সংবাদ',
    'Sonkirno(Shrink)':            'সংকীর্ণ',
    'Sorto(Condition)':            'শর্ত',
    'Table':                       'টেবিল',
    'Taka(Money)':                 'টাকা',
    'Tamasha(Joke)':               'তামাশা',
    'Tapmatra(Temperature)':       'তাপমাত্রা',
    'Tarikh(Date)':                'তারিখ',
    'Toiri_kora(Create something)':'তৈরি করা',
    'Tumi(You)':                   'তুমি',
    'Unnoto(Improved)':            'উন্নত',
    'Upor(Up)':                    'উপর',
    'Vaggo(Luck)':                 'ভাগ্য',
    'Vari(Heavy)':                 'ভারী',
    'Vule_jawa(Forget Something)': 'ভুলে যাওয়া',
}


def _get_keypoints(hp, img_path):
    frame = cv2.imread(img_path)
    if frame is None:
        return None
    frame = cv2.resize(frame, (640, 480))
    keypoints, has_hands, _ = hp.process(frame)
    if not has_hands:
        return None
    return keypoints.astype(np.float32)


def _make_sequences(keypoints, n_aug):
    base = np.tile(keypoints, (SEQUENCE_LENGTH, 1))
    seqs = [base]
    for _ in range(n_aug):
        noise = np.random.normal(0, AUGMENT_NOISE, base.shape).astype(np.float32)
        seqs.append((base + noise).astype(np.float32))
    return seqs


def main():
    parser = argparse.ArgumentParser(
        description='Convert Final_Dataset word images to landmark sequences')
    parser.add_argument('--aug', type=int, default=4,
                        help='Augmented copies per image (default: 4)')
    args = parser.parse_args()

    if not os.path.isdir(DATASET_DIR):
        print(f"ERROR: Dataset not found at '{DATASET_DIR}'")
        print("       Expected: data/dataset/Final_Dataset/")
        sys.exit(1)

    # Verify all 102 WORDS have a folder mapping
    bangla_to_folder = {v: k for k, v in FOLDER_TO_BANGLA.items()}
    missing = [w for w in WORDS if w not in bangla_to_folder]
    if missing:
        print(f"WARNING: No folder mapping for: {missing}")

    print("=== Final_Dataset (102 Words) → Landmark Sequence Converter ===\n")
    print(f"  Source  : {DATASET_DIR}")
    print(f"  Output  : {OUT_DIR}")
    print(f"  Words   : {len(WORDS)}")
    print(f"  Augment : {args.aug} copies per image\n")

    hp = HandPipeline()
    os.makedirs(OUT_DIR, exist_ok=True)

    total_saved   = 0
    total_skipped = 0

    for bangla in WORDS:
        folder_name = bangla_to_folder.get(bangla)
        if not folder_name:
            print(f"  '{bangla}' — no folder mapping, skipping.")
            continue

        src_dir = os.path.join(DATASET_DIR, folder_name)
        if not os.path.isdir(src_dir):
            print(f"  '{bangla}' — folder not found: {src_dir}, skipping.")
            continue

        out_dir = os.path.join(OUT_DIR, bangla)
        os.makedirs(out_dir, exist_ok=True)

        images = sorted(
            f for f in os.listdir(src_dir)
            if os.path.splitext(f)[1].lower() in IMG_EXTS
        )

        if not images:
            print(f"  '{bangla}' — no images found, skipping.")
            continue

        saved   = 0
        skipped = 0
        seq_idx = len([f for f in os.listdir(out_dir) if f.endswith('.npy')])

        for fname in images:
            keypoints = _get_keypoints(hp, os.path.join(src_dir, fname))
            if keypoints is None:
                skipped += 1
                continue

            for seq in _make_sequences(keypoints, args.aug):
                np.save(os.path.join(out_dir, f"{seq_idx}.npy"), seq)
                seq_idx += 1
                saved   += 1

        total_saved   += saved
        total_skipped += skipped
        print(f"  '{bangla}' ({folder_name}) — {len(images)} images → {saved} sequences"
              + (f" ({skipped} skipped)" if skipped else ''))

    print(f"\n=== Done ===")
    print(f"  Total sequences saved : {total_saved}")
    print(f"  Total images skipped  : {total_skipped} (no hand detected)")
    print(f"\nNext step:  python train/train_sign.py")


if __name__ == '__main__':
    main()
