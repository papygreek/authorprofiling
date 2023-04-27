from pathlib import Path
import pandas as pd
import regex as re
import unicodedata as ud

script_location = Path(__file__).absolute().parent
grave_to_acute = lambda x: ud.normalize('NFC', ud.normalize('NFD', x).translate({ord('\N{COMBINING GRAVE ACCENT}'):ord('\N{COMBINING ACUTE ACCENT}')}))


def get_greek(x):
    a = re.sub(r'[^\p{Greek} ]', '', x)
    return a if a else 'UNK'

def to_tokens(text, onlygreek=False):
    return [get_greek(x) if onlygreek else x for x in text.split()]

def normalize(x):
    return grave_to_acute(" ".join(x.split()).lower())

def run():
    target_files = {
        'writer': 'writer_texts.csv',
        'author': 'author_texts.csv'
    }

    lemmas = set()

    def function_word_lemmas(d):
        text = to_tokens(d['lemmas'], onlygreek=True)
        postags = to_tokens(d['postags'])
        res = []
        for i, p in enumerate(postags):
            if p[0] in ['p', 'c', 'd', 'r']:
                lemmas.add(text[i])
                res.append(text[i])
        return ' '.join(res)

    for target, target_path in target_files.items():

        

        # Read from CSV
        df = pd.read_csv(script_location / target_path, delimiter='\t')
        df['postags'] = df['postags'].map(normalize)
        df['lemmas'] = df['lemmas'].map(normalize)
        df['text'] = df['text'].map(normalize)
        # Remove nans
        df = df.fillna('')

        # Sort

        df['function_words'] = df.apply(function_word_lemmas, axis=1)

    for lemma in sorted(list(lemmas)):
        print(f"'{lemma}',")

run()