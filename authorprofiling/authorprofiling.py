import pandas as pd
import numpy as np
import regex as re
from pathlib import Path

from natsort import index_natsorted

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from adjustText import adjust_text

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.compose import ColumnTransformer
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.cluster import KMeans

from gensim.models.fasttext import load_facebook_vectors
import unicodedata as ud

from . import fw

pd.set_option('display.max_rows', None)
grave_to_acute = lambda x: ud.normalize('NFC', ud.normalize('NFD', x).translate({ord('\N{COMBINING GRAVE ACCENT}'):ord('\N{COMBINING ACUTE ACCENT}')}))

script_location = Path(__file__).absolute().parent

markers = ('o', '^', 's', 'P', '+', 'D', 'x', '2', '|', '_')

class EmbeddingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, model, tfidf=False):
        self.model = model
        self.tfidf = tfidf

    def fit(self, x, y=None):
        return self

    def embed(self, w, weights):
        vectors = np.asarray([self.model[x] * (weights.get(x, 1)) if x in w else [0.0] for x in w.split()])
        if vectors.size:
            return vectors.mean(axis=0)
        return np.zeros(self.model.vector_size)
    
    def transform(self, X, y=None):
        weights = {}
        if self.tfidf:
            vectorizer = TfidfVectorizer()
            vectorizer.fit_transform(X)
            weights = dict(zip(vectorizer.get_feature_names_out(), vectorizer.idf_))

        embeddings = np.array([self.embed(d, weights) for d in X])
        return pd.DataFrame(embeddings)


def transformers(init_model=None):
    model = None
    if init_model == 'fasttext':
        model = load_facebook_vectors(script_location / 'fasttext-ancientgreek.bin')

    return [
        # Word forms
        (
            'words_bow',
            CountVectorizer(),
            'text',
        ),
        (
            'function_words_bow',
            CountVectorizer(),
            'function_words',
        ),
        (
            'words_tfidf',
            TfidfVectorizer(),
            'text',
        ),
        (
            'function_words_tfidf',
            TfidfVectorizer(),
            'function_words',
        ),

        # Lemmas
        (
            'lemmas_bow',
            CountVectorizer(),
            'lemmas',
        ),
        (
            'function_word_lemmas_bow',
            CountVectorizer(),
            'function_word_lemmas',
        ),
        (
            'lemmas_tfidf',
            TfidfVectorizer(),
            'lemmas',
        ),
        (
            'function_word_lemmas_tfidf',
            TfidfVectorizer(),
            'function_word_lemmas',
        ),

        # Postags
        (
            'postags_bow',
            CountVectorizer(),
            'postags',
        ),
        (
            'function_word_postags_bow',
            CountVectorizer(),
            'function_word_postags',
        ),
        (
            'postags_tfidf',
            TfidfVectorizer(),
            'postags',
        ),
        (
            'function_word_postags_tfidf',
            TfidfVectorizer(),
            'function_word_postags',
        ),

        # Variations
        (
            'variations_bow',
            CountVectorizer(),
            'variations',
        ),
        (
            'variations_tfidf',
            TfidfVectorizer(),
            'variations',
        ),

        # Fasttext
        (
            'words_fasttext',
            EmbeddingTransformer(model),
            'text'
        ),
        (
            'words_fasttext_tfidf',
            EmbeddingTransformer(model, tfidf=True),
            'text'
        ),
        (
            'function_words_fasttext',
            EmbeddingTransformer(model),
            'function_words'
        ),
        (
            'function_words_fasttext_tfidf',
            EmbeddingTransformer(model, tfidf=True),
            'function_words'
        ),
    ]

def normalize(x):
    
    return grave_to_acute(" ".join(x.split()))

def get_token_n(x):
    return len(x.split())

def get_greek(x):
    a = re.sub(r'[^\p{Greek} ]', '', x)
    return a if a else 'UNK'

def to_tokens(text, onlygreek=False):
    return [get_greek(x) if onlygreek else x for x in text.split()]

def function_words(d):
    text = to_tokens(d['text'], onlygreek=True)
    lemmas = to_tokens(d['lemmas'], onlygreek=True)
    res = []
    for i, p in enumerate(lemmas):
        if p in fw.function_word_lemmas:
            res.append(text[i])
    return ' '.join(res)

def function_word_lemmas(d):
    lemmas = to_tokens(d['lemmas'], onlygreek=True)
    res = []
    for i, p in enumerate(lemmas):
        if p in fw.function_word_lemmas:
            res.append(lemmas[i])
    return ' '.join(res)

def function_word_postags(d):
    postags = to_tokens(d['postags'])
    lemmas = to_tokens(d['lemmas'], onlygreek=True)
    res = []
    for i, p in enumerate(lemmas):
        if p in fw.function_word_lemmas:
            res.append(postags[i])
    return ' '.join(res)

def get_stats(target, df, tail):
    if not tail:
        tail = 1000000
        tail_text = ''
    else:
        tail_text = f' (top {tail})'
    
    person_texts = df.groupby("person")["person"].count().sort_values().tail(tail).to_string(header=False)
    print(f'{target.capitalize()}s with most texts{tail_text}: \n--------------------\n{person_texts}\n')

    person_tokens = df.groupby("person")["text_len"].sum().sort_values().tail(tail).to_string(header=False)
    print(f'{target.capitalize()}s with most tokens{tail_text}: \n--------------------\n{person_tokens}\n')
    
    person_variations = df.groupby("person")["variations_len"].sum().sort_values().tail(tail).to_string(header=False)
    print(f'{target.capitalize()}s with most variations{tail_text}: \n--------------------\n{person_variations}\n')
    
    text_len = df.sort_values(by="text_len").tail(tail)[['person', 'text_id', 'text_len']].to_string(index=False)
    print(f'{target.capitalize()} texts with most tokens{tail_text}: \n--------------------\n{text_len}\n')
    
    text_variations = df.sort_values(by="variations_len").tail(tail)[['person', 'text_id', 'variations_len']].to_string(index=False)
    print(f'{target.capitalize()} texts with most variations{tail_text}: \n--------------------\n{text_variations}\n')

def get_column_transformer(features):
    model = None
    if any(['fasttext' in x for x in features]):
        model = 'fasttext'
    elif any(['word2vec' in x for x in features]):
        model = 'word2vec'
    return ColumnTransformer([
        (tr[0], tr[1], tr[2]) 
        for tr in transformers(model) if tr[0] in features
    ])

def run(args):
    target_files = {
        'writer': 'writer_texts.csv',
        'author': 'author_texts.csv'
    }

    # Read from CSV
    df = pd.read_csv(script_location / target_files[args.target], delimiter='\t')

    # Remove nans
    df = df.fillna('')

    # Sort

    df = df.sort_values(
        by="person",
        key=lambda x: np.argsort(index_natsorted(df["person"]))
    )

    # Preprocess
    df['variations'] = df['variations'].map(normalize)
    df['postags'] = df['postags'].map(normalize)
    df['lemmas'] = df['lemmas'].map(normalize)
    df['text'] = df['text'].map(normalize)
    df['variations_len'] = df['variations'].map(get_token_n)
    df['text_len'] = df['text'].map(get_token_n)
    df['function_words'] = df.apply(function_words, axis=1)
    df['function_word_lemmas'] = df.apply(function_word_lemmas, axis=1)
    df['function_word_postags'] = df.apply(function_word_postags, axis=1)

    print(f'Total documents: {df.shape[0]}')

    if not args.persons:
        if args.stats:
            get_stats(args.target, df, args.stats)
        else:
            print('Nothing to do')
        exit()

    df = df[df['person'].isin([str(x) for x in args.persons])]
    print(f'Documents with selected {args.target}s: {df.shape[0]}')
    if not df.shape[0]:
        exit()

    # Filter by minimum variation count
    df = df[df['variations'].apply(lambda x: len(x.split()) >= args.min_vars)]

    # Filter by minimum token count
    df = df[df['text'].apply(lambda x: len(x.split()) >= args.min_tokens)]

    # Filter by treebanks (require annotations)
    if args.treebanks:
        df = df[df['annotated'].apply(lambda x: x == 1)]

    print(f'Length of df using filters (min_vars: {args.min_vars}, min_tokens: {args.min_tokens}, treebanks {args.treebanks}: {df.shape[0]}')
    
    if not df.shape[0]:
        exit()

    # Reset index
    df = df.reset_index(drop=True)

    # Print stats
    get_stats(args.target, df, args.stats)

    if not args.features:
        print('Please specify features. Implemented features are:')
        print(f'{",".join([t[0] for t in transformers()])}')
        exit()

    ct = get_column_transformer(args.features)
    X = ct.fit_transform(df)
    meanPoint = X.mean(axis = 0)

    # subtract mean point
    X -= meanPoint

    # Get clusters
    kmeans = KMeans(n_clusters=args.clusters, max_iter=100, n_init=5,random_state=0).fit(X)

    # Dimensionality reduction
    if args.algorithm == 'LSA':
        svd = TruncatedSVD(n_components=2, random_state=42)
        data = svd.fit_transform(X)

    elif args.algorithm == 'PCA':
        pca = PCA(random_state=42, n_components=2)
        data = pca.fit_transform(X)

    elif args.algorithm == 'TSNE':
        tsne = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=10)
        data = tsne.fit_transform(X) # TSNE
    
    # Dimensionality reduction results back to dataframe
    df2 = pd.DataFrame(data, columns = ['x', 'y'])
    df2['person'] = df['person']
    df2['text_id'] = df['text_id']
    df2['cluster'] = kmeans.labels_

    unique_persons = list(df.person.unique())

    def person_to_smallint(x):
        return unique_persons.index(x)

    df2['person_unique'] = df2['person'].map(person_to_smallint)

    # Plotting
    fig = plt.figure(figsize=(15,10))
    ax = fig.add_subplot(111)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    seen_persons = set()
    handles = []
    texts = []
    for _, row in df2.iterrows():
        ax.scatter(x=row['x'], y=row['y'], label=row['person'], marker=markers[row['cluster']], color=colors[row['person_unique']], s=50)
        texts.append(ax.text(row['x'], row['y'], ' '+row['text_id'], size='smaller'))

        if row['person'] not in seen_persons:
            seen_persons.add(row['person'])
            handles.append(mpatches.Patch(color=colors[row['person_unique']], label=row['person']))
    
    adjust_text(texts)

    plt.legend(handles=handles)
    plt.show()