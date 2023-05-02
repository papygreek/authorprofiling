import pandas as pd
import numpy as np
import regex as re
from pathlib import Path

from natsort import index_natsorted

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from adjustText import adjust_text

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import cross_val_score, cross_val_predict, cross_validate, KFold, StratifiedKFold, train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.utils import resample

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
        # Character bigrams
        (
            'character_bigrams_bow',
            CountVectorizer(analyzer='char', ngram_range=(2,2)),
            'text',
        ),
        (
            'character_bigrams_tfidf',
            TfidfVectorizer(analyzer='char', ngram_range=(2,2)),
            'text',
        ),
        # Character trigrams
        (
            'character_trigrams_bow',
            CountVectorizer(analyzer='char', ngram_range=(3,3)),
            'text',
        ),
        (
            'character_trigrams_tfidf',
            TfidfVectorizer(analyzer='char', ngram_range=(3,3)),
            'text',
        ),
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
        (
            'lemmas_bigrams_tfidf',
            TfidfVectorizer(ngram_range=(2,2)),
            'lemmas',
        ),
        (
            'lemmas_trigrams_tfidf',
            TfidfVectorizer(ngram_range=(3,3)),
            'lemmas',
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
        (
            'postags_bigrams_tfidf',
            TfidfVectorizer(ngram_range=(2,2)),
            'postags',
        ),
        (
            'postags_trigrams_tfidf',
            TfidfVectorizer(ngram_range=(3,3)),
            'postags',
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
    return grave_to_acute(" ".join(x.split())).lower()

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

def run(args, _):

    # Require stats or features
    if not (args.stats or args.features):
        print('Available features:')
        for t in transformers():
            print(t[0])
        exit()  

    target_files = {
        'writer': 'writer_texts.csv',
        'author': 'author_texts.csv'
    }

    target_dimensions = 2 if args.plot == '2d' else 3

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

    print(f'Documents in complete dataset (without filters): {df.shape[0]}')

    for k, v in vars(args).items():
        if v:
            if k == 'persons':
                df = df[df['person'].isin([str(x) for x in args.persons])]
            elif k == 'text_name_contains':
                df = df[df['text_id'].str.contains(args.text_name_contains)]
            elif k == 'text_name_not_contains':
                df = df[~df['text_id'].str.contains(args.text_name_not_contains)]
            elif k == 'min_vars':
                df = df[df['variations'].apply(lambda x: len(x.split()) >= args.min_vars)]
            elif k == 'min_tokens':
                df = df[df['text'].apply(lambda x: len(x.split()) >= args.min_tokens)]
            elif k == 'treebanks':
                df = df[df['annotated'].apply(lambda x: x == 1)]
            
        if df.empty:
            print('No documents with these filters.')
            exit()

    # Reset index
    df = df.reset_index(drop=True)

    # Print stats
    print(df)
    get_stats(args.target, df, args.stats)
    print(f'Total filtered documents: {df.shape[0]}')
    
    if args.stats:
        exit()
    
    # Start transforming
    ct = get_column_transformer(args.features)
    X = ct.fit_transform(df)

    # Centering
    meanPoint = X.mean(axis = 0)
    X -= meanPoint

    # Make sure X is np array
    X = np.asarray(X)

    # Print n. of features
    print(f'Features: {X.shape}')

    # Get unique persons
    unique_persons = list(df.person.unique())

    if len(unique_persons) > 10:
        print(f'Too many persons ({len(unique_persons)}). Maximum is 10.')
        exit()

    if args.mode == 'classification':


        clf = make_pipeline(ct, svm.SVC(kernel='linear', probability=True))

        # Make cross validator
        cv = StratifiedKFold(n_splits=3, shuffle=True)

        scoring = ('precision_macro', 'recall_macro', 'accuracy', 'f1_macro')

        scores = cross_validate(clf, df, df["person"], cv=cv, scoring=scoring)
        print('Cross validation results (folds = 3): \n --------------------')
        print("%0.2f mean accuracy with a standard deviation of %0.2f" % (scores["test_accuracy"].mean(), scores["test_accuracy"].std()))
        print()
        # Get predictions
        pred_y = cross_val_predict(clf, df, df["person"], cv=cv) 
        print(classification_report(df["person"], pred_y, target_names=unique_persons))

        # Get confusion matrix
        cm = confusion_matrix(df["person"], pred_y)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=unique_persons
        
        )
        disp.plot(cmap=plt.cm.Blues)
        plt.show()

    elif args.mode == 'clustering':
        # Get n clusters
        n_clusters = len(list(set([x.replace('?', '') for x in unique_persons]))) \
            if not args.clusters \
            else args.clusters
        
        # Get kmeans clusters
        kmeans = KMeans(n_clusters=n_clusters, max_iter=500, n_init=10,random_state=0).fit(X)

        # LSA dimensionality reduction
        svd = TruncatedSVD(n_components=target_dimensions, random_state=42)
        data = svd.fit_transform(X)
        
        # Dimensionality reduction results back to dataframe
        df2 = pd.DataFrame(data, columns = ['x', 'y']) \
            if target_dimensions == 2 \
            else pd.DataFrame(data, columns = ['x', 'y', 'z'])
        
        df2['person'] = df['person']
        df2['text_id'] = df['text_id']
        df2['cluster'] = kmeans.labels_

        # Get indexes of unique persons [0,1,2,3...]
        df2['person_unique'] = df2['person'].map(lambda x: unique_persons.index(x))

        # Plotting
        plt.rcParams['savefig.dpi']=300
        fig = plt.figure(figsize=(15,10))
        ax = fig.add_subplot(111, projection=None if target_dimensions == 2 else '3d')
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

        seen_persons = set()
        handles = []
        texts = []
        for _, row in df2.iterrows():
            scatter_config = {
                'label': row['person'],
                'marker': markers[row['cluster']],
                'color': colors[row['person_unique']],
                's': 100 if target_dimensions == 2 else 50
            }
            ax_text = [row['x'], row['y'], f" {row['text_id']} "]

            if target_dimensions == 3:
                scatter_config['zs'] = row['z']
                scatter_config['xs'] = row['x'],
                scatter_config['ys'] = row['y'],
                ax_text.insert(2, row['z'])
            elif target_dimensions == 2:
                scatter_config['x'] = row['x'],
                scatter_config['y'] = row['y'],
            
            ax.scatter(**scatter_config)
            texts.append(ax.text(*ax_text, size='smaller' if target_dimensions == 3 else 'medium'))

            if row['person'] not in seen_persons:
                seen_persons.add(row['person'])
                handles.append(mpatches.Patch(color=colors[row['person_unique']], label=row['person']))
        
        if target_dimensions == 2:
            if args.arrows:
                adjust_text(texts, arrowprops=dict(arrowstyle="-", color='k', lw=0.5))
            else:
                adjust_text(texts)
        else:
            # Drop lines
            zs_l = np.asarray([[i, -1] for i in df2.z])

            for i, _ in enumerate(zs_l):
                ax.plot(xs=[df2.x[i]]*2, ys=[df2.y[i]]*2, zs=zs_l[i], color="lightgrey")

        plt.legend(handles=handles)
        plt.show()