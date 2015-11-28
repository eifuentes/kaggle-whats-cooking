import unicodedata
from collections import Counter
import pandas as pd
import numpy as np
from sklearn import preprocessing
from nltk.stem import WordNetLemmatizer


""" load what's cooking data """
def load_wc_data(path, encoder=None, shuffle=True, verbose=False):
    # read kaggle data json
    df = pd.read_json(path, orient='records')
    n_records = len(df)
    if verbose:
        print 'loaded %s records' % n_records
    # label encode cuisine
    labels = df['cuisine'].as_matrix()
    if not encoder:
        encoder = preprocessing.LabelEncoder()
        encoder.fit(labels)
    labels_encoded = encoder.transform(labels)
    target_df = pd.DataFrame({'cuisine_code': labels_encoded},
        index=df.index)
    # append encoded labels to original dataframe
    df = pd.concat([df, target_df], axis=1)
    if shuffle:
        df = df.iloc[np.random.permutation(n_records)]
    return df, encoder


def normalize_str(value):
    return unicodedata.normalize('NFKD', value.strip()).encode('ASCII', 'ignore').lower()


wnltz = WordNetLemmatizer()
def lemmatize_str(value):
    return wnltz.lemmatize(value)


def clean_recipe(recipe_ingrdnts):
    return [' '.join(lemmatize_str(normalize_str(i)) for i in ingrdnts.split(' ')) for ingrdnts in recipe_ingrdnts]


def clean_recipes(recipes, verbose=False):
    if verbose:
        print 'cleaning all %s recipes...' % len(recipes)
    for idx, recipe in enumerate(recipes):
        recipes[idx] = clean_recipe(recipe)
    if verbose:
        print 'finished cleaning'
    return recipes


# """ count ingredients in all recipes """
# def ingrdnts_idf(recipe_ingrdnts):
#     all_ingrdnts = []
#     for ingrdnts in recipe_ingrdnts:
#         all_ingrdnts += ingrdnts
#     counts = Counter(all_ingrdnts)
#     if tf_idf:
#         total_n_ingrdnts = float(len(all_ingrdnts))
#         for key in counts:
#             counts[key] = 1.0 - (counts[key] / total_n_ingrdnts)
#     return counts


def main():
    train_df, cuisine_encoder = load_wc_data('data/train.json')
    # wc_train_recipe_ingrdnts = train_df['ingredients']
    # wc_ingrdnts_counts = count_ingrdnts(wc_train_recipe_ingrdnts, tf_idf=True)

if __name__ == '__main__':
    main()
