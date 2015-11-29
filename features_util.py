""" common feature generation utility functions """
import unicodedata
import itertools
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


""" normalize ingredient string """
def normalize_str(value):
    return unicodedata.normalize('NFKD', value.strip()).encode('ASCII', 'ignore').lower()


""" lemmatize ingredient string """
wnltz = WordNetLemmatizer()
def lemmatize_str(value):
    return wnltz.lemmatize(value)


""" normalize & lemmentize set of ingredients in a recipe """
def clean_recipe(recipe_ingrdnts):
    return [' '.join(lemmatize_str(normalize_str(i)) for i in ingrdnts.split(' ')) for ingrdnts in recipe_ingrdnts]


""" normalize & lemmentize all recipes """
def clean_recipes(recipes, verbose=False):
    if verbose:
        print 'cleaning all %s recipes...' % len(recipes)
    for idx, recipe in enumerate(recipes):
        recipes[idx] = clean_recipe(recipe)
    if verbose:
        print 'finished cleaning'
    return recipes


""" combine recipe ingredients into a single string aka doc """
def build_recipe_docs(recipes):
    for idx, recipe in enumerate(recipes):
        recipes[idx] = ' '.join(recipe)
    return recipes

""" breakdown each recipe into single word ingredients """
def flatten_recipes(recipes):
    for idx, recipe in enumerate(recipes):
        single_ingrdnts = [ingrdnts.split(' ') for ingrdnts in recipe]
        flat_ingrdnts = itertools.chain(*single_ingrdnts)
        recipes[idx] = list(flat_ingrdnts)
    return recipes


""" main entry method """
def main(verbose=True):
    train_df, cuisine_encoder = load_wc_data('data/train.json', verbose=verbose)
    wc_train_recipe_ingrdnts = clean_recipes(train_df['ingredients'].as_matrix(), verbose=verbose)


if __name__ == '__main__':
    main()
