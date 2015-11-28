"""
generates features from kaggle what's cooking competition's data
"""
import multiprocessing
from collections import Counter
import pandas as pd
import numpy as np
from sklearn import preprocessing
from gensim.models import Word2Vec


""" public variables """
random_state = None  # random state, none -> random
verbose = True  # print to console on/off


""" load what's cooking data """
def load_wc_data(path, encoder=None, shuffle=True):
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


""" build what's cooking gensim Word2Vec model """
def build_wc_model(recipe_ingrdnts, size=100, n_jobs=1):
    if verbose:
        print 'training Word2Vec model on %s recipes using %s cores...' % (len(recipe_ingrdnts), n_jobs)
    model = Word2Vec(recipe_ingrdnts, size=size, min_count=1, workers=n_jobs)
    return model


""" count ingredients in all recipes """
def count_ingrdnts(recipe_ingrdnts, tf_idf=False):
    all_ingrdnts = []
    for ingrdnts in recipe_ingrdnts:
        all_ingrdnts += ingrdnts
    counts = Counter(all_ingrdnts)
    if tf_idf:
        total_n_ingrdnts = float(len(all_ingrdnts))
        for key in counts:
            counts[key] = 1.0 - (counts[key] / total_n_ingrdnts)
    return counts


""" build what's cooking recipe vector representations aka feature matrix """
def build_wc_vec(recipe_ingrdnts, model, size, avg=True, tf=None):
    if verbose:
        print 'building feature vectors...'
    n_recipes = len(recipe_ingrdnts)
    features_matrix = np.zeros((n_recipes, size), np.float32)
    for idx_recipe, recipe in enumerate(recipe_ingrdnts):
        n_ingrdnts = 0
        for ingrdnt in recipe:
            try:
                temp_vec = model[ingrdnt]
                if tf:
                    features_matrix[idx_recipe, :] += (temp_vec * tf[ingrdnt])
                else:
                    features_matrix[idx_recipe, :] += temp_vec
                n_ingrdnts += 1
            except:
                print 'error trying to transform %s' % ingrdnt
                continue
        if avg:
            features_matrix[idx_recipe, :] /= float(n_ingrdnts)
    return features_matrix


""" generate what's cooking feature matrix & components """
def generate_wc_setup(feature_vec_size=200, avg=True):
    n_cores = multiprocessing.cpu_count()
    print '\nloading training data...'
    train_df, cuisine_encoder = load_wc_data('data/train.json')
    wc_train_recipe_ingrdnts = train_df['ingredients']
    print 'counting ingredient...'
    wc_ingrdnts_counts = count_ingrdnts(wc_train_recipe_ingrdnts, tf_idf=True)
    print 'building size %s vectors...' % feature_vec_size
    wc_features_model = build_wc_model(wc_train_recipe_ingrdnts,
        size=feature_vec_size, n_jobs=n_cores)
    wc_train_features = build_wc_vec(wc_train_recipe_ingrdnts,
        wc_features_model, feature_vec_size, avg=avg, tf=wc_ingrdnts_counts)
    return {
        'train': {
            'df': train_df,
            'features_matrix': wc_train_features,
        },
        'label_encoder': cuisine_encoder,
        'model': wc_features_model
    }


""" main entry method """
def main():
    wc_components = generate_wc_setup()
    print 'what\'s cooking features matrix of dim %s by %s' % wc_components['train']['features_matrix'].shape


if __name__ == '__main__':
    main()
