""" generate word2vec based feature representations """
import multiprocessing
from collections import Counter
import pandas as pd
import numpy as np
from sklearn import preprocessing
from gensim.models import Word2Vec
from features_util import *


""" build what's cooking gensim Word2Vec model """
def build_word2vec_wc_model(recipe_ingrdnts, size=100, n_jobs=1, verbose=False):
    if verbose:
        print 'training Word2Vec model on %s recipes using %s cores...' % (len(recipe_ingrdnts), n_jobs)
    model = Word2Vec(recipe_ingrdnts, size=size, min_count=1, workers=n_jobs)
    return model


""" build what's cooking recipe vector representations aka feature matrix """
def build_word2vec_wc_recipes(recipe_ingrdnts, model, size, avg=True, idf=None, verbose=False):
    if verbose:
        print 'building feature vectors...'
    n_recipes = len(recipe_ingrdnts)
    features_matrix = np.zeros((n_recipes, size), np.float32)
    for idx_recipe, recipe in enumerate(recipe_ingrdnts):
        n_ingrdnts = 0
        for ingrdnt in recipe:
            try:
                temp_vec = model[ingrdnt]
                if idf:
                    if ingrdnt in idf.keys():
                        features_matrix[idx_recipe, :] += (temp_vec * idf[ingrdnt])
                        n_ingrdnts += 1
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
def build_word2vec_wc(feature_vec_size=120, avg=True, idf=None, verbose=False):
    n_cores = multiprocessing.cpu_count()
    print '\nloading training data...'
    train_df, cuisine_encoder = load_wc_data('data/train.json')
    wc_train_recipe_ingrdnts = clean_recipes(train_df['ingredients'].as_matrix(), verbose=verbose)
    if idf:
        print 'flattening recipes...'
        wc_train_recipe_ingrdnts = flatten_recipes(wc_train_recipe_ingrdnts)
    print 'building size %s vectors...' % feature_vec_size
    wc_features_model = build_word2vec_wc_model(wc_train_recipe_ingrdnts,
        size=feature_vec_size, n_jobs=n_cores, verbose=verbose)
    wc_train_features = build_word2vec_wc_recipes(wc_train_recipe_ingrdnts,
        wc_features_model, feature_vec_size, avg=avg, idf=idf, verbose=verbose)
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
    wc_components = build_word2vec_wc(verbose=True)
    print 'what\'s cooking features matrix of dim %s by %s' % wc_components['train']['features_matrix'].shape
    ingrdnt = 'soy sauce'
    print '\nMost Similar Words to %s' % ingrdnt
    similar_words = wc_components['model'].most_similar_cosmul(ingrdnt)
    for word in similar_words:
        print word


if __name__ == '__main__':
    main()
