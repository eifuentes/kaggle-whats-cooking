"""
generates features from kaggle what's cooking competition's data
"""
import multiprocessing
import pandas as pd
import numpy as np
from sklearn import preprocessing


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


def main():
    n_cores = multiprocessing.cpu_count()
    print '\nloading training data...'
    train_df, cuisine_encoder = load_wc_data('data/train.json')

if __name__ == '__main__':
    main()
