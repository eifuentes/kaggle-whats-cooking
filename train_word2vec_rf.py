"""
train supervised classifier with what's cooking recipe data
objective - determine recipe type categorical value from 20
"""
import time
from features_bow import *
from features_word2vec import *
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.cross_validation import cross_val_score


""" main entry method """
def main(use_idf=False, random_state=None, std=False, n_jobs=-1, verbose=2):
    wc_idf_map = None
    if use_idf:
        # ingredients inverse document frequencies
        wc_components = build_tfidf_wc(verbose=(verbose > 0))
        wc_idf = wc_components['model'].idf_
        wc_idf_words = wc_components['model'].get_feature_names()
        wc_idf_map = dict(zip(wc_idf_words, wc_idf))
    # word2vec recipe feature vectors
    wc_components = build_word2vec_wc(feature_vec_size=120, avg=True, idf=wc_idf_map, verbose=(verbose > 0))
    y_train = wc_components['train']['df']['cuisine_code'].as_matrix()
    X_train = wc_components['train']['features_matrix']
    # standardize features aka mean ~ 0, std ~ 1
    if std:
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
    # random forest supervised classifier
    time_0 = time.time()
    clf = RandomForestClassifier(n_estimators=100, max_depth=None,
        n_jobs=n_jobs, random_state=random_state, verbose=verbose)
    # perform cross validation
    cv_n_fold = 8
    print 'cross validating %s ways...' % cv_n_fold
    scores_cv = cross_val_score(clf, X_train, y_train, cv=cv_n_fold, n_jobs=-1)
    print 'accuracy: %0.5f (+/- %0.5f)' % (scores_cv.mean(), scores_cv.std() * 2)
    time_1 = time.time()
    elapsed_time = time_1 - time_0
    print 'cross validation took %.3f seconds' % elapsed_time


if __name__ == '__main__':
    main()
