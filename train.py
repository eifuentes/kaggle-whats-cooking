"""
train supervised classifier with what's cooking recipe data
objective - determine recipe type categorical value from 20
"""
import time
from feature_gen import *
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.cross_validation import cross_val_score


""" main entry method """
def main(random_state=22, std=False):

    # load, train, build feature vectors
    wc_components = generate_wc_setup()

    # prepare training set
    y_train = wc_components['train']['df']['cuisine_code'].as_matrix()
    X_train = wc_components['train']['features_matrix']

    # standardize features aka mean ~ 0, std ~ 1
    if std:
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)

    # creating supervised classifier
    time_0 = time.time()

    n_estimators = 100
    print 'instantiate extra random forest with %s estimators' % n_estimators
    # ExtraTreesClassifier: 0.67275 (+/- 0.00465), cv 5-fold, 6.8 min
    clf = ExtraTreesClassifier(n_estimators=n_estimators, max_features=None,
        n_jobs=-1, random_state=random_state, verbose=1)

    # SGDClassifier: accuracy: 0.65286 (+/- 0.00780), cv 5-fold, log, l1, 1e-7, avg
    # clf =  SGDClassifier(loss='log', penalty='l1', n_iter=50, alpha=1e-7, average=True,
    #     shuffle=True, n_jobs=-1, verbose=0, random_state=random_state)

    # perform cross validation
    cv_n_fold = 4
    print 'cross validating %s ways...' % cv_n_fold
    scores_cv = cross_val_score(clf, X_train, y_train, cv=cv_n_fold, n_jobs=-1)
    print 'accuracy: %0.5f (+/- %0.5f)' % (scores_cv.mean(), scores_cv.std() * 2)

    time_1 = time.time()
    elapsed_time = time_1 - time_0
    print 'cross validation took %.3f seconds' % elapsed_time


if __name__ == '__main__':
    main()
