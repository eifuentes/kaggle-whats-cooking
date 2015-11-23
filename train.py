"""
train supervised classifier with what's cooking recipe data
objective - determine recipe type categorical value from 20
"""
from feature_gen import *
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.cross_validation import cross_val_score


""" main entry method """
def main(random_state=22):
    # load, train, build feature vectors
    wc_components = generate_wc_setup()
    # prepare training set
    y_train = wc_components['train']['df']['cuisine_code'].as_matrix()
    X_train = wc_components['train']['features_matrix']
    # supervised classifier
    n_estimators = 500
    print 'instantiate extra random forest with %s estimators' % n_estimators
    # ExtraTreesClassifier: 0.67275 (+/- 0.00465), cv 5-fold, 6.8 min
    clf = ExtraTreesClassifier(n_estimators=n_estimators, max_features=None,
        n_jobs=-1, random_state=random_state, verbose=2)
    # cross validation
    cv_n_fold = 5
    print 'cross validating %s ways...' % cv_n_fold
    scores_cv = cross_val_score(clf, X_train, y_train, cv=cv_n_fold, n_jobs=-1)
    print 'accuracy: %0.5f (+/- %0.5f)' % (scores_cv.mean(), scores_cv.std() * 2)


if __name__ == '__main__':
    main()
