import time
from features_bow import *
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.cross_validation import cross_val_score


def train_wc(cv_n_fold=8, random_state=None, n_jobs=1, verbose=0):

    # use term frequencey inverse document frequencey feature representation
    wc_components = build_tfidf_wc(verbose=verbose)
    # prepare training set
    y_train = wc_components['train']['df']['cuisine_code'].as_matrix()
    X_train = wc_components['train']['features_matrix']

    # create random forest supervised classifier
    time_0 = time.time()

    clf = RandomForestClassifier(n_estimators=100, max_depth=None,
        n_jobs=n_jobs, random_state=random_state, verbose=verbose)

    # perform cross validation
    print 'cross validating %s ways...' % cv_n_fold
    scores_cv = cross_val_score(clf, X_train, y_train, cv=cv_n_fold, n_jobs=n_jobs)
    print 'accuracy: %0.5f (+/- %0.5f)' % (scores_cv.mean(), scores_cv.std() * 2)

    time_1 = time.time()

    print 'cross validation took %.3f seconds' % (time_1 - time_0)


if __name__ == '__main__':
    train_wc(n_jobs=-1, verbose=2)
