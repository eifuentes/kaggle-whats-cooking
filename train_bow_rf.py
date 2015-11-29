""" use bag of words recipe representations to build a random forest classifier """
import time
from features_bow import *
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.cross_validation import cross_val_score


""" train random forest classifier with tf-idf recipe representations """
def train_wc(cv_n_fold=8, random_state=None, n_jobs=1, verbose=0):
    # use term frequencey inverse document frequencey feature representation
    wc_components = build_tfidf_wc(verbose=(verbose > 0))
    # prepare training set
    y_train = wc_components['train']['df']['cuisine_code'].as_matrix()
    X_train = wc_components['train']['features_matrix']
    # create random forest supervised classifier
    time_0 = time.time()
    clf = RandomForestClassifier(n_estimators=100, max_depth=None,
        n_jobs=n_jobs, random_state=random_state, verbose=verbose)
    if cv_n_fold > 0:
        # perform cross validation
        print 'cross validating %s ways...' % cv_n_fold
        scores_cv = cross_val_score(clf, X_train, y_train, cv=cv_n_fold, n_jobs=n_jobs)
        print 'accuracy: %0.5f (+/- %0.5f)' % (scores_cv.mean(), scores_cv.std() * 2)
    else:
        # fit random forest model
        print 'training random forest classifier...'
        clf.fit(X_train, y_train)
    time_1 = time.time()
    print 'process took %.3f seconds' % (time_1 - time_0)
    return wc_components, clf


""" apply transforms & classifier to what's cooking test set """
def test_wc(encoder, vectorizer, clf, verbose=False):
    wc_test_df = pd.read_json('data/test.json', orient='records')
    wc_test_recipe_ingrdnts = clean_recipes(wc_test_df['ingredients'].as_matrix(), verbose=verbose)
    wc_test_recipe_ingrdnts = build_recipe_docs(wc_test_recipe_ingrdnts)
    wc_test_features_bow = vectorizer.transform(wc_test_recipe_ingrdnts)
    wc_test_predict = clf.predict(wc_test_features_bow)
    wc_test_predict = encoder.inverse_transform(wc_test_predict)
    wc_test_predict_df = pd.DataFrame({'cuisine': wc_test_predict}, index=wc_test_df['id'])
    wc_test_predict_df.to_csv('data/submission_bow_rf.csv')


""" train & test bag of words random forest classifier """
def run(n_jobs=-1, verbose=1):
    components, clf = train_wc(cv_n_fold=0, n_jobs=n_jobs, verbose=verbose)
    test_wc(components['label_encoder'], components['model'], clf, verbose=(verbose > 0))


if __name__ == '__main__':
    run()
