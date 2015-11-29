from features_util import *
from sklearn.feature_extraction.text import TfidfVectorizer


def build_tfidf_wc(verbose=False):
    train_df, cuisine_encoder = load_wc_data('data/train.json', verbose=verbose)
    wc_train_recipe_ingrdnts = clean_recipes(train_df['ingredients'].as_matrix(), verbose=verbose)
    wc_train_recipe_ingrdnts = build_recipe_docs(wc_train_recipe_ingrdnts)
    vectorizer = TfidfVectorizer(stop_words='english', sublinear_tf=True, use_idf=True)
    wc_train_features_bow = vectorizer.fit_transform(wc_train_recipe_ingrdnts)
    if verbose:
        print 'tf-idf fit %s features' % wc_train_features_bow.shape[1]
    return {
        'train': {
            'df': train_df,
            'features_matrix': wc_train_features_bow,
        },
        'label_encoder': cuisine_encoder,
        'model': vectorizer
    }


def main(verbose=True):
    wc_components = build_tfidf_wc(verbose=verbose)
    print 'what\'s cooking features matrix of dim %s by %s' % wc_components['train']['features_matrix'].shape


if __name__ == '__main__':
    main()
