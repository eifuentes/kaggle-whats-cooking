from features_util import *


def main():
    verbose = True
    train_df, cuisine_encoder = load_wc_data('data/train.json', verbose=verbose)
    wc_train_recipe_ingrdnts = train_df['ingredients'].as_matrix()
    wc_train_recipe_ingrdnts = clean_recipes(wc_train_recipe_ingrdnts, verbose=verbose)


if __name__ == '__main__':
    main()
