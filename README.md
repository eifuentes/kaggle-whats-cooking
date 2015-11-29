# Kaggle "What's Cooking" Competition Project

Given a list of recipes and their ingredients, the objective of this competition
is to predict the type of cuisine category, e.g. Italian or Mexican, for each recipe.

The training set found in the *data* directory consists of
- recipe id
- list of ingredients
- cuisine category

To translate the raw ingredients list into a feature vector several
Natural Language Processing (NLP) methods can be used. For this project
gensim's Word2Vec and scikit-learn's tf-idf vectorizer implementations were used.

After each recipe was converted to a feature vector representing the raw ingredients,
training a variety of supervised learning classifiers becomes available, such as
Random Forests and Neural Networks.
