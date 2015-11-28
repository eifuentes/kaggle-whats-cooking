import time
from feature_gen import *
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, RMSprop, Adam
from keras.utils import np_utils


""" main entry method """
def main(random_state=None, std=False):

    # load, train, build feature vectors
    feature_vec_size = 200
    wc_components = generate_wc_setup(feature_vec_size=feature_vec_size, avg=True)
    n_categories = len(wc_components['label_encoder'].classes_)
    print n_categories
    # prepare training set
    y_train = wc_components['train']['df']['cuisine_code'].as_matrix()
    Y_train = np_utils.to_categorical(y_train, n_categories)
    X_train = wc_components['train']['features_matrix']

    # standardize features aka mean ~ 0, std ~ 1
    if std:
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)

    # creating supervised classifier
    time_0 = time.time()
    n_inputs = X_train.shape[1]
    n_hidden = [int(feature_vec_size * 1.5), int(feature_vec_size * 1.5)]
    X_train = X_train.astype("float32")
    print n_inputs
    print n_hidden
    model = Sequential()
    model.add(Dense(n_hidden[0], input_shape=(n_inputs,), activation='relu', init='glorot_uniform'))
    model.add(Dropout(0.5))
    model.add(Dense(n_hidden[1], activation='relu', init='glorot_uniform'))
    model.add(Dropout(0.5))
    model.add(Dense(n_categories, activation='softmax', init='glorot_uniform'))

    # sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    # rms = RMSprop()
    model.compile(loss='categorical_crossentropy', optimizer="adam")

    model.fit(X_train, Y_train, nb_epoch=20, batch_size=20, show_accuracy=True, verbose=2, validation_split=0.2)
    #score = model.evaluate(X_test, y_test, batch_size=16)

    time_1 = time.time()
    elapsed_time = time_1 - time_0
    print 'training took %.3f seconds' % elapsed_time

if __name__ == '__main__':
    main()
