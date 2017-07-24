from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
import numpy as np
#from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt
import pickle
import itertools
import sift


def histogram_intersection(M, N):
    m = M.shape[0]
    print(m)
    n = N.shape[0]
    print(n)
    print(M.T.shape)

    result = np.zeros((m,n))
    for i in range(m):
        print(i)
        for j in range(n):

            temp = np.sum(np.minimum(np.array(M)[i], np.array(N)[j]))
            result[i][j] = temp

    return result


def classify_hi(train_df, test_df):
    print('loading pickles')
    train_df = sift.pickle_load(train_df)
    test_df = sift.pickle_load(test_df)
    print('running kernel...')
    matrix = histogram_intersection(train_df.T.iloc[:, : -1], train_df.T.iloc[:, : -1])
    print('fitting svc...')
    clf = SVC(kernel='precomputed')
    clf.fit(matrix, train_df.T.ix[:, -1])
    print('predict matrix...')
    predict_matrix = histogram_intersection(test_df.T.iloc[:, : -1], train_df.T.iloc[:, : -1])
    print('predicting results...')
    SVMResults = clf.predict(predict_matrix)
    print('calculating')
    correct = sum(1.0 * (SVMResults == test_df.T['y']))
    accuracy = correct / len(test_df.T['y'])
    print("SVM (Histogram Intersection): " + str(accuracy) + " (" + str(int(correct)) + "/" + str(len(test_df.T['y'])) + ")")

    #print('score: ' + str(clf.score(test_df.T.ix[:, test_df.T.columns != 'y'], test_df.T['y'])))
    # results = clf.predict(predict_matrix)
    # correct = sum(1.0 * (results == test_df.T['y']))
    # accuracy = correct / len(test_df.T['y'])

    # cnf_matrix = confusion_matrix(test_df.T['y'], clf.predict(test_df.T.ix[:, test_df.T.columns != 'y']))
    # np.set_printoptions(precision=2)
    #
    # plt.figure()
    # plot_confusion_matrix(cnf_matrix, classes=['ant', 'bee', 'butterfly', 'centipede', 'dragonfly', 'ladybug', 'tick', 'beetle',
    #                                        'termite', 'worm'],
    #                       normalize=True,
    #                       title='Normalized confusion matrix')
    #
    # plt.show()



def classify_images_svc(train_df, test_df):
    param_grid = {'C': [1.0, 5.0],
                  'degree': [2, 3],
                  'kernel': ['rbf']}

    clf = GridSearchCV(SVC(), param_grid=param_grid)
    with open(train_df, 'rb') as train:
        train_df = pickle.load(train)
        with open(test_df, 'rb') as test:
            test_df = pickle.load(test)
            clf.fit(train_df.T.ix[:, : -1], train_df.T.ix[:, -1])
            print(clf.score(test_df.T.ix[:, : -1], test_df.T.ix[:, -1]))

            # Compute confusion matrix
            cnf_matrix = confusion_matrix(test_df.T['y'], clf.predict(test_df.T.ix[:, test_df.T.columns != 'y']))
            np.set_printoptions(precision=2)

            # Plot normalized confusion matrix
            plt.figure(figsize=(10, 10))
            plot_confusion_matrix(cnf_matrix,
                                  classes=['ant', 'bee', 'butterfly', 'centipede', 'dragonfly', 'ladybug', 'tick',
                                           'beetle', 'termite', 'worm'],
                                  normalize=True,
                                  title='Normalized confusion matrix')

            plt.show()


def classify_images_rf(train_df, test_df):
    clf = RandomForestClassifier()
    with open(train_df, 'rb') as train:
        train_df = pickle.load(train)
        with open(test_df, 'rb') as test:
            test_df = pickle.load(test)
            clf.fit(train_df.T.ix[:, train_df.T.columns != 'y'], train_df.T['y'])
            print(clf.score(test_df.T.ix[:, test_df.T.columns != 'y'], test_df.T['y']))


def classify_images_one_v_all(train_df, test_df):
    param_grid = {'C': [0.1, 0.5, 1.0, 5., 10.]}
    clf = OneVsRestClassifier(GridSearchCV(LinearSVC(), param_grid=param_grid))
    with open(train_df, 'rb') as train:
        train_df = pickle.load(train)
        with open(test_df, 'rb') as test:
            test_df = pickle.load(test)
            clf.fit(train_df.T.ix[:, train_df.T.columns != 'y'], train_df.T['y'])

            print('score: ' + str(clf.score(test_df.T.ix[:, test_df.T.columns != 'y'], test_df.T['y'])))

            # Compute confusion matrix
            cnf_matrix = confusion_matrix(test_df.T['y'], clf.predict(test_df.T.ix[:, test_df.T.columns != 'y']))
            np.set_printoptions(precision=2)

            # Plot normalized confusion matrix
            plt.figure(figsize=(10,10))
            plot_confusion_matrix(cnf_matrix,
                                  classes=['ant', 'bee', 'butterfly', 'centipede', 'dragonfly', 'ladybug', 'tick', 'beetle',
                                           'termite', 'worm'],
                                  normalize=True,
                                  title='Normalized confusion matrix')

            plt.show()


def classify_images_sgd(train_df, test_df):
    clf = SGDClassifier()
    with open(train_df, 'rb') as train:
        train_df = pickle.load(train)
        with open(test_df, 'rb') as test:
            test_df = pickle.load(test)
            clf.fit(train_df.T.ix[:, train_df.T.columns != 'y'], train_df.T['y'])
            print(clf.score(test_df.T.ix[:, test_df.T.columns != 'y'], test_df.T['y']))


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, "{:.2}".format(cm[i, j]),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
