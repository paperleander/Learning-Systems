# Connect4 Classifier Comparison

import numpy as np
import time
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

# Naive Bayes
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import BernoulliNB

# Other methods
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


x = []
y = []
encode_input = {'b': [0, 0],
                'x': [1, 0],
                'o': [0, 1]}
encode_output = {'win': 0,
                 'draw': 1,
                 'loss': 2}

print("Loading data...")
with open('data/connect-4.data') as f:
    for line in f.readlines():
        line_i = line.strip().split(',')
        mapped_line = [encode_input[i] if i in 'bxo' else encode_output[i] for i in line_i]
        x.append(mapped_line[:42])
        y.append(mapped_line[42])

x_tmp = np.array(x)
y_tmp = np.array(y)
x = x_tmp.reshape(x_tmp.shape[0], -1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)


models = {'Gaussian Naive Bayes': GaussianNB(),
          # 'Multinomial Naive Bayes': MultinomialNB(),
          # 'Complement Naive Bayes': ComplementNB(),
          # 'Bernoulli Naive Bayes': BernoulliNB(),
          'KNearestNeighbors': KNeighborsClassifier(n_neighbors=10),
          'DecisionTree': DecisionTreeClassifier(max_depth=20, min_samples_leaf=5, min_samples_split=10),
          'RandomForest': RandomForestClassifier(max_depth=30, n_estimators=800),
          'LogisticRegression': LogisticRegression(C=100, multi_class='auto'),
          'AdaBoostClassifier': AdaBoostClassifier(learning_rate=0.1, n_estimators=1024),
          # 'RadiusNeighbors': RadiusNeighborsClassifier(), NOT WORKING
          'SVC Linear': SVC(kernel="linear", C=1),
          'SVC RBF': SVC(kernel="rbf", C=1, gamma=2),
          }


def display_accuracy(estimator):
    # display_accuracy
    start = time.time()
    estimator.fit(x_train, y_train)
    y_pred = estimator.predict(x_test)
    total = x_test.shape[0]
    accuracy = (100 * ((y_test == y_pred).sum() / total))
    print("Accuracy = ", round(accuracy, 2), "%")
    print("Time: ", round(time.time() - start, 3), "s")
    print("")


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    axes : array of 3 axes, optional (default=None)
        Axes to use for plotting the curves.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    #if axes is None:
        #_, axes = plt.subplots(1, 1, figsize=(20, 5))

    fig, axes = plt.subplots()
    axes.set_title(title)
    if ylim is not None:
        axes.set_ylim(*ylim)
    axes.set_xlabel("Training examples")
    axes.set_ylabel("Score")

    train_sizes, train_scores, test_scores = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # Plot learning curve
    axes.grid()
    axes.fill_between(train_sizes, train_scores_mean - train_scores_std,
                      train_scores_mean + train_scores_std, alpha=0.1,
                      color="g")
    axes.plot(train_sizes, train_scores_mean, 'o-', color="g", label="Training score")

    # axes.fill_between(train_sizes, test_scores_mean - test_scores_std,
    # test_scores_mean + test_scores_std, alpha=0.1, color="g")
    # axes.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

    axes.legend(loc="best")

    plt.show()


if __name__ == "__main__":
    for name, classifier in models.items():
        print("Running:", name)

        # Accuracy
        display_accuracy(estimator=classifier)

        # Plot Curves
        cv = ShuffleSplit(n_splits=10, test_size=0.1, random_state=0)
        plot_learning_curve(estimator=classifier, title=name, X=x, y=y, ylim=(0.1, 1.01),
                            cv=cv, n_jobs=4)
