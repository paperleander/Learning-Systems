# Connect4 Classifier Comparison
from collections import defaultdict
from time import time
import json

import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

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
          'Multinomial Naive Bayes': MultinomialNB(),
          'Complement Naive Bayes': ComplementNB(),
          'Bernoulli Naive Bayes': BernoulliNB(),
          'DecisionTree': DecisionTreeClassifier(max_depth=20, min_samples_leaf=5, min_samples_split=10),
          'LogisticRegression': LogisticRegression(C=100, multi_class='auto'),
          'KNearestNeighbors': KNeighborsClassifier(n_neighbors=10),
          'RandomForest': RandomForestClassifier(max_depth=30, n_estimators=800),
          # REDACTED BELOW
          # 'AdaBoostClassifier': AdaBoostClassifier(learning_rate=0.1, n_estimators=1024),  #
          # 'RadiusNeighbors': RadiusNeighborsClassifier(), NOT WORKING
          # 'SVC Linear': SVC(kernel="linear", C=1),
          # 'SVC RBF': SVC(kernel="rbf", C=1, gamma=2),
          }


def get_accuracy(estimator):
    # display_accuracy
    start = time.time()
    estimator.fit(x_train, y_train)
    y_pred = estimator.predict(x_test)
    total = x_test.shape[0]
    accuracy = (100 * ((y_test == y_pred).sum() / total))
    time_ = round(time.time() - start, 3)
    print("Accuracy = ", round(accuracy, 2), "%")
    print("Time: ", time_, "s")
    print("")
    return round(accuracy, 2), time_


def plot_accuracy(classifiers_, accuracy):
    plt.figure(figsize=(10, 4))
    ax = sns.barplot(x=classifiers_, y=accuracy, palette='Blues_d')
    ax.set(xlabel='Classifiers', ylabel='Accuracy')
    ax.set_title("Accuracy of each Classifier")
    plt.show()


def plot_times(classifiers_, times_):
    plt.figure(figsize=(10, 4))
    ax = sns.barplot(x=classifiers_, y=times_, palette='Blues_d')
    ax.set(xlabel='Classifiers', ylabel='Time')
    ax.set_title("Time of each Classifier")
    plt.show()


def plot_all_learning_curves_in_one(curves):
    fig, axes = plt.subplots()
    axes.set_title("Accuracy for each Classifier over Training Examples")

    # set ylim?
    axes.set_xlabel("Training examples")
    axes.set_ylabel("Score")

    # Plot learning curve
    axes.grid()
    print(curves)
    for name_ in curves.keys():
        print(name_)
        axes.plot(curves[name_]['train_sizes'], curves[name_]['test_scores'], '-', label=name_)

    # axes.fill_between(train_sizes, test_scores_mean - test_scores_std,
    # test_scores_mean + test_scores_std, alpha=0.1, color="g")
    # axes.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

    axes.legend(loc="best")

    plt.show()


def plot_learning_curve(estimator, title, X, y, ylim=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
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

    cv = ShuffleSplit(n_splits=1, test_size=0.1, random_state=0)

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
    axes.plot(train_sizes, train_scores_mean, '-', color="g", label="Training score")

    # axes.fill_between(train_sizes, test_scores_mean - test_scores_std,
    # test_scores_mean + test_scores_std, alpha=0.1, color="g")
    # axes.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

    axes.legend(loc="best")

    plt.show()
    return train_sizes, train_scores, test_scores


if __name__ == "__main__":
    classifiers = []
    accuracies = []
    times = []

    results = defaultdict(list)
    curves = defaultdict()

    for name, classifier in models.items():
        print("Running:", name)

        acc, time_ = get_accuracy(estimator=classifier)
        train_sizes, train_scores, test_scores = plot_learning_curve(estimator=classifier, title=name, X=x, y=y, ylim=(0.1, 1.01), n_jobs=4)
        curves[name] = {
            "train_sizes": train_sizes.tolist(),
            "train_scores": train_scores.tolist(),
            "test_scores": test_scores.tolist()
        }

        results['classifiers'].append(name)
        results['accuracies'].append(acc)
        results['times'].append(time_)

    # plot_accuracy(results['classifiers'], results['accuracies'])
    # plot_times(results['classifiers'], results['times'])
    plot_all_learning_curves_in_one(curves)

    outfile_path = f"results_{time.time()}.json"
    with open(outfile_path, 'w') as fp:
        fp.write(json.dumps(dict(results)))

    outfile_path2 = f"curves_{time.time()}.json"
    with open(outfile_path2, 'w') as fp:
        fp.write(json.dumps(dict(curves)))
