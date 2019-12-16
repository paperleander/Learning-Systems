# Connect4

import numpy as np
import time

from sklearn.model_selection import train_test_split

# Naive Bayes
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import BernoulliNB
# from sklearn.naive_bayes import CategoricalNB

# Other methods
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

x = []
y = []
encode_input = {'b': [1, 0, 0],
                'x': [0, 1, 0],
                'o': [0, 0, 1]}
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

x = np.array(x)
y = np.array(y)
x = x.reshape(x.shape[0], -1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

names = ["Gaussian Naive Bayes",
         "Multinomial Naive Bayes",
         "Complement Naive Bayes",
         "Bernoulli Naive Bayes",
         # "K Nearest Neighbors",
         "Decision Tree",
         "Random Forest",
         # "Neural Net",
         "Linear SVC",
         "RBF SVC"]

classifiers = [GaussianNB(),
               MultinomialNB(),
               ComplementNB(),
               BernoulliNB(),
               # KNeighborsClassifier(3),
               DecisionTreeClassifier(max_depth=5),
               RandomForestClassifier(max_depth=1000, n_estimators=10, max_features=10),
               # MLPClassifier(alpha=1, max_iter=1000),
               SVC(kernel="linear", C=0.025),
               SVC(gamma=2, C=1)]

for i, cay in enumerate(classifiers):
    start = time.time()
    print(names[i])

    cay.fit(x_train, y_train)
    y_pred = cay.predict(x_test)

    total = x_test.shape[0]
    accuracy = (100 * ((y_test == y_pred).sum() / total))
    print("Accuracy = ", round(accuracy, 2), "%")
    print("Time: ", round(time.time() - start, 3), "s")
    print("")
