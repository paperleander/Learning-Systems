# Connect4

import numpy as np
import time

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

# Naive Bayes
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import BernoulliNB
# from sklearn.naive_bayes import CategoricalNB

# Other methods
from sklearn.neighbors import KNeighborsClassifier

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

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


bayeser = [GaussianNB(), MultinomialNB(), ComplementNB(), BernoulliNB(), KNeighborsClassifier(3)]

for bay in bayeser:
    start = time.time()
    print(bay)
    bay.fit(x_train, y_train)
    y_pred = bay.predict(x_test)

    total = x_test.shape[0]
    accuracy = (100 * ((y_test == y_pred).sum() / total))
    print("Accuracy = ", round(accuracy, 3), "%")
    print("Time: ", time.time() - start)
    print("")
