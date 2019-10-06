#!/usr/bin/python

"""
NAIVE BAYES TEXT CLASSIFICATION
"""


from pre_process import gather_thy_data
from naive_bayes import NaiveBayes

for i in range(9, 10):
    # Gather text from newsgroups
    x_train, x_test, y_train, y_test = gather_thy_data(split=i/10)

    # Initialize the mainframe boyos
    nb = NaiveBayes()

    # Locomotive on words in each group
    nb.train(x_train, y_train)

    # Display accuracy
    nb.evaluate(x_test, y_test)
