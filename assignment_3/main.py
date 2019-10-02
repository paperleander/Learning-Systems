#!/usr/bin/python

"""
NAIVE BAYES TEXT CLASSIFICATION
"""


from pre_process import gather_thy_data
from pre_process import process

from naive_bayes import NaiveBayes


# Gather text from newsgroups
x_train, x_test, y_train, y_test = gather_thy_data()

# Initialize the mainframe boyos
nb = NaiveBayes()

nb.train(x_train, y_train)
nb.evaluate()
