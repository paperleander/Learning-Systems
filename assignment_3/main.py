#!/usr/bin/python

"""
NAIVE BAYES TEXT CLASSIFICATION
"""


from pre_process import gather_thy_data
from pre_process import process

# Gather text from newsgroups
x_train, x_test, y_train, y_test = gather_thy_data()

process(x_train)
