#!/usr/bin/python

"""
NAIVE BAYES TEXT CLASSIFICATION
"""
import os
from sklearn.model_selection import train_test_split


MAIN_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join("data/20_newsgroups")

###################
# GATHER THY DATA #
###################

# List of folders (20)
folders = [f for f in os.listdir(DATA_DIR)]

# list of all files (19997)
files = []
for folder in folders:
    files.append([f for f in os.listdir(os.path.join(DATA_DIR, folder))])

# DEBUG
print("Total number of folders:", len(folders))
print("Total number of files:", sum(len(files[i]) for i in range(len(folders))))

# List of pathname for all files with corresponding class
list_of_file_pathnames = []
list_of_groups = []
for i, folder in enumerate(folders):
    for file in files[i]:
        list_of_file_pathnames.append(os.path.join(DATA_DIR, os.path.join(folders[i], file)))
        list_of_groups.append(folder)

# Split training and test
x_train, x_test, y_train, y_test = train_test_split(list_of_file_pathnames, list_of_groups, test_size=0.3)
