"""
Functions to process newsgroup documents, including:
-Converting all letters to lower case
-Removing numbers
-Removing symbols (punctuation etc)
-Removing metadata?
-Removing whitespace
-Removing stopwords

source: https://medium.com/@datamonsters/text-preprocessing-in-python-steps-tools-and-examples-bf025f872908
"""

# TODO: remove tabs

import os
import re
import string

from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

MAIN_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join("data/20_newsgroups_stripped")


def gather_thy_data():
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
    return train_test_split(list_of_file_pathnames, list_of_groups, test_size=0.3)


def split_line_to_words(line):
    return line[0:len(line) - 1].strip().split(" ")


def remove_metadata(lines):
    index = 0
    for line in range(len(lines)):
        if lines[line] == '\n':
            index = line+1
            break
    lines = lines[index:]
    return lines


def remove_symbols(words):
    words = [word.translate((str.maketrans('', '', string.punctuation))) for word in words]
    words = [word.translate((str.maketrans('', '', '\t'))) for word in words]  # remove tabs
    return list(filter(None, words))  # Remove empty items in list


def remove_numbers(words):
    return [re.sub(r'\d+', '', word) for word in words]


def convert_to_lowercase(words):
    return [word.lower() for word in words]


def remove_stopwords(words):
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if w not in stop_words]
    return [str for str in words if str]  # do not touch. it works. exclude empty words


def remove_trash(words):
    # Exclude first and last character.
    # also remove words with only 1 character.
    less_trashy = []
    for word in words:
        if word[0] and word[-1] == "'":
            word = word[1:-1]
        elif word[0] == "'":
            word = word[1:]
        less_trashy.append(word)
    return [w for w in less_trashy if not len(w) == 1]


def pre_process(file):
    with open(file, "r") as f:
        lines = f.readlines()

    lines = remove_metadata(lines)

    words_in_doc = []
    for line in lines:
        words = split_line_to_words(line)
        words = remove_symbols(words)
        words = remove_numbers(words)
        words = convert_to_lowercase(words)
        words = remove_stopwords(words)
        words = remove_trash(words)
        if len(words) > 0:
            words_in_doc.append(words)

    return words_in_doc


def process(documents):
    words = []
    for doc in documents:
        words.append(pre_process(doc))

    # DEBUG
    print(">>>>>>>>>>>>>>>")
    for w in words:
        print(w)
    print(">>>>>>>>>>>>>>>")
