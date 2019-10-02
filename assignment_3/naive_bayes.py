from collections import defaultdict

import numpy as np


class NaiveBayes:
    def __init__(self):
        self.vocabulary = {}
        self.doc_to_class = defaultdict(list)

    def train(self, train_x, train_y):
        """
        BOKA:
        1. Collect all words that occur in examples
            Vocabulary <-- set of all distinct words in any document from examples
        2. Calculate P(Vj) and P(Wk|Vj) probability terms:
            for class in y:


        :param train_x: Words from each document to train on
        :param train_y: Class the document belongs to
        :return:
        """
        # BAO:
        # get number of training documents
        # create vocabulary of training set
        # zip data and labels to train
        # get set of all classes
        # create bow for all classes
        # for each class
        # for class in all_classes:
        #   docsj: the subset of documents from Examples for which the target value is vj
        #   compute prior for glass
        #   calculate the sum of counts of words in current class
        #   for every word, get the count and compute the likelihood for the class

        # MIN:
        # Connect data and labels together (x -> y)
        for x, y in zip(train_x, train_y):
            self.doc_to_class[y].append(x)



        # Vocabulary (set of distinct words from all documents)
        self.make_vocabulary(train_x)

        pass

    def make_vocabulary(self, train_x):
        for document_index, words_in_document in enumerate(train_x):
            words_in_document = np.asarray(words_in_document)
            unique_words, count_of_words = np.unique(words_in_document, return_counts=True)
            self.vocabulary[document_index] = {}
            for i, word in enumerate(unique_words):
                self.vocabulary[document_index][word] = count_of_words[i]
        # Is this working now? (flatten maybe?)

        pass

    def evaluate(self):
        pass
