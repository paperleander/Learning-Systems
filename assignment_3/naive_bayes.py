from collections import defaultdict
import math
import time

from pre_process import pre_process


class NaiveBayes:
    def __init__(self):
        self.vocabulary = set()
        self.posts = defaultdict(list)

        self.p_word_given_group = {}
        self.p_group = {}

    def train(self, train_x, train_y):
        """
        Well, it "trains" on what words are in each group,
        so that it can guess on a group given only words..?
        :param train_x: Words from each document to train on
        :param train_y: Class the document belongs to
        :return:
        """
        # Timekeeping
        print("Start Training.")
        start_time = time.time()

        # Connect data and labels together (x -> y)
        for x, y in zip(train_x, train_y):
            # print("x", x)
            words = pre_process(x)
            for word in words:
                self.posts[y].append(word)
                self.vocabulary.add(word)

        # Calculate P(Hj) and P(Wk|Hj)
        for group in self.posts.keys():
            self.p_word_given_group[group] = {}
            docs_in_group = self.posts[group]
            self.p_group[group] = len(docs_in_group) / len(train_x)

            # Count number of words
            for word in self.vocabulary:
                self.p_word_given_group[group][word] = 1.0

            for word in self.posts[group]:
                if word in self.vocabulary:
                    self.p_word_given_group[group][word] += 1.0

            for word in self.vocabulary:
                self.p_word_given_group[group][word] /= len(self.posts[group]) + len(self.vocabulary)

        # Timekeeping
        timed = int(time.time() - start_time)
        print("Training finished in ", timed, "seconds.")

    def evaluate(self, test_x, test_y):
        # Timekeeping
        print("Start Evaluating.")
        start_time = time.time()

        correct = 0
        for x, y in zip(test_x, test_y):
            max_group = ""
            max_p = 1
            x_words = pre_process(x)

            for candidate_group in self.posts.keys():
                # P(O|H) * P(H) for each candidate group
                p = math.log(self.p_group[candidate_group])
                for word in x_words:
                    if word in self.vocabulary:
                        p += math.log(self.p_word_given_group[candidate_group][word])

                if p > max_p or max_p == 1:
                    max_p = p
                    max_group = candidate_group

            if max_group == y:
                correct += 1

        # Timekeeping
        timed = int(time.time() - start_time)
        print("Evaluation finished in ", timed, "seconds.")

        # Accuracy
        accuracy = (correct/len(test_y)) * 100
        print("Accuracy:", accuracy)
