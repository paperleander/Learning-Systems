

# Leander Berg Thorkildsen
# 9. September 2019
# Assignment 2


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import random
import sys


def generate_data(operator="AND", n_bits=2, n_training_sets=5000, n_test_sets=5000, debug=False):
    # print("Generating data...")

    AND_Training_arrays = []
    OR_Training_arrays = []
    XOR_Training_arrays = []

    AND_Test_arrays = []
    OR_Test_arrays = []
    XOR_Test_arrays = []

    for _ in range(n_training_sets):
        # Make a random array of integers (example 8 bits -> [0, 1, 0, 0, 1, 1, 0, 1]
        arr = np.random.randint(2, size=n_bits)

        # AND logic adds a new integer at end of array
        if arr[0] == 1 and arr[1] == 1:
            new_AND_arr = np.append(arr, [1])
        else:
            new_AND_arr = np.append(arr, [0])
        AND_Training_arrays.append(new_AND_arr)

        # OR logic adds a new integer at end of array
        if arr[0] == 1 or arr[1] == 1:
            new_OR_arr = np.append(arr, [1])
        else:
            new_OR_arr = np.append(arr, [0])
        OR_Training_arrays.append(new_OR_arr)

        # XOR logic adds a new integer at end of array
        if not (arr[0] == arr[1]):
            new_XOR_arr = np.append(arr, [1])
        else:
            new_XOR_arr = np.append(arr, [0])
        XOR_Training_arrays.append(new_XOR_arr)

        # DEBUG
        if debug:
            print("{:-^19}".format("DEBUG"))
            print(new_AND_arr)
            print(new_OR_arr)
            print(new_XOR_arr)

    for _ in range(n_test_sets):
        # Make a random array of integers (example -> [0, 1, 0, 0, 1, 1, 0, 1]
        arr = np.random.randint(2, size=n_bits)

        # AND logic adds a new integer at end of array
        if arr[0] == 1 and arr[1] == 1:
            new_AND_arr = np.append(arr, [1])
        else:
            new_AND_arr = np.append(arr, [0])
        AND_Test_arrays.append(new_AND_arr)

        # OR logic adds a new integer at end of array
        if arr[0] == 1 or arr[1] == 1:
            new_OR_arr = np.append(arr, [1])
        else:
            new_OR_arr = np.append(arr, [0])
        OR_Test_arrays.append(new_OR_arr)

        # XOR logic adds a new integer at end of array
        if not (arr[0] == arr[1]):
            new_XOR_arr = np.append(arr, [1])
        else:
            new_XOR_arr = np.append(arr, [0])
        XOR_Test_arrays.append(new_XOR_arr)

        # DEBUG
        if debug:
            print("{:-^19}".format("DEBUG"))
            print(new_AND_arr)
            print(new_OR_arr)
            print(new_XOR_arr)

    if operator == "AND":
        return AND_Training_arrays, AND_Test_arrays
    if operator == "OR":
        return OR_Training_arrays, OR_Test_arrays
    if operator == "XOR":
        return XOR_Training_arrays, XOR_Test_arrays


class TsetlinMachine:
    def __init__(self, number_of_clauses, number_of_features, number_of_states, s, threshold):
        self.number_of_clauses = number_of_clauses
        self.number_of_features = number_of_features
        self.number_of_states = number_of_states
        self.s = s
        self.threshold = threshold

        # State of each Tsetlin Automata. Either "number_of_states" or "number_of_states + 1"
        self.ta_state = np.random.choice([self.number_of_states, self.number_of_states+1],
                                         size=(self.number_of_clauses, self.number_of_features, 2)).astype(
                                            dtype=np.int32)

        # Keep track of sign of clause
        self.clause_sign = np.zeros(self.number_of_clauses, dtype=np.int32)

        # More data structures to calculate things like clause_output, summation_of_votes, feedback_to_clauses
        self.clause_output = np.zeros(shape=self.number_of_clauses, dtype=np.int32)
        self.feedback_to_clauses = np.zeros(shape=self.number_of_clauses, dtype=np.int32)

        # Set up structure
        for i in range(self.number_of_clauses):
            if i % 2 == 0:
                self.clause_sign[i] = 1
            else:
                # NB: this is originally minus 1, but the task require both clauses to be plus
                self.clause_sign[i] = 1

    # Calculate output of each clause using actions of each Tsetlin automaton.
    def calculate_clause_output(self, x):
        for j in range(self.number_of_clauses):
            self.clause_output[j] = 1
            for k in range(self.number_of_features):
                action_include = self.action(self.ta_state[j, k, 0])
                action_include_negated = self.action(self.ta_state[j, k, 1])

                if (action_include == 1 and x[k] == 0) or (action_include_negated == 1 and x[k] == 1):
                    self.clause_output[j] = 0
                    break

    # Translate state to action
    def action(self, state):
        if state <= self.number_of_states:
            return 0
        else:
            return 1

    # Sum up the votes for each output decision (y = 0 or y = 1)
    def sum_up_clause_votes(self):
        output_sum = 0
        for i in range(self.number_of_clauses):
            output_sum += self.clause_output[i]*self.clause_sign[i]

        if output_sum > self.threshold:
            output_sum = self.threshold
        elif output_sum < -self.threshold:
            output_sum = -self.threshold

        return output_sum

    def get_state(self, clause, feature, automaton_type):
        return self.ta_state[clause, feature, automaton_type]

    def predict(self, x):
        self.calculate_clause_output(x)
        output_sum = self.sum_up_clause_votes()

        if output_sum >= 0:
            return 1
        else:
            return 0

    def evaluate(self, x, y, number_of_examples):
        xi = np.zeros((self.number_of_features, ), dtype=np.int32)

        errors = 0
        for i in range(number_of_examples):
            for j in range(self.number_of_features):
                xi[j] = x[i][j]

            self.calculate_clause_output(xi)

            output_sum = self.sum_up_clause_votes()

            if output_sum > 0 and y[i] == 0:
                errors += 1
            elif output_sum == 0 and y[i] == 1:
                errors += 1

        return (1.0 - (1.0 * errors / number_of_examples)) * 100

    def update_simple(self, x, y):
        self.calculate_clause_output(x)
        output_sum = self.sum_up_clause_votes()

        # Reset feedback
        for j in range(self.number_of_clauses):
            self.feedback_to_clauses[j] = 0

        # Calculate feedback to clauses
        if y == 1 and output_sum < 1:
            for j in range(self.number_of_clauses):
                self.feedback_to_clauses[j] += 1

        if y == 0 and output_sum > 0:
            for j in range(self.number_of_clauses):
                self.feedback_to_clauses[j] -= 1

        for j in range(self.number_of_clauses):
            # Type I feedback (Combats False Negative Output)
            if self.feedback_to_clauses[j] > 0:

                if self.clause_output[j] == 0:
                    for k in range(self.number_of_features):
                        if self.ta_state[j, k, 0] > 1:
                            self.ta_state[j, k, 0] -= 1
                        if self.ta_state[j, k, 1] > 1:
                            self.ta_state[j, k, 1] -= 1

                if self.clause_output[j] == 1:
                    for k in range(self.number_of_features):
                        if x[k] == 1:
                            if self.ta_state[j, k, 0] < self.number_of_states * 2:
                                self.ta_state[j, k, 0] += 1

                            if self.ta_state[j, k, 1] > 1:
                                self.ta_state[j, k, 1] -= 1

                        elif x[k] == 0:
                            if self.ta_state[j, k, 1] < self.number_of_states * 2:
                                self.ta_state[j, k, 1] += 1

                            if self.ta_state[j, k, 0] > 1:
                                self.ta_state[j, k, 0] -= 1

                # Type II feedback (Combats False Positive Output)
                elif self.feedback_to_clauses[j] < 0:
                    if self.clause_output[j] == 1:
                        for k in range(self.number_of_features):
                            action_include = self.action(self.ta_state[j, k, 0])
                            action_include_negated = self.action(self.ta_state[j, k, 1])

                            if x[k] == 0:
                                if action_include == 0 and self.ta_state[j, k, 0] < self.number_of_states * 2:
                                    self.ta_state[j, k, 0] += 1
                            elif x[k] == 1:
                                if action_include_negated == 0 and self.ta_state[j, k, 1] < self.number_of_states * 2:
                                    self.ta_state[j, k, 1] += 1

    def update(self, x, y):
        self.calculate_clause_output(x)
        output_sum = self.sum_up_clause_votes()

        # Reset feedback
        for j in range(self.number_of_clauses):
            self.feedback_to_clauses[j] = 0

        if y == 1:
            # Calculate feedback to clauses
            for j in range(self.number_of_clauses):
                if 1.0 * random.random() > 1 * (self.threshold - output_sum) / (2 * self.threshold):
                    continue

                if self.clause_sign[j] > 0:
                    # Type I feedback
                    self.feedback_to_clauses[j] += 1
                elif self.clause_sign[j] < 0:
                    # Type II feedback
                    self.feedback_to_clauses[j] -= 1
        elif y == 0:
            for j in range(self.number_of_clauses):
                if 1.0 * random.random() > 1 * (self.threshold + output_sum) / (2 * self.threshold):
                    continue

                if self.clause_sign[j] > 0:
                    # Type II feedback
                    self.feedback_to_clauses[j] -= 1
                elif self.clause_sign[j] < 0:
                    # Type I feedback
                    self.feedback_to_clauses[j] += 1

        for j in range(self.number_of_clauses):
            # Type I feedback (Combats False Negative Output)
            if self.feedback_to_clauses[j] > 0:

                if self.clause_output[j] == 0:
                    for k in range(self.number_of_features):
                        if 1.0 * random.random() <= 1.0 / self.s:
                            if self.ta_state[j, k, 0] > 1:
                                self.ta_state[j, k, 0] -= 1
                        if 1.0 * random.random() <= 1.0 / self.s:
                            if self.ta_state[j, k, 1] > 1:
                                self.ta_state[j, k, 1] -= 1

                if self.clause_output[j] == 1:
                    for k in range(self.number_of_features):
                        if x[k] == 1:
                            if 1.0 * random.random() <= 1.0 * (self.s - 1) / self.s:
                                if self.ta_state[j, k, 0] < self.number_of_states * 2:
                                    self.ta_state[j, k, 0] += 1

                            if 1.0 * random.random() <= 1.0 / self.s:
                                if self.ta_state[j, k, 1] > 1:
                                    self.ta_state[j, k, 1] -= 1

                        elif x[k] == 0:
                            if 1.0 * random.random() <= 1.0 * (self.s - 1) / self.s:
                                if self.ta_state[j, k, 1] < self.number_of_states * 2:
                                    self.ta_state[j, k, 1] += 1

                            if 1.0 * random.random() <= 1.0 / self.s:
                                if self.ta_state[j, k, 1] > 1:
                                    self.ta_state[j, k, 0] -= 1

                # Type II feedback (Combats False Positive Output)
                elif self.feedback_to_clauses[j] < 0:
                    if self.clause_output[j] == 1:
                        for k in range(self.number_of_features):
                            action_include = self.action(self.ta_state[j, k, 0])
                            action_include_negated = self.action(self.ta_state[j, k, 1])

                            if x[k] == 0:
                                if action_include == 0 and self.ta_state[j, k, 0] < self.number_of_states * 2:
                                    self.ta_state[j, k, 0] += 1
                            elif x[k] == 1:
                                if action_include_negated == 0 and self.ta_state[j, k, 1] < self.number_of_states * 2:
                                    self.ta_state[j, k, 1] += 1

    def fit(self, x, y, number_of_examples, epochs=100):
        xi = np.zeros((self.number_of_features, ), dtype=np.int32)
        random_index = np.arange(number_of_examples)

        for epoch in range(epochs):
            # progress(epoch, epochs)
            np.random.shuffle(random_index)

            for i in range(number_of_examples):
                example_id = random_index[i]
                target_class = y[example_id]

                for j in range(self.number_of_features):
                    xi[j] = x[example_id][j]

                self.update(xi, target_class)
                # self.update_simple(xi, target_class)
        return


def progress(count, total, status=''):
    bar_len = 10
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush()
    if count == total - 1:
        print("\n")


def train(operator, s, t):
    # Load training and test data
    n_bits = 2
    training_data, test_data = generate_data(operator=operator, n_bits=n_bits, n_training_sets=200, n_test_sets=100)
    # Data structures
    x_training = []
    y_training = []
    x_test = []
    y_test = []

    for line in training_data:
        x_training.append(line[0:n_bits])
        y_training.append(line[n_bits])

    for line in test_data:
        x_test.append(line[0:n_bits])
        y_test.append(line[n_bits])

    # Initialize the Tsetlin Machine
    tsetlin_machine = TsetlinMachine(number_of_clauses, number_of_features, number_of_states, s, threshold)

    # Train the Tsetlin Machine
    tsetlin_machine.fit(x_training, y_training, len(y_training), epochs=epochs)

    # Evaluate the Tsetlin Machine
    accuracy = tsetlin_machine.evaluate(x_test, y_test, len(y_test))
    print("Accuracy on", operator, "with s:", s, "and t:", t, "-->", int(accuracy))
    return t, s, accuracy


print("Creating Tsetlin Machine")

# Parameters for Tsetlin Machine
threshold = 1
# s = 3.9
number_of_clauses = 2
number_of_features = 2
number_of_states = 100

# Training config
epochs = 100

operators = ["AND", "OR", "XOR"]
parameter_s = [0.5, 1, 1.5, 2, 3.5, 3.9, 4, 4.5, 5, 8, 10]
parameter_t = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

s_col = []
t_col = []
a_col = []

df_t = pd.DataFrame()
df_s = pd.DataFrame()
df_a = pd.DataFrame()

for op in operators:
    for t in parameter_t:
        for s in parameter_s:
            stats = train(op, s, t)
            t_col += stats[0]
            s_col += stats[1]
            a_col += stats[2]


# TODO:
# clause sign - begge pluss
# make simpler activation function (update?)

# AND
# 0.5   6
# 1     7
# 1.5   2
# 3.5   8
# 3.5   9
# 4     6
# 5     1
# 5     3
# 8     3

# OR
# 0.5   9
# 1.5   10
# 2     1
# 2     5
# 3.5   6
# 3.5   8
# 3.9   10
# 4     2
# 4.5   4
# 4.5   7
# 4.5   9
# 5     5
# 5     10
# 8     10

# XOR
# 3.9   5   (0.75)
# 4.5   3   (0.74)
# 4.5   9   (0.72)
# 10    3
