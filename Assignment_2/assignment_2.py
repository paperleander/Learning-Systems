

# Leander Berg Thorkildsen
# 9. September 2019
# Assignment 2


import numpy as np
import random


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
                self.clause_sign[i] = -1

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
                xi[j] = x[i, j]

            self.calculate_clause_output(xi)

            output_sum = self.sum_up_clause_votes()

            if output_sum >= 0 and y[i] == 0:
                errors += 1
            elif output_sum < 0 and y[i] == 1:
                errors += 1

        return 1.0 - (1.0 * errors / number_of_examples)

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
            np.random.shuffle(random_index)

            for i in range(number_of_examples):
                example_id = random_index[i]
                target_class = y[example_id]

                for j in range(self.number_of_features):
                    xi[j] = x[example_id, j]

                self.update(xi, target_class)
        return


print("Creating Tsetlin Machine")

# Parameters for Tsetlin Machine
threshold = 15
s = 3.9
number_of_clauses = 2
number_of_states = 100

# Parameters for pattern recognition
number_of_features = 12

# Training config
epochs = 100

# Load training and test data
training_data = np.loadtxt("data/AND_Training_Data.txt").astype(dtype=np.int32)
test_data = np.loadtxt("data/AND_Test_Data.txt").astype(dtype=np.int32)

# DEBUG
print("TRAINING DATA LOOP:")
for line in training_data:
    print(line)

print("TRAINING DATA:")
# print(training_data)

x_training = training_data[:, 0:8]  # Input features
y_training = training_data[:, 8]  # target value

x_test = test_data[:, 0:8]  # Input features
y_test = test_data[:, 8]  # Target value

# Initialize the Tsetlin Machine
tsetlin_machine = TsetlinMachine(number_of_clauses, number_of_features, number_of_states, s, threshold)

# Train the Tsetlin Machine
tsetlin_machine.fit(x_training, y_training, y_training.shape[0], epochs=epochs)

# Evaluate the Tsetlin Machine
print("Accuracy on test data:", tsetlin_machine.evaluate(x_test, y_test, y_test.shape[0]))
