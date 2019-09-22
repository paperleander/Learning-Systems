import numpy as np
import random


class Automaton:
    def __init__(self, n_states):
        self.n_states = n_states
        self.state = np.random.randint(2 * n_states)

    def evaluate(self):
        return self.state >= self.n_states

    def reward(self):
        """
        Reward each automaton by pushing the state away from the middle
        """
        if self.state >= self.n_states:
            if self.state < self.n_states * 2:
                self.state += 1
        else:
            if self.state > 1:
                self.state -= 1

    def penalize(self):
        """
        Penalize each automaton by pushing the state towards the middle
        """
        if self.state >= self.n_states:
            self.state -= 1
        else:
            self.state += 1


class Clause:
    def __init__(self, n_inputs, n_states, positive):
        self.n_inputs = n_inputs
        self.positive = positive
        self.polarity = 1 if positive else -1
        self.positive_automatons = [Automaton(n_states) for _ in range(n_inputs)]
        self.negative_automatons = [Automaton(n_states) for _ in range(n_inputs)]

    def get_info(self):
        x = [[pos.state, neg.state] for pos, neg in zip(self.positive_automatons, self.negative_automatons)]
        return x

    def evaluate(self, x):
        """
        Return True if x follows all the rules described in the paper
        """
        for x_input, positive_automaton, negative_automaton in zip(x, self.positive_automatons, self.negative_automatons):
            if positive_automaton.evaluate() and not x_input:
                return False
            if negative_automaton.evaluate() and x_input:
                return False
        return True

    def value(self, x):
        """
        Takes the sum of votes from automatons and outputs value of the clause (used when predicting)
        """
        return self.evaluate(x) * self.polarity


class TsetlinMachine:
    def __init__(self, n_clauses, n_states, s, t):
        self.n_clauses = n_clauses
        self.n_states = n_states
        self.s = s  # Precision
        self.t = t  # Threshold
        self.clauses = None  # Initialized later when we know the input and output

    def fit(self, x, y, test_x, test_y, epochs=100):
        """
        Fit TsetlinMachine on training set
        """
        n_inputs = x.shape[1]  # 2 bits input

        self.clauses = [Clause(n_inputs, self.n_states, index % 2 == 0) for index in range(self.n_clauses)]

        indices = list(range(x.shape[0]))  # indices based on number of rows
        for epoch in range(epochs):
            random.shuffle(indices)  # Learn each epoch in random order
            for sample in indices:
                sample_x, sample_y = x[sample], y[sample]
                self.train(sample_x, sample_y)

            # Debug validation accuracy
            #print("Epoch:", epoch, " Accuracy:", self.accuracy(test_x, test_y))

    def predict(self, x):
        """
        Predict on array of input x, outputs array of predicted y
        """
        y = []
        for index in range(x.shape[0]):
            sample_x = x[index]
            sample_y = list()
            # for y_clause in self.clauses:
            # sample_y.append(np.sum([clause.value(sample_x) for clause in y_clause]) >= 0)
            sample_y.append(np.sum([clause.value(sample_x) for clause in self.clauses]) >= 0)
            y.append(sample_y)
        return np.array(y)

    def accuracy(self, x, y):
        """
        Accuracy of an array of inputs, returns fraction of correct predicted y
        """
        predicted_y = self.predict(x)
        #print("predicted_y", predicted_y)
        #print("y", y)
        #print("mean", np.mean(predicted_y == y))
        return int(np.mean(predicted_y == y) * 100)

    def train(self, sample_x, sample_y):
        output_sum = np.sum([clause.value(sample_x) for clause in self.clauses])

        for clause in self.clauses:
            if clause.positive:
                if sample_y:
                    if self.sample_feedback(output_sum, True):
                        self.apply_type_1_feedback(sample_x, clause)
                else:
                    if self.sample_feedback(output_sum, False):
                        self.apply_type_2_feedback(sample_x, clause)
            else:  # reverse for negative clause
                if sample_y:
                    if self.sample_feedback(output_sum, True):
                        self.apply_type_2_feedback(sample_x, clause)
                else:
                    if self.sample_feedback(output_sum, False):
                        self.apply_type_1_feedback(sample_x, clause)

    def sample_feedback(self, output_sum, y_true):
        """
        Return whether or not to sample feedback
        """
        if y_true:
            return np.random.rand() < (self.t - max(-self.t, min(self.t, output_sum))) / (2 * self.t)
        else:
            return np.random.rand() < (self.t + max(-self.t, min(self.t, output_sum))) / (2 * self.t)

    def apply_type_1_feedback_automaton(self, clause_output, x, automaton):
        """
        Reward or penalize, see paper for the table/matrix
        """
        if clause_output:
            if x:
                if automaton.evaluate():
                    if self.sample_high_probability():
                        automaton.reward()
                else:
                    if self.sample_high_probability():
                        automaton.penalize()
            else:
                if not automaton.evaluate():
                    if self.sample_low_probability():
                        automaton.reward()
        else:
            if automaton.evaluate():
                if self.sample_low_probability():
                    automaton.penalize()
            else:
                if self.sample_low_probability():
                    automaton.reward()

    def apply_type_1_feedback(self, sample_x, clause):
        clause_ouput = clause.evaluate(sample_x)
        for x, positive_automaton, negative_automaton in zip(sample_x, clause.positive_automatons,
                                                             clause.negative_automatons):
            self.apply_type_1_feedback_automaton(clause_ouput, x, positive_automaton)
            self.apply_type_1_feedback_automaton(clause_ouput, not x, negative_automaton)

    def apply_type_2_feedback(self, sample_x, clause):
        clause_output = clause.evaluate(sample_x)
        for x, positive_automaton, negative_automaton in zip(sample_x, clause.positive_automatons,
                                                             clause.negative_automatons):
            if clause_output:
                if not x:
                    if not positive_automaton.evaluate():
                        positive_automaton.penalize()
                if x:
                    if not negative_automaton.evaluate():
                        negative_automaton.penalize()

    def sample_high_probability(self):
        return np.random.rand() < (self.s - 1) / self.s

    def sample_low_probability(self):
        return np.random.rand() < 1 / self.s
