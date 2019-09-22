#!/usr/bin/python


#LEARNING AUTOMATONS


import random

class Environment:
    def reward(self, n_yes):
        p = 0
        if n_yes == 1:
            p = 0.2
        if n_yes == 2:
            p = 0.4
        if n_yes == 3:
            p = 0.6
        if n_yes == 4:
            p = 0.4
        if n_yes == 5:
            p = 0.2
        if random.random() <= p:
            return True
        else:
            return False

class Tsetlin:
    def __init__(self, n):
        self.n = n  # Number of states
        self.state = random.choice([self.n, self.n+1])

    def reward(self):
        if self.state <= self.n and self.state > 1:
            self.state -= 1
        elif self.state > self.n and self.state < 2*self.n:
            self.state += 1

    def penalize(self):
        if self.state <= self.n:
            self.state += 1
        elif self.state > self.n:
            self.state -= 1

    def vote(self):
        if self.state <= self.n:
            return "no"
        else:
            return "yes"


env = Environment()

# number of states
n_s = 10

# Create 5 atomatons
automatons = [Tsetlin(n_s) for i in range(5)]

n_votes = 0

for i in range(100):
    # count number of yes-votes
    n_yes = 0

    # make each automaton vote, count number of "yes"
    for i in automatons:
        vote = i.vote()
        if vote == "yes":
            n_yes += 1
    n_votes += n_yes

    # debug
    print("Number of yes:", n_yes)
    print("Total votes:", n_votes)

    # adjust states based on number of yes-votes
    for i in automatons:
        reward = env.reward(n_yes)
        if reward:
            i.reward()
        else:
            i.penalize()
        print(i.state)

average = n_votes/100.0

print("average: {}".format(average))







