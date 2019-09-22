from tsetlin_machine import TsetlinMachine
from generate_data import generate_data, generate_test

import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from collections import defaultdict

"""
-How many epochs?
-How many samples?
-Parameters s and t?
-AND -> S=1 is tha bomb, from T=2 and upwards
-OR  -> 
-XOR -> S=3, T=8 is currently best
"""

data_to_plot = defaultdict(list)


def run(t, s, op):
    tsetlin_machine = TsetlinMachine(n_clauses=2, n_states=100, s=s, t=t)

    x, y = generate_data(op, 100, noise=0.0)
    # x, y = generate_test()

    test_x, test_y = generate_data(op, 100, noise=0.0)

    tsetlin_machine.fit(x, y, test_x, test_y, 100)
    # print("S:", s, " T:", t)
    # print("Training accuracy:", tsetlin_machine.accuracy(x, y))
    print("OP:", op, "S:", s, "T:", t, "Validation accuracy:", tsetlin_machine.accuracy(test_x, test_y))

    for clause in tsetlin_machine.clauses:
        print(clause.get_info())

    return tsetlin_machine.accuracy(test_x, test_y)


operators = ["XOR", "AND", "OR"]
for op in operators:
    for t in range(1, 10):
        acc_total = []
        s_list = []
        for s in np.arange(1, 6, step=0.2):
            acc = run(t, round(s, 2), op)
            acc_total.append(acc)
            s_list.append(s)
        plt.plot(s_list, acc_total)
        plt.title("{}, acc vs s, T {}, ".format(op, t))
        plt.xlabel("s")
        plt.ylabel("accuracy")
        plt.show()
