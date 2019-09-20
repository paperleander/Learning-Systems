
# Leander Berg Thorkildsen
# 17. September 2019
# Assignment 2 - Generate data sets based on logic gates

# NB: Might have to remove the last line of each file, or script it yourself if your not lazy like me


import numpy as np


print("Generating AND, OR, XOR data sets...")
debug = False

n_training_sets = 5000
n_test_sets = 5000

AND_Training_arrays = []
OR_Training_arrays = []
XOR_Training_arrays = []

AND_Test_arrays = []
OR_Test_arrays = []
XOR_Test_arrays = []

for i in range(n_training_sets):
    # Make a random array of integers (example -> [0, 1, 0, 0, 1, 1, 0, 1]
    arr = np.random.randint(2, size=8)

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


for i in range(n_test_sets):
    # Make a random array of integers (example -> [0, 1, 0, 0, 1, 1, 0, 1]
    arr = np.random.randint(2, size=8)

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


#################################
# Make files with the Data sets #
#################################

with open("AND_Training_Data.txt", "w") as f:
    for line in AND_Training_arrays:
        f.write(''.join([str(i) for i in line]) + "\n")

with open("OR_Training_Data.txt", "w") as f:
    for line in OR_Training_arrays:
        f.write(''.join([str(i) for i in line]) + "\n")

with open("XOR_Training_Data.txt", "w") as f:
    for line in XOR_Training_arrays:
        f.write(''.join([str(i) for i in line]) + "\n")

with open("AND_Test_Data.txt", "w") as f:
    for line in AND_Test_arrays:
        f.write(''.join([str(i) for i in line]) + "\n")

with open("OR_Test_Data.txt", "w") as f:
    for line in OR_Test_arrays:
        f.write(''.join([str(i) for i in line]) + "\n")

with open("XOR_Test_Data.txt", "w") as f:
    for line in XOR_Test_arrays:
        f.write(''.join([str(i) for i in line]) + "\n")

print("Done.")
