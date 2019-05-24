import cvxpy as cp
import csv
import numpy as np
f = open('train.csv', 'r')
csvreader = csv.reader(f)
c = 0
y = []
x = []
for rows in csvreader:
    if c == 0:
        c = c + 1
        continue
    x.append(float(rows[0]))
    y.append(float(rows[1]))
x = np.array(x)
y = np.array(y)
w = cp.Variable()
b = cp.Variable()
constraints = []
tmp = ((w * x + b) - y)
obj = cp.Minimize(cp.sum_squares(tmp))
sol = cp.Problem(obj, constraints)
sol.solve()
print(w.value)
print(b.value)