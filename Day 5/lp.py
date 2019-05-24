import cvxpy as cp
k = cp.Variable()
s = cp.Variable()

constraints = [k + s <= 5, 3 * k + 2 * s <= 12]
obj = cp.Maximize(6 * k + 5 * s)

sol = cp.Problem(obj, constraints)
sol.solve()
print(obj.value)
print(k.value)
print(s.value)