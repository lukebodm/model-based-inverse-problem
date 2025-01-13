import random
from forward_problem import forward_problem

n_datasets = 50000

for i in range(n_datasets):
    eps = round(random.uniform(1, 100), 3)
    mu = round(random.uniform(1, 10), 3)
    length = random.randint(10, 250)
    width = random.randint(10, 250)
    forward_problem(eps, mu, length, width, False)
