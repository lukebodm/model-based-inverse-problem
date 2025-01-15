import random
from forward_problem import forward_problem
from tqdm import tqdm

n_datasets = 50000

# Generate data
for i in tqdm(range(n_datasets), desc="Generating datasets"):
    eps = round(random.uniform(1, 2), 3)
    mu = round(random.uniform(1, 2), 3)
    length = random.randint(4, 130) * 2  # Generate even numbers between 8 and 260
    width = random.randint(3, 130) * 2   # Generate even numbers between 8 and 260
    forward_problem(eps, mu, length, width, False)

# Below is a parallel version that can be used
## Function to wrap the forward_problem execution
#def run_forward_problem(_):
#    eps = round(random.uniform(1, 2), 3)
#    mu = round(random.uniform(1, 2), 3)
#    length = random.randint(4, 130) * 2  # Generate even numbers between 8 and 260
#    width = random.randint(3, 130) * 2   # Generate even numbers between 8 and 260
#    forward_problem(eps, mu, length, width, False)
#
## Use a ProcessPoolExecutor for parallel execution
#if __name__ == "__main__":
#    with ProcessPoolExecutor() as executor:
#        # Use tqdm to track progress
#        list(tqdm(executor.map(run_forward_problem, range(n_datasets)), desc="Generating datasets", total=n_datasets))
