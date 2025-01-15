import numpy as np
from scipy.optimize import minimize
from forward_problem import forward_problem

# Load the target data
file_name = "test_data/eps1581_mu1363_w90_l164.npz"
data = np.load(file_name)
target_vector = data['output_data']  # Target vector as NumPy array


# Define the loss function
def loss_function(input_vector):
    eps_r, mu_r, length, width = input_vector
    model_output = forward_problem(eps_r, mu_r, length, width, plot=False, save=False)
    return np.linalg.norm(model_output - target_vector) ** 2  # L2 norm (squared)


# Define bounds for the input variables
bounds = [
    (1.0, 2.0),    # eps_r
    (1.0, 2.0),    # mu_r
    (1.0, 280.0),  # length
    (1.0, 280.0),  # width
]

# Initial guess for the input vector
initial_guess = np.array([1.5, 1.5, 130.0, 130.0])


# callback function to print the current guess at each iteration
def print_guess(current_guess):
    print(f"Current guess: {current_guess}")


# Perform optimization
result = minimize(
    loss_function,
    initial_guess,
    method='SLSQP',
    bounds=bounds,
    options={'maxiter': 1000,
             'disp': True,   # display progress
             'ftol': 1e-12,  # function tolerance
             'eps': 1e-9     # step size for gradients
             },
    callback=print_guess,
)

# Extract the best input vector and loss
best_input_vector = result.x
best_loss = result.fun

# Final results
print("Optimization complete!")
print("Best input vector:", best_input_vector)
print("Best loss:", best_loss)
