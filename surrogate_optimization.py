import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from surrogate_model import PinnModel

# Initialize the model and load weights
model = PinnModel()
model.load_state_dict(torch.load('model.pth'))
model.eval()  # Set the model to evaluation mode

# Initialize input vector
input_vector = torch.tensor([1.0, 1.0, 100.0, 100.0], requires_grad=True)

# Load one target file
file_name = "test_data/eps3740_mu1597_w214_l136.npz"  # Example file
data = np.load(file_name)
target_vector = torch.tensor(data['output_data'], dtype=torch.float32)  

# Flatten the target matrix
# target_vector = target_matrix.flatten()

# Define the optimizer
optimizer = optim.SGD([input_vector], lr=0.01)

# Loss function (L2 norm)
loss_fn = nn.MSELoss()

# Optimization loop
num_iterations = 10000  # number of gradient descent iterations
best_loss = float('inf')  # Track the best loss
best_input_vector = input_vector.clone().detach()  # Track the best input vector

for iteration in range(num_iterations):
    optimizer.zero_grad()  # Clear previous gradients

    # Forward pass
    model_output = model(input_vector)  # Compute model output
    model_vector = model_output.flatten()  # Flatten the output matrix

    # Compute loss
    loss = loss_fn(model_vector, target_vector)

    # Backward pass
    loss.backward()

    # Update parameters
    optimizer.step()

    # Clamp inputs to their valid ranges
    with torch.no_grad():
        input_vector[0].clamp_(1.0, 10.0)  # First input: [1, 100]
        input_vector[1].clamp_(1.0, 10.0)   # Second input: [1, 10]
        input_vector[2:].clamp_(0.0, 300.0) # Last two inputs: [0, 300]

    # Update the best input vector if the current loss is better
    if loss.item() < best_loss:
        best_loss = loss.item()
        best_input_vector = input_vector.clone().detach()

    # Print progress every 100 iterations
    if iteration % 100 == 0:
        print(f"Iteration {iteration}, Loss: {loss.item()}, Current Best Input: {best_input_vector.numpy()}")

# Final results
print("Optimization complete!")
print("Best input vector:", best_input_vector.numpy())
print("Best loss:", best_loss)
