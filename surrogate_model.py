import torch
import numpy as np
import os

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F

import pdb


class PinnModel(nn.Module):
    def __init__(self):
        super(PinnModel, self).__init__()
        # Define layers
        self.fc1 = nn.Linear(4, 64)     # Input layer (4 parameters to 64 neurons)
        self.fc2 = nn.Linear(64, 128)  # Hidden layer (64 to 128 neurons)
        self.fc3 = nn.Linear(128, 7020)  # Hidden layer (128 to 7020 neurons = 20 * 351)
        self.fc4 = nn.Linear(7020, 7020)  # Hidden layer (128 to 7020 neurons = 20 * 351)

    def forward(self, x):
        # Pass through layers
        x = F.relu(self.fc1(x))  # Shape: [1, 64]
        x = F.relu(self.fc2(x))  # Shape: [1, 128]
        x = F.relu(self.fc3(x))  # Shape: [1, 128]
        x = self.fc4(x)          # Shape: [1, 7000]
        return x


class SyntheticDataset(Dataset):
    def __init__(self, data_dir="./training_data"):
        """
        Initializes the SyntheticDataset class.
        """
        self.data_dir = data_dir
        self.file_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".npz")]
        self.normalize()

    def normalize(self):
        inputs = []
        self.output_min = float("inf")
        self.output_max = float("-inf")

        for file in self.file_paths:
            npz_file = np.load(file)
            inputs.append(npz_file["input_data"])
            output_data = npz_file["output_data"]

            self.output_min = min(self.output_min, output_data.min())
            self.output_max = max(self.output_max, output_data.max())

        inputs = np.vstack(inputs)  # shape (N, 4)
        self.input_mean = np.mean(inputs, axis=0)
        self.input_std = np.std(inputs, axis=0)

    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.file_paths)

    def __getitem__(self, idx):
        """
        Retrieves a single sample from the dataset.
        """
        # load data
        file_path = self.file_paths[idx]
        npz_file = np.load(file_path)
        # get input and output
        input_data = npz_file["input_data"]
        output_data = npz_file["output_data"]
        # normalize and scale data
        input_data = (input_data - self.input_mean) / self.input_std
        output_data = (output_data - self.output_min) / (self.output_max - self.output_min)
        # convert to tensors
        input_tensor = torch.tensor(input_data, dtype=torch.float32)
        output_tensor = torch.tensor(output_data, dtype=torch.float32)
        # flatten output
        output_tensor = torch.flatten(output_tensor)
        return input_tensor, output_tensor


def test(dataloader, model, loss_fn):
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
    test_loss /= num_batches
    print(f"Test Error: \n  Avg loss: {test_loss:>8f} \n")


def train_one_epoch(dataloader, model, loss_fn, optimizer, epoch_ind):
    running_loss = 0.0
    last_loss = 0.0

    model.train()
    # the following for loop will run
    # n = ceiling(number of samples / batch_size)
    # amount of times. data is a list of length 2 that contains
    # two tensors, data[0] of length [batch_size, input vector size]
    # and data[1] of length [batch_size, output vector size]
    for batch, data in enumerate(dataloader):

        inputs, labels = data

        # zero optimizer gradients
        optimizer.zero_grad()

        # Make a prediction for the batch
        pred = model(inputs)

        # Compute the loss and its gradient
        loss = loss_fn(pred, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if batch % 100 == 0:
            last_loss = running_loss / 100  # loss per batch
            print('  batch {} loss: {}'.format(batch + 1, last_loss))
            running_loss = 0.0
    return last_loss


if __name__ == "__main__":
    # select computing device
    # use GPU if available
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    # Initialize the dataset
    training_data = SyntheticDataset(data_dir="./training_data")
    test_data = SyntheticDataset(data_dir="./test_data")

    # Create data loaders.
    batch_size = 32  # Adjust the batch size as needed
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    # Create Model and move it to device
    model = PinnModel().to(device)

    # Create Loss function
    loss_fn = nn.MSELoss()

    # Create optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    # train and test model
    epoch_number = 0
    epochs = 20
    for i in range(epochs):
        print(f"Epoch {i+1}\n-------------------------------")
        # train
        avg_loss = train_one_epoch(train_dataloader, model, loss_fn, optimizer, i)
        # test
        test(test_dataloader, model, loss_fn)
        print("Done!")

    # save model
    torch.save(model.state_dict(), "model.pth")
    print("Saved PyTorch Model State to model.pth")
