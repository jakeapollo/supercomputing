import matplotlib.pyplot as plt
import numpy as np
import torch as t
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

device = t.device("cuda:0" if t.cuda.is_available() else "cpu")


# An autoencoder with a single narrow hidden layer
class BottleNeck(nn.Module):
    def __init__(self, in_features, hidden_size):
        super(BottleNeck, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_size)
        self.fc2 = nn.Linear(hidden_size, in_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x


# A function which takes as input dataset_size, dimension, sparsity, and generates a dataset of
# vectors with entries which are 1 with probability sparsity and 0 otherwise.
# returns an instance of pytorch Dataset
class SparseDataset(Dataset):
    def __init__(self, dataset_size, dimension, p_on):
        self.dataset_size = dataset_size
        self.dimension = dimension
        self.p_on = p_on
        self.data = t.zeros(self.dataset_size, self.dimension).to(device)
        self.data[t.rand(self.dataset_size, self.dimension) < self.p_on] = 1

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        return self.data[idx]

    def get_data(self):
        return self.data


def L_p_loss(p):
    def loss(x, y):
        return t.norm(x - y, p)

    return loss


def train(model, train_loader, optimizer, criterion, epochs=1):
    losses = []
    for _ in tqdm(range(epochs)):
        for data in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, data)
            loss.backward()
            losses.append(loss.item())
            optimizer.step()
    return losses


# a function which takes in a matrix of shape (2,dimension) and returns a figure with each
# 2-dimensional row plotted as an arrow in the plane
# takes in a savefigname and saves the figure to file
def plot_vectors(matrix, savefigname):
    fig, ax = plt.subplots()
    ax.quiver(
        np.zeros(matrix.shape[0]),
        np.zeros(matrix.shape[0]),
        matrix[:, 0],
        matrix[:, 1],
        scale=1,
        scale_units="xy",
        angles="xy",
        color="red",
    )
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_aspect("equal")
    plt.savefig(savefigname)


# a function which takes in a rank and a dimension, and returns a matrix of shape dimension *
# dimension. For each i in range(rank), generate a random unit vector v_i, and add the rank one
# matrix v_i * v_i^T to the output matrix. Return the output matrix.
def generate_rank_one_matrix(rank, dimension):
    output = t.zeros(dimension, dimension)
    for i in range(rank):
        v = t.randn(dimension)
        v = v / t.norm(v)
        output += t.outer(v, v)
    return output


if __name__ == "__main__":
    dataset_size = 100000
    dimension = 20000
    hidden_size = 100
    avg_num_on = 0.5
    p = 20
    test_size = 1000
    epochs = 200
    one_hot_dataset = t.eye(dimension, device=device)

    p_on = avg_num_on / dimension
    # train_dataset = SparseDataset(dataset_size, dimension, p_on)
    # train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
    train_loader = DataLoader(one_hot_dataset, batch_size=100, shuffle=True)
    model = BottleNeck(dimension, hidden_size).to(device)
    optimizer = t.optim.Adam(model.parameters(), lr=3e-4)
    criterion = L_p_loss(p)
    losses = train(model, train_loader, optimizer, criterion, epochs=epochs)
    test_set = SparseDataset(test_size, dimension, p_on)
    # test_matrix = test_set.get_data()
    test_matrix = one_hot_dataset
    test_output = model(test_matrix)
    errors = test_output - test_matrix
    # plot losses and save figure to file
    plt.plot(losses)
    plt.savefig("losses.png")

    # plot hist of errors and save figure to file
    plt.figure()
    plt.hist(errors.cpu().detach().numpy().flatten(), bins=100)
    plt.savefig("errors.png")

    read_ins = model.fc1.weight.data.cpu().numpy()
    read_offs = model.fc2.weight.data.cpu().numpy()

    # plot read_ins and read_offs and save figure to file. Make sure the relevant matrices are
    # transposed as needed
    # plot_vectors(read_ins.T, "read_ins.png")
    # plot_vectors(read_offs, "read_offs.png")

    # use plt.imshow with colorbar to visualise the matrix output of model(test_matrix) and save
    # figure to file
    plt.figure()
    plt.imshow(test_output.cpu().detach().numpy())
    plt.colorbar()
    plt.savefig("test_output.png")

    comparison = (
        dimension / hidden_size * generate_rank_one_matrix(hidden_size, dimension).to(device)
    )
    comparison_output = comparison
    print("trained loss: ", criterion(test_output, test_matrix).item())
    print("comparison loss: ", criterion(comparison_output, test_matrix).item())
    print("trained L_infinity: ", (test_output - test_matrix).abs().max().item())
    print("comparison L_infinity: ", (comparison_output - test_matrix).abs().max().item())

    comparison_errors = comparison_output - test_matrix
    # plot hist of comparison_errors and save figure to file
    plt.figure()
    plt.hist(comparison_errors.cpu().detach().numpy().flatten(), bins=100)
    plt.savefig("comparison_errors.png")
    # use plt.imshow with colorbar to visualise the matrix output of model(test_matrix) and save
    # figure to file
    plt.figure()
    plt.imshow(comparison_output.cpu().detach().numpy())
    plt.colorbar()
    plt.savefig("test_comparison_output.png")
