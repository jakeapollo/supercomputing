import os

import matplotlib.pyplot as plt
import numpy as np
import torch as t
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

device = t.device("cuda:0" if t.cuda.is_available() else "cpu")


# A sparse autoencoder architecture
class SAE(nn.Module):
    def __init__(self, dimension, hidden_size, nonlinearity=nn.ReLU(), freeze=True):
        super(SAE, self).__init__()
        self.fc1 = nn.Linear(dimension, hidden_size)
        self.fc2 = nn.Linear(hidden_size, dimension)
        self.nonlinearity = nonlinearity
        if freeze:
            self.fc2.weight.requires_grad = False
            self.fc2.bias.requires_grad = False

    def forward(self, x):
        x = self.fc1(x)
        acts = self.nonlinearity(x)
        out = self.fc2(acts)
        return out, acts


# a dataset of random unit vectors in R^dimension of size dataset_size
class RandomUnitVectors(Dataset):
    def __init__(self, dataset_size, dimension):
        self.dataset_size = dataset_size
        self.dimension = dimension
        self.data = t.randn(self.dataset_size, self.dimension).to(device)
        self.data = self.data / t.norm(self.data, dim=1).unsqueeze(1)

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        return self.data[idx]

    def get_data(self):
        return self.data


def train(
    model,
    train_loader,
    optimizer,
    sparsity_loss_fn,
    sparsity_penality,
    epochs=1,
    checkpoint_freq=10,
):
    reconstruction_losses = []
    sparsity_losses = []
    epochs = list(range(epochs))
    checked_epochs = []
    explained_variances = []
    reconstruction_criterion = nn.MSELoss()
    for epoch in epochs:
        for data in train_loader:
            optimizer.zero_grad()
            output, acts = model(data)
            reconstruction_loss = reconstruction_criterion(output, data)
            sparsity_loss = sparsity_loss_fn(acts)
            loss = reconstruction_loss + sparsity_penality * sparsity_loss
            loss.backward()
            reconstruction_losses.append(reconstruction_loss.item())
            sparsity_losses.append(sparsity_loss.item())
            optimizer.step()
        if epoch % checkpoint_freq == 0:
            checked_epochs.append(epoch)
            # calculate explained variance
            explained_variance = 1 - t.var(output) / t.var(data)
            explained_variances.append(explained_variance.item())
            print(
                f"Epoch {epoch}, reconstruction loss {reconstruction_loss.item()}, "
                f"sparsity loss {sparsity_loss.item()}, explained variance {explained_variance}"
            )
    out_dict = {
        "reconstruction_losses": reconstruction_losses,
        "sparsity_losses": sparsity_losses,
        "all_epochs": epochs,
        "checked_epochs": checked_epochs,
        "explained_variances": explained_variances,
    }
    return out_dict


if __name__ == "__main__":
    dataset_size = 50
    dimension = 768
    hidden_size = 25000
    nonlinearity = nn.ReLU()
    freeze = True
    sparsity_penality = 0.02
    epochs = 1000
    checkpoint_freq = 50
    batch_size = 50
    learning_rate = 0.01
    model = SAE(dimension, hidden_size, nonlinearity, freeze).to(device)
    dataset = RandomUnitVectors(dataset_size, dimension)
    train_loader = DataLoader(dataset, batch_size=batch_size)
    optimizer = t.optim.Adam(model.parameters(), lr=learning_rate)
    sparsity_loss_fn = lambda x: t.norm(x, p=1)
    results = train(
        model,
        train_loader,
        optimizer,
        sparsity_loss_fn,
        sparsity_penality,
        epochs,
        checkpoint_freq,
    )
    save_directory = "random_sae_results/"
    # if save directory does not exist, create it
    try:
        os.mkdir(save_directory)
    except FileExistsError:
        pass
    t.save(model.state_dict(), save_directory + "model.pt")
    t.save(results, save_directory + "results.pt")
    t.save(dataset.get_data(), save_directory + "data.pt")
    plt.plot(results["all_epochs"], results["reconstruction_losses"])
    plt.savefig(save_directory + "reconstruction_losses.png")
    plt.plot(results["all_epochs"], results["sparsity_losses"])
    plt.savefig(save_directory + "sparsity_losses.png")
    plt.plot(results["checked_epochs"], results["explained_variances"])
    plt.savefig(save_directory + "explained_variances.png")
