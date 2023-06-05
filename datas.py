import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose, Normalize

transform = Compose([ToTensor(), Normalize((0.5), (0.5))])

mnist_train_data = datasets.MNIST(
    root = "data",
    train = True,
    download = True,
    transform = transform
)
mnist_test_data = datasets.MNIST(
    root = "data",
    train = False,
    download = True,
    transform = transform
)

data_emnist = datasets.EMNIST(
    root = "data",
    train = True,
    download = True,
    transform = transform,
    split = "letters"
)

loaders = {
    'train' : DataLoader(mnist_train_data, batch_size=64, shuffle=True),
    'test'  : DataLoader(mnist_test_data, batch_size=64,  shuffle=True),
    'attack' : DataLoader(data_emnist, batch_size=64,  shuffle=True),
}

final_loaders = {
    'test'  : DataLoader(mnist_test_data, batch_size=1,  shuffle=True),
}

