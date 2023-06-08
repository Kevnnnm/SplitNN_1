import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import numpy as np
import classes
import datas

trainloader = datas.loaders['train']
testloader = datas.loaders['test']

model = classes.SplitNN().to('mps')

criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
epochs = 12

#train and test the split learning model
classes.train(epochs, model, trainloader, criterion, optimizer)
classes.test(model, testloader)

#train the attacker (malicious server)
attack = classes.Attacker().to('mps')
attack_loader = datas.loaders['attack']
final_loader = datas.final_loaders['test']
optimizer_attack = optim.Adam(attack.parameters(), lr = 1e-3)
classes.attack(epochs, attack, model, optimizer_attack, attack_loader, testloader, True)

#save models
# torch.save(model.state_dict(), "Trained_models/SplitNN_model_MNIST_1.pth")
# print("Saved PyTorch Model State to SplitNN_model.pth")

# torch.save(attack.state_dict(), "Trained_models/attack_model_MNIST_1.pth")
# print("Saved PyTorch Model State to attack_1_model.pth")
