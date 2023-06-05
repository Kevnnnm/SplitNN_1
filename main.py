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
epochs = 10

classes.train(epochs, model, trainloader, criterion, optimizer)
classes.test(model, testloader)

attack = classes.Attacker().to('mps')
attack_loader = datas.loaders['attack']
final_loader = datas.final_loaders['test']
optimizer_attack = optim.Adam(attack.parameters(), lr = 1e-3)
classes.attack(epochs, attack, model, optimizer_attack, attack_loader, testloader, final_loader)
