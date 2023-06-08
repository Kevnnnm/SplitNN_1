import torch
from torch import nn, optim
from torch.utils.data import random_split, DataLoader
import datas
import classes

train_data = datas.mnist_train_data
test_data= datas.mnist_test_data
client_train, shadow_train = random_split(train_data, [int(len(train_data) / 2), int(len(train_data) / 2)])
client_test, shadow_test = random_split(test_data, [int(len(test_data) / 2), int(len(test_data) / 2)])

torch.manual_seed(0)
client_loaders = {
    'train': DataLoader(client_train, batch_size = 64, shuffle = True),
    'test': DataLoader(client_test, batch_size = 64, shuffle = True)
}
shadow_loaders = {
    'train': DataLoader(shadow_train, batch_size = 64, shuffle = True),
    'test': DataLoader(shadow_test, batch_size = 64, shuffle = True)
}



def train_models():
    client_model = classes.SplitNN().to('mps')
    shadow_model = classes.SplitNN().to('mps')
    client_criterion = nn.NLLLoss()
    shadow_criterion = nn.NLLLoss()
    client_optimizer = optim.SGD(client_model.parameters(), lr=0.003, momentum=0.9)
    shadow_optimizer = optim.SGD(shadow_model.parameters(), lr=0.003, momentum=0.9)
    epochs = 12

    classes.train(epochs, client_model, client_loaders['train'], client_criterion, client_optimizer)
    classes.test(client_model, client_loaders['test'])

    classes.train(epochs, shadow_model, shadow_loaders['train'], shadow_criterion, shadow_optimizer)
    classes.test(shadow_model, shadow_loaders['test'])

    torch.save(client_model.state_dict(), "Trained_models/Split_MNIST_client.pth")
    print("Saved PyTorch Model State to Split_MNIST_client.pth")

    torch.save(shadow_model.state_dict(), "Trained_models/Split_MNIST_shadow.pth")
    print("Saved PyTorch Model State to Split_MNIST_shadow.pth")

def train_attack():
    shadow_model = classes.SplitNN().to('mps')
    shadow_model.load_state_dict(torch.load("Trained_models/Split_MNIST_shadow.pth"))
    attack = classes.Attacker().to('mps')
    shadow_optimizer = optim.SGD(shadow_model.parameters(), lr=0.003, momentum=0.9)
    epochs = 12


    optimizer_attack = optim.Adam(attack.parameters(), lr = 1e-3)   
    classes.attack(epochs, attack, shadow_model, optimizer_attack, shadow_loaders['train'], shadow_loaders['test'])
    
    torch.save(attack.state_dict(), "Trained_models/Split_MNIST_attack.pth")
    print("Saved PyTorch Model State to Split_MNIST_attack.pth")



#train_models()
#train_attack()
