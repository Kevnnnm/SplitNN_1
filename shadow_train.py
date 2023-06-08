import torch
from torch import nn, optim

import classes
import datas



#-------------------------------------------------------
#Shadowing the Client Model with assumptions:
# - knowledge of client model architecture
# - possession of a dataset similar to the client's 
#   private data

#-------------------------------------------------------
def train_shadow_attack():
    trainloader = datas.shadow_loaders['train']
    testloader = datas.shadow_loaders['test']

    shadow_model = classes.ShadowNN().to('mps')
    #shadow_model.load_state_dict(torch.load("Trained_models/shadow_model.pth"))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(shadow_model.parameters(), lr=0.01, momentum=0.95)
    epochs = 12

    # #train and test the split learning model
    classes.train(epochs, shadow_model, trainloader, criterion, optimizer)
    classes.test(shadow_model, testloader)



    # train the attacker (malicious server)
    attack = classes.Attacker().to('mps')
    #attack.load_state_dict(torch.load("Trained_models/attack_1_model.pth"))
    attack_loader = datas.shadow_loaders['attack']

    optimizer_attack = optim.Adam(attack.parameters(), lr = 1e-3)
    classes.attack(epochs, attack, shadow_model, optimizer_attack, attack_loader, testloader)

    # save models
    torch.save(shadow_model.state_dict(), "Trained_models/Shadow_Model_EMNIST_1.pth")
    print("Saved PyTorch Model State to SplitNN_model.pth")

    torch.save(attack.state_dict(), "Trained_models/attack_model_EMNIST_1.pth")
    print("Saved PyTorch Model State to attack_1_model.pth")

def attack_on_shadow():
    trainloader = datas.loaders['train']
    testloader = datas.loaders['test']

    model = classes.SplitNN().to('mps')
    attack = classes.Attacker().to('mps')

    model.load_state_dict(torch.load("Trained_models/SplitNN_model_MNIST_1.pth"))
    attack.load_state_dict(torch.load("Trained_models/attack_model_EMNIST_1.pth"))
    optimizer_attack = optim.Adam(attack.parameters(), lr = 1e-3)
    optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
    criterion = nn.NLLLoss()
    
    #epochs = 12

    #run original client model with NMIST data (to verify model is correct one)
    classes.test(model, testloader)

    #test our attacker on the client model
    classes.shadow_attack(attack, model, testloader)

#train_shadow_attack()

attack_on_shadow()