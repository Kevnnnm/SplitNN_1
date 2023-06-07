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

trainloader = datas.shadow_loaders['train']
testloader = datas.shadow_loaders['test']

shadow_model = classes.ShadowNN().to('mps')

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(shadow_model.parameters(), lr=0.01, momentum=0.95)
epochs = 25

# #train and test the split learning model
# classes.train(epochs, shadow_model, trainloader, criterion, optimizer)
# classes.test(shadow_model, testloader)

shadow_model.load_state_dict(torch.load("Trained_models/shadow_model.pth"))

# train the attacker (malicious server)
attack = classes.Attacker().to('mps')
attack_loader = datas.shadow_loaders['attack']
final_loader = datas.shadow_loaders['test']

optimizer_attack = optim.Adam(attack.parameters(), lr = 1e-3)
classes.attack(epochs, attack, shadow_model, optimizer_attack, attack_loader, testloader, final_loader)

# save models
torch.save(shadow_model.state_dict(), "Trained_models/Shadow_Model.pth")
print("Saved PyTorch Model State to SplitNN_model.pth")

torch.save(attack.state_dict(), "attack_1_model.pth")
print("Saved PyTorch Model State to attack_1_model.pth")