import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.autograd import Variable
from time import time
import numpy as np
import matplotlib.pyplot as plt
import math
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse

#-------------------------------------------------------
#initial Client Server normal Split Learning Model
#-------------------------------------------------------

#SplitNN simulating client side before cut layer and server side after first layer
class SplitNN(nn.Module):
  def __init__(self):
    super(SplitNN, self).__init__()
    self.input_size = 784
    self.hidden_sizes = [500, 128]
    self.output_size = 10
    self.cut_layer = 500

    self.first_part = nn.Sequential(
      nn.Linear(self.input_size, self.hidden_sizes[0]),
      nn.ReLU(),
      nn.Linear(self.hidden_sizes[0], self.cut_layer),
      nn.ReLU(),
      )
    self.second_part = nn.Sequential(
      nn.Linear(self.cut_layer, self.hidden_sizes[1]),
      nn.Linear(self.hidden_sizes[1], self.output_size),
      nn.LogSoftmax(dim=1)
    )

  def forward(self, x):
    return self.second_part(self.first_part(x))

class ShadowNN(nn.Module):
  def __init__(self):
    super(ShadowNN, self).__init__()
    self.output_size = 26
    self.input_size = 784
    self.hidden_sizes = [1600, 500, 128, 64]
    self.cut_layer = 500

    self.first_part = nn.Sequential(
      nn.Linear(self.input_size, self.hidden_sizes[1]),
      nn.ReLU(),
      # nn.Linear(self.hidden_sizes[1], self.hidden_sizes[1]),
      # nn.ReLU(),
      nn.Linear(self.hidden_sizes[1], self.cut_layer),
      nn.ReLU(),
      )
    self.second_part = nn.Sequential(
      nn.Linear(self.cut_layer, self.hidden_sizes[2]),
      nn.ReLU(),
      # nn.Linear(self.hidden_sizes[2], self.hidden_sizes[2]),
      # nn.ReLU(),
      nn.Linear(self.hidden_sizes[2], self.output_size),
      
      nn.LogSoftmax(dim=1)
    )

  def forward(self, x):
    return self.second_part(self.first_part(x))


#Malicious server that will attack after receiving cut layer
class Attacker(nn.Module):
  def __init__(self):
    super(Attacker, self).__init__()
    self.layers= nn.Sequential(
        nn.Linear(500, 1000),
        nn.ReLU(),
        nn.Linear(1000, 784),
    )

  def forward(self, x):
    return self.layers(x)





#Train the client model on the NMIST dataset of handwritten digits
def train(num_epochs, model, loader, loss_fn, optimizer):
    device = 'mps'
        
    # Train the model
    time0 = time()
        
    for e in range(num_epochs):
        running_loss = 0
        for images, labels in loader:
          images, labels = images.view(images.shape[0], -1).to(device), labels.to(device) #(number batches, auto fill columns based on exisitng dimensions)
    
          # Training pass
          optimizer.zero_grad()
          
          output = model(images)
          loss = loss_fn(output, labels)
          
          #This is where the model learns by backpropagating
          loss.backward()
          
          #And optimizes its weights here
          optimizer.step()
          
          running_loss += loss.item()
        else:
          print("Epoch {} - Training loss: {}".format(e + 1, running_loss/len(loader)))
    print("\nTraining Time (in minutes) =",(time()-time0)/60)

def test(model, loader):
  #iterate through test set and calculate accuracy
  correct_count, all_count = 0, 0
  for images,labels in loader:
    for i in range(len(labels)):
      img, labels = images[i].view(1, 784).to('mps'), labels.to('mps')
      with torch.no_grad():
          logps = model(img)

      
      ps = torch.exp(logps)
      probab = list(ps.cpu().numpy()[0])
      pred_label = probab.index(max(probab))
      true_label = labels.cpu().numpy()[i]
      if(true_label == pred_label):
        correct_count += 1
      all_count += 1

  print("Number Of Images Tested =", all_count)
  print("\nModel Accuracy =", (correct_count/all_count))


def attack(num_epochs, attack, model, optimizer, attack_loader, test_loader):
  for e in range(num_epochs):
    running_loss = 0
    for data, targets in attack_loader:
      #print(data.shape)
      data, targets = data.to('mps'), targets.to('mps')
      data = data.reshape(data.shape[0], -1)
      # Reset gradients
      optimizer.zero_grad()

      # First, get outputs from the target model
      target_outputs = model.first_part(data)

      # Next, recreate the data with the attacker
      attack_outputs = attack(target_outputs)

      # We want attack outputs to resemble the original data
      loss = ((data - attack_outputs)**2).mean()

      # Update the attack model
      loss.backward()
      optimizer.step()
      running_loss += loss.item()
    else:
      print("Epoch {} - Training loss: {}".format(e + 1, running_loss/len(attack_loader)))
  total_mse, total_ssim = 0, 0
  for i, (data, targets) in enumerate(test_loader):
    data, targets = data.to('mps'), targets.to('mps')
    #print(data.shape)
    data = data.reshape(data.shape[0], -1)
    target_outputs = model.first_part(data)
    recreated_data = attack(target_outputs)

    data_np = data.cpu().numpy()
    recreated_data_np = recreated_data.cpu().detach().numpy()

    total_mse += mse(data_np, recreated_data_np)
    total_ssim += ssim(data_np, recreated_data_np, data_range = 1.0)
    
    if i < 3:

      # print(data_np.shape)
      # print(recreated_data_np.shape)

      # Display the original data
      plt.imshow(data_np[-1].reshape(28, 28), cmap='gray')
      plt.title("Original Data")
      plt.show()

      # Display the reconstructed data
      plt.imshow(recreated_data_np[-1].reshape(28, 28), cmap='gray')
      plt.title("Reconstructed Data")
      plt.show()
  print(f"AVG MSE: {total_mse / len(test_loader)}")
  print(f"AVG SSIM: {total_ssim / len(test_loader)}")
        
def shadow_attack(attack, model, test_loader):
  total_mse, total_ssim = 0, 0
  for i, (data, targets) in enumerate(test_loader):
    data, targets = data.to('mps'), targets.to('mps')
    #print(data.shape)
    data = data.reshape(data.shape[0], -1)
    target_outputs = model.first_part(data)
    recreated_data = attack(target_outputs)
    
    data_np = data.cpu().numpy()
    recreated_data_np = recreated_data.cpu().detach().numpy()

    total_mse += mse(data_np, recreated_data_np)
    total_ssim += ssim(data_np, recreated_data_np, data_range = 1.0) #look into this more
    


    if i < 3:
      # Convert the tensors to numpy arrays
      data_np = data.cpu().numpy()
      recreated_data_np = recreated_data.cpu().detach().numpy()

      # print(data_np.shape)
      # print(recreated_data_np.shape)

      # Display the original data
      plt.imshow(data_np[-1].reshape(28, 28), cmap='gray')
      plt.title("Original Data")
      plt.show()

      # Display the reconstructed data
      plt.imshow(recreated_data_np[-1].reshape(28, 28), cmap='gray')
      plt.title("Reconstructed Data")
      plt.show()
  print(f"AVG MSE: {total_mse / len(test_loader)}")
  print(f"AVG SSIM: {total_ssim / len(test_loader)}")
