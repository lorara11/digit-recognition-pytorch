import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

# Load data
train_set = datasets.MNIST(root='./datasets', train=True, transform=transforms.ToTensor(),download=True)
test_set = datasets.MNIST(root='./datasets', train=False, transform=transforms.ToTensor(),download=True)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=100, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=True)


# TRAINING
# Initialize parameters
# 784: size of image vector, 10: num of classes
W = torch.randn(784,10)/np.sqrt(784)    
W.requires_grad_()
b = torch.zeros(10,requires_grad = True)

# Optimizer
optimizer = torch.optim.SGD([W,b], lr=0.1)

# Iterate through train set minibatchs
for images, labels in train_loader:
    # Zero out the gradients
    optimizer.zero_grad()
    
    # Forward pass
    x = images.view(-1, 28*28)
    y = torch.matmul(x,W) + b
    cross_entropy = F.cross_entropy(y,labels)
    
    # Backward pass
    cross_entropy.backward()    # Computes gradients
    optimizer.step()    # Adjusts parameters a bit better
   
   
# TESTING
correct = 0
total = len(test_set)

# Turn off autograd engine to speed up evaluation time.
with torch.no_grad():
    # Iterate through test set minibatchs
    for images, labels in test_loader:
        # Forward pass
        x = images.view(-1, 28*28)
        y = torch.matmul(x,W) + b
        
        predictions = torch.argmax(y,dim=1)
        correct += torch.sum((predictions==labels).float())
        
print('Test accuracy: {}'.format(correct/total))
    
    
