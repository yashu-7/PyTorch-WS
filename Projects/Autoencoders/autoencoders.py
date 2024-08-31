import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from models import Autoencoder
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
batch_size = 512
learning_rate = 1e-3
num_epochs = 10

# MNIST Dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
])

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

# Initialize model, loss function, and optimizer
model = Autoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training Loop
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for images, _ in trainloader:
        images = images.to(device)
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, images)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(trainloader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}')

# Evaluation
model.eval()
with torch.no_grad():
    test_loss = 0
    for images, _ in testloader:
        images = images.to(device)
        outputs = model(images)
        loss = criterion(outputs, images)
        test_loss += loss.item()

    test_loss /= len(testloader)
    print(f'Test Loss: {test_loss:.4f}')

torch.save(model.state_dict(), 'Projects/models/autoencoder_mnist.pth')