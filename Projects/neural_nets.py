import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Model is training on {device}")

class FeedForward(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FeedForward, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)

        return out

# Download=False was set as i had it downloaded already
train_dataset = torchvision.datasets.MNIST('',train=True,transform=transforms.ToTensor(),download=False)
test_dataset = torchvision.datasets.MNIST('',train=False,transform=transforms.ToTensor())

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True)

input_shape = 28 * 28
hidden_shape = 128
output_shape = 10

model = FeedForward(input_shape, hidden_shape, output_shape).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10

for epoch in range(num_epochs):
    for images,labels in train_loader:
        images = images.reshape(-1,input_shape).to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

model.eval()

with torch.no_grad():
    correct = 0
    total = 0
    for images,labels in test_loader:
        images = images.reshape(-1, input_shape).to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print(f'Accuracy of the model on the 10,000 test images: {100 * correct / total:.2f}%')
