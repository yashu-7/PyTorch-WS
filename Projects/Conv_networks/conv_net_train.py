import os
import torch
import pandas as pd
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Model is running on {device}")

class CustomDataset(Dataset):
    def __init__(self, csv_file, img_path, label_mapping, transforms=None, img_size=(112,112)):
        self.data_frame = pd.read_csv(csv_file)
        self.img_path = img_path
        self.transform = transforms
        self.img_size = img_size
        self.label_mapping = label_mapping

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_path, self.data_frame.iloc[idx, 0])
        image = Image.open(img_name).convert("RGB")
        image = image.resize(self.img_size)
        label_str = self.data_frame.iloc[idx, 1]
        label = self.label_mapping[label_str]
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label)

# Label mapping
label_mapping = {
    "YOUNG": 0,
    "MIDDLE": 1,
    "OLD": 2
}

dataset = CustomDataset('faces/train.csv', 'faces/Train', label_mapping, transforms=transforms.ToTensor())
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

print(f"Train {len(train_dataset)}\nTest {len(test_dataset)}")

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(32 * 14 * 14, 128)  # Adjusted for 3 maxpools = 14*14
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 3)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 32 * 14 * 14)  # Adjusted for pooling thrice
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return x

model = ConvNet().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 25

for epoch in range(num_epochs):
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"EPOCH [{epoch+1}/{num_epochs}]\tLOSS {loss.item()}")

model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f"ACCURACY : {100 * correct / total:.2f}%")


torch.save(model.state_dict(), 'Projects\\models\\face_classification_model.pth')