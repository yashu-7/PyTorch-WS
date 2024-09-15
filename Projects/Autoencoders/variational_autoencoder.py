import os
import cv2
import torch
from models import VAE
import torch.optim as optim
import torch.nn.functional as F
from torchvision.utils import save_image
from torch.utils.data import DataLoader, Dataset

dataset_path = r'datasets/cryptopunks'

class Crypto_punks(Dataset):
    def __init__(self, dataset_path):
        self.images_path = [os.path.join(dataset_path, filename) for filename in os.listdir(dataset_path) if filename.endswith('.png')]  # Ensure only image files are read
    
    def __len__(self):
        return len(self.images_path)
    
    def __getitem__(self, index):
        image_path = self.images_path[index]
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (128, 128))  # Resize images to 128x128
        img = torch.tensor(img).permute(2, 0, 1).float() / 255.0  # Normalize to [0, 1] and convert to tensor
        img = (img * 2) - 1  # Normalize to [-1, 1]
        return img

data = Crypto_punks(dataset_path)

train_size = int(0.8 * len(data))
test_size = len(data) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(data, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


model = VAE(latent_dim=256)
print(model)

def custom_loss(reconstructed_x, x, mu, logvar):
    recon_loss = F.mse_loss(reconstructed_x, x, reduction='sum')  # MSE loss for reconstruction
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())  # KL divergence loss
    return recon_loss + kl_loss

def train(model, train_loader, optimizer, epoch, device):
    model.train()
    train_loss = 0
    for batch_idx, images in enumerate(train_loader):
        images = images.to(device)

        optimizer.zero_grad()
        recon_images, mu, logvar = model(images)
        loss = custom_loss(recon_images, images, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f'Epoch: {epoch} [{batch_idx * len(images)}/{len(train_loader.dataset)}] Loss: {loss.item() / len(images):.4f}')

    print(f'====> Epoch: {epoch} Average loss: {train_loss / len(train_loader.dataset):.4f}')

def test(model, test_loader, device):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for images in test_loader:
            images = images.to(device)
            recon_images, mu, logvar = model(images)
            loss = custom_loss(recon_images, images, mu, logvar)
            test_loss += loss.item()

    test_loss /= len(test_loader.dataset)
    print(f'====> Test set loss: {test_loss:.4f}')

def generate_and_save_images(model, epoch, device):
    os.makedirs('new_results', exist_ok=True)  # Ensure the directory exists
    with torch.no_grad():
        z = torch.randn(64, 256).to(device)  # Latent space size 256
        generated_images = model.decode(z).cpu()
        save_image(generated_images, f'new_results/generated_epoch_{epoch}.png')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VAE(latent_dim=256).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 500
for epoch in range(1, epochs + 1):
    train(model, train_loader, optimizer, epoch, device)
    test(model, test_loader, device)
    generate_and_save_images(model, epoch, device)

torch.save(model.state_dict(), "variational_autoencoder.pth")
print("Model saved as variational_autoencoder.pth")