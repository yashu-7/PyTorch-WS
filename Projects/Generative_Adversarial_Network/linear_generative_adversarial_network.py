import os
import cv2
import torch
import torch.nn as nn
from torch import optim
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader, Dataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'model running on {device}')

dataset_path = r'datasets/cryptopunks'

class CryptoPunksDataset(Dataset):
    def __init__(self, dataset_path):
        self.images_path = [os.path.join(dataset_path, img) for img in os.listdir(dataset_path) if img.endswith(('.png','.jpg','.jpeg'))]
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (x * 2) - 1)  # Normalize to [-1, 1]
        ])
    
    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, index):
        img = cv2.imread(self.images_path[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(img)
        return img

data = CryptoPunksDataset(dataset_path)

# Use fewer workers to limit GPU overclocking
data_loader = DataLoader(data, batch_size=32, shuffle=True)

# Generator
class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 64 * 64 * 3),
            nn.Tanh(),
        )

    def forward(self, x):
        img = self.model(x)
        img = img.view(img.size(0), 3, 64, 64)
        return img

# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(64 * 64 * 3, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 512),  # Fixed size from 1023 to 1024
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        img_flatten = img.view(img.size(0), -1)
        validity = self.model(img_flatten)
        return validity

latent_dim = 256
lr = 0.0002
b1, b2 = 0.5, 0.999  # Corrected to commonly used value for b2

gen = Generator(latent_dim).to(device)
des = Discriminator().to(device)

adversarial_loss = nn.BCELoss()

gen_optim = optim.Adam(gen.parameters(), lr=lr, betas=(b1, b2))
des_optim = optim.Adam(des.parameters(), lr=lr, betas=(b1, b2))

# Training Loop
def train(gen, des, data_loader, gen_optim, des_optim, epoch, device):
    gen.train()
    des.train()

    for batch_idx, images in enumerate(data_loader):
        real_images = images.to(device)
        batch_size = real_images.size(0)

        # Train Discriminator
        des_optim.zero_grad()
        
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        real_loss = adversarial_loss(des(real_images), real_labels)

        z = torch.randn(batch_size, latent_dim).to(device)
        fake_imgs = gen(z)
        fake_loss = adversarial_loss(des(fake_imgs.detach()), fake_labels)  # Detach fake images

        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        des_optim.step()

        # Train Generator
        gen_optim.zero_grad()
        
        g_loss = adversarial_loss(des(fake_imgs), real_labels)  # Trick discriminator
        g_loss.backward()
        gen_optim.step()

        if batch_idx % 100 == 0:
            print(f'Epoch: {epoch} [{batch_idx * len(images)}/{len(data_loader.dataset)}] '
                  f'Generator Loss: {g_loss.item():.4f} Discriminator Loss: {d_loss.item():.4f}')

# Image Generation
def generate_images(gen, batch_size, epoch, device):
    gen.eval()
    z = torch.randn(batch_size, latent_dim).to(device)
    generate_imgs = gen(z)
    save_image(generate_imgs[:64], f'linear_gan_results/sample_from_epoch_{epoch}.png', nrow=8, normalize=True)

# Training parameters
num_epochs = 100

# Training
for epoch in range(num_epochs):
    train(gen, des, data_loader, gen_optim, des_optim, epoch, device)
    if epoch % 10 == 0:
        generate_images(gen, 64, epoch, device)
