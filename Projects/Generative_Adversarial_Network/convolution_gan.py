import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Model running on {device}')

dataset_path = r'datasets/cryptopunks'

class CryptoPunks(Dataset):
    def __init__(self, dataset_path):
        super(CryptoPunks, self).__init__()
        self.images_path = [os.path.join(dataset_path, image) for image in os.listdir(dataset_path) if image.endswith(('.jpg', 'jpeg', 'png'))]
        self.transform = transforms.Compose([
            transforms.ToPILImage(), 
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (x * 2) - 1)  
        ])
    
    def __len__(self):
        return len(self.images_path)
    
    def __getitem__(self, index):
        img = cv2.imread(self.images_path[index])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(img)  
        return img

data = CryptoPunks(dataset_path)
data_loader = DataLoader(data, batch_size=64, shuffle=True)

class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        
        self.linear = nn.Linear(latent_dim, 128 * 8 * 8)
        
        self.conv_block = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),    
            nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  
        )
    
    def forward(self, x):
        x = self.linear(x)
        x = x.view(-1, 128, 8, 8)  
        return self.conv_block(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=4, stride=2, padding=1), 
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        
        self.fc_layer = nn.Sequential(
            nn.Linear(128 * 8 * 8, 256), 
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid() 
        )

    def forward(self,x):
        x = self.conv_block(x)
        x = x.view(x.size(0), -1) 
        return self.fc_layer(x)

latent_dim = 256
gen = Generator(latent_dim).to(device)
disc = Discriminator().to(device)

print(gen)
print(disc)

lr = 0.0003
b1 = 0.5
b2 = 0.999

adversarial_loss = nn.BCELoss()

gen_optim = optim.Adam(gen.parameters(), lr, (0.8,b2))
disc_optim = optim.Adam(disc.parameters(), lr, (b1,b2))

def train(gen, disc, device, data_loader, gen_optim, disc_optim, epoch):
    gen.train()
    disc.train()

    for batch_idx, images in enumerate(data_loader):
        real_images = images.to(device)
        batch_size = real_images.size(0)

        disc_optim.zero_grad()

        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        real_loss = adversarial_loss(disc(real_images), real_labels)

        z = torch.randn(batch_size, latent_dim).to(device)
        fake_imgs = gen(z)
        fake_loss = adversarial_loss(disc(fake_imgs.detach()),fake_labels)

        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        disc_optim.step()

        gen_optim.zero_grad()
        g_loss = adversarial_loss(disc(fake_imgs), real_labels)
        g_loss.backward()
        gen_optim.step()

    # print(batch_idx)
    if batch_idx > 100:
        print(f'Epoch:{epoch} [{batch_idx*len(images)}/{len(data_loader)}]'
              f'G loss: {g_loss.item():.4} D loss: {d_loss.item():.4}')

def generate_images(gen, batch_size, device, epoch):
    gen.eval()
    
    z = torch.randn(batch_size, latent_dim).to(device)
    generated_imgs = gen(z)
    save_image(generated_imgs[:batch_size], f'conv_gan_results/Generated_image_on_epoch_{epoch}.png', nrow=8, normalize=True)

num_epochs = 150

for epoch in range(num_epochs):
    train(gen,disc,device,data_loader, gen_optim,disc_optim,epoch)
    if epoch%10 == 0:
        generate_images(gen,batch_size=64,device=device,epoch=epoch)