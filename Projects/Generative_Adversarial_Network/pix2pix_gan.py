import os
import cv2
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import random_split, Dataset, DataLoader

dataset_path = r'datasets\custom_cam_data'

class CustomData(Dataset):
    def __init__(self, dataset_path):
        super(CustomData, self).__init__()
        self.images_path = [os.path.join(dataset_path, images) for images in os.listdir(dataset_path) if images.endswith(('.png', '.jpg', '.jpeg'))]
        print(f"Total images: {len(self.images_path)}")
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
    
    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, index):
        img = cv2.imread(self.images_path[index])
        label = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        image = self.transform(image)
        label = self.transform(label)

        return image, label

data = CustomData(dataset_path)
train_len = int(0.8 * len(data))
test_len = len(data) - train_len
train_data, test_data = random_split(data, [train_len, test_len])

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

class UNetGenerator(nn.Module):
    def __init__(self):
        super(UNetGenerator, self).__init__()
        
        def down_block(in_channels, out_channels, kernel_size=4, stride=2, padding=1, batch_norm=True):
            layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)]
            if batch_norm:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        def up_block(in_channels, out_channels, kernel_size=4, stride=2, padding=1, dropout=0.0):
            layers = [nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)]
            layers.append(nn.BatchNorm2d(out_channels))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.ReLU(inplace=True))
            return nn.Sequential(*layers)

        self.enc1 = down_block(1, 64, batch_norm=False) 
        self.enc2 = down_block(64, 128)
        self.enc3 = down_block(128, 256)
        self.enc4 = down_block(256, 512)
        self.enc5 = down_block(512, 512)
        self.enc6 = down_block(512, 512)
        self.enc7 = down_block(512, 512)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )

        self.dec1 = up_block(512, 512, dropout=0.5)
        self.dec2 = up_block(1024, 512, dropout=0.5)
        self.dec3 = up_block(1024, 512, dropout=0.5)
        self.dec4 = up_block(1024, 512)
        self.dec5 = up_block(1024, 256)
        self.dec6 = up_block(512, 128)
        self.dec7 = up_block(256, 64)

        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()   
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        e6 = self.enc6(e5)
        e7 = self.enc7(e6)

        b = self.bottleneck(e7)

        d1 = self.dec1(b)
        d2 = self.dec2(torch.cat([d1, self._resize(e7, d1)], 1))  # Resize e7 to match d1
        d3 = self.dec3(torch.cat([d2, self._resize(e6, d2)], 1))  # Resize e6 to match d2
        d4 = self.dec4(torch.cat([d3, self._resize(e5, d3)], 1))  # Resize e5 to match d3
        d5 = self.dec5(torch.cat([d4, self._resize(e4, d4)], 1))  # Resize e4 to match d4
        d6 = self.dec6(torch.cat([d5, self._resize(e3, d5)], 1))  # Resize e3 to match d5
        d7 = self.dec7(torch.cat([d6, self._resize(e2, d6)], 1))  # Resize e2 to match d6

        out = self.final(torch.cat([d7, self._resize(e1, d7)], 1))  # Resize e1 to match d7
        return out

    def _resize(self, tensor, target_tensor):
        """Resize tensor to match the spatial size of the target tensor."""
        return F.interpolate(tensor, size=target_tensor.shape[2:], mode='bilinear', align_corners=False)

    
class PatchGANDiscriminator(nn.Module):
    def __init__(self):
        super(PatchGANDiscriminator, self).__init__()
        
        def discriminator_block(in_channels, out_channels, kernel_size=4, stride=2, padding=1, batch_norm=True):
            layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)]
            if batch_norm:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        self.model = nn.Sequential(
            discriminator_block(4, 64, batch_norm=False),   
            discriminator_block(64, 128),
            discriminator_block(128, 256),
            discriminator_block(256, 512, stride=1),       
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)  
        )

    def forward(self, img_A, img_B):
        x = torch.cat((img_A, img_B), dim=1)
        return self.model(x)
    
os.makedirs("pix2pix_results", exist_ok=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Model is training on {device}')

generator = UNetGenerator().to(device)
discriminator = PatchGANDiscriminator().to(device)
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
criterion_GAN = nn.MSELoss()
criterion_L1 = nn.L1Loss()

num_epochs = 35
for epoch in range(num_epochs):
    generator.train()
    discriminator.train()
    train_loss_G, train_loss_D = 0, 0
    
    for gray_images, color_images in tqdm(train_loader):
        gray_images, color_images = gray_images.to(device), color_images.to(device)
        
        optimizer_G.zero_grad()
        generated_images = generator(gray_images)
        gan_loss = criterion_GAN(discriminator(generated_images, gray_images), torch.ones_like(discriminator(generated_images, gray_images)))
        l1_loss = criterion_L1(generated_images, color_images)
        loss_G = gan_loss + 100 * l1_loss
        loss_G.backward()
        optimizer_G.step()
        
        optimizer_D.zero_grad()
        real_loss = criterion_GAN(discriminator(color_images, gray_images), torch.ones_like(discriminator(color_images, gray_images)))
        fake_loss = criterion_GAN(discriminator(generated_images.detach(), gray_images), torch.zeros_like(discriminator(generated_images.detach(), gray_images)))
        loss_D = (real_loss + fake_loss) * 0.5
        loss_D.backward()
        optimizer_D.step()
        
        train_loss_G += loss_G.item()
        train_loss_D += loss_D.item()
    
    generator.eval()
    with torch.no_grad():
        val_loss_G = 0
        for gray_images, color_images in tqdm(test_loader):
            gray_images, color_images = gray_images.to(device), color_images.to(device)
            generated_images = generator(gray_images)
            gan_loss = criterion_GAN(discriminator(generated_images, gray_images), torch.ones_like(discriminator(generated_images, gray_images)))
            l1_loss = criterion_L1(generated_images, color_images)
            val_loss_G += gan_loss + 100 * l1_loss
            
            if epoch % 5 == 0:
                result_path = f"gray_to_color_results/epoch_{epoch}_sample.png"
                plt.figure(figsize=(10, 5))
                plt.subplot(1, 3, 1)
                plt.title("Grayscale Input")
                plt.imshow(gray_images[0].cpu().squeeze(), cmap="gray")
                plt.subplot(1, 3, 2)
                plt.title("Generated Color")
                plt.imshow(generated_images[0].cpu().permute(1, 2, 0))
                plt.subplot(1, 3, 3)
                plt.title("Real Color")
                plt.imshow(color_images[0].cpu().permute(1, 2, 0))
                plt.savefig(result_path)
                plt.close()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss G: {train_loss_G/len(train_loader):.4f}, Loss D: {train_loss_D/len(train_loader):.4f}, Val Loss G: {val_loss_G/len(test_loader):.4f}")

torch.save(generator.state_dict(), "models/gray_to_color_generator.pth")
torch.save(discriminator.state_dict(), "models/gray_to_color_discriminator.pth")
print("Training complete and model saved.")
