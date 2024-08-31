import torch
import torchvision
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from models import Autoencoder 

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = Autoencoder().to(device)
model.load_state_dict(torch.load(r'Projects\models\autoencoder_mnist.pth'))
model.eval() 

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
])

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=512, shuffle=False)

with torch.no_grad():
    test_images, _ = next(iter(testloader))
    test_images = test_images.to(device)

    # Get the reconstructed images and latent space
    reconstructed_images, latent_space = model(test_images)
    
    def show_images(original, reconstructed, latent):
        original = original.cpu().numpy()
        reconstructed = reconstructed.cpu().numpy()
        latent = latent.cpu().numpy()
        
        fig, axes = plt.subplots(3, 5, figsize=(12, 6))
        for i in range(5):
            axes[0, i].imshow(original[i].reshape(28, 28), cmap='gray')
            axes[1, i].imshow(reconstructed[i].reshape(28, 28), cmap='gray')
            axes[2, i].imshow(latent[i].reshape(16, 8), cmap='gray')  
            axes[0, i].axis('off')
            axes[1, i].axis('off')
            axes[2, i].axis('off')
        plt.show()

    show_images(test_images[:5], reconstructed_images[:5], latent_space[:5])