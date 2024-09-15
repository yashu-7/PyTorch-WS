import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from models import Autoencoder  # Ensure this is the correct import path

# Device configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the trained model
model = Autoencoder().to(device)
model.load_state_dict(torch.load(r'Projects\models\autoencoder_mnist.pth'))
model.eval()  # Set the model to evaluation mode

# Define the transformation and load the MNIST test dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
])

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=10, shuffle=False)

# Function to introduce missing pixels
def add_missing_pixels(images, missing_fraction=0.2):
    images = images.clone()  # Clone to avoid modifying the original data
    num_pixels = images.size(2) * images.size(3)  # Total number of pixels
    num_missing = int(missing_fraction * num_pixels)  # Number of pixels to remove

    for i in range(images.size(0)):
        # Flatten the image to randomly choose which pixels to remove
        flattened = images[i].view(-1)
        missing_indices = torch.randperm(num_pixels)[:num_missing]  # Random indices to zero out
        flattened[missing_indices] = 0
        images[i] = flattened.view(images[i].size())  # Reshape back to original

    return images

# Display original, corrupted, and reconstructed images
def show_images(original, corrupted, reconstructed):
    original = original.cpu().numpy()
    corrupted = corrupted.cpu().numpy()
    reconstructed = reconstructed.cpu().numpy()

    fig, axes = plt.subplots(3, 5, figsize=(12, 6))
    for i in range(5):
        axes[0, i].imshow(original[i].reshape(28, 28), cmap='gray')
        axes[1, i].imshow(corrupted[i].reshape(28, 28), cmap='gray')
        axes[2, i].imshow(reconstructed[i].reshape(28, 28), cmap='gray')
        axes[0, i].set_title("Original")
        axes[1, i].set_title("Corrupted")
        axes[2, i].set_title("Reconstructed")
        for ax in axes[:, i]:
            ax.axis('off')
    plt.show()

# Evaluate the model on corrupted test data
with torch.no_grad():  # Disable gradient calculation
    test_images, _ = next(iter(testloader))
    test_images = test_images.to(device)
    
    corrupted_images = add_missing_pixels(test_images, missing_fraction=0.25)  # Add missing pixels
    
    # Assuming your model returns a tuple, extract the first item
    reconstructed_images = model(corrupted_images)[0]  # Adjust this index if needed
    
    # Reshape reconstructed images to 28x28
    reconstructed_images = reconstructed_images.view(-1, 28, 28)
    
    show_images(test_images[:5], corrupted_images[:5], reconstructed_images[:5])