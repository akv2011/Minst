import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Data preprocessing
transform = T.Compose([
    T.ToTensor(),
    T.Normalize((0.5,), (0.5,)) 
])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

print(f"Train Loader size: {len(train_dataset)}")

# Fixed Generator class
class Generator(nn.Module):
    def __init__(self, latent_dim=100, num_classes=10):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.model = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 784),
            nn.Tanh()
        )
    
    def forward(self, noise, labels):
        labels = self.label_emb(labels)
        gen_input = torch.cat((noise, labels), dim=1)
        img = self.model(gen_input)
        img = img.view(img.size(0), 1, 28, 28)
        return img

# Fixed Discriminator class
class Discriminator(nn.Module):
    def __init__(self, num_classes=10):
        super(Discriminator, self).__init__()
        self.num_classes = num_classes
        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.model = nn.Sequential(
            nn.Linear(28*28 + num_classes, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, img, labels):
        labels = self.label_emb(labels)
        img_flat = img.view(img.size(0), -1)
        d_input = torch.cat((img_flat, labels), dim=1)
        validity = self.model(d_input)
        return validity

# Model initialization
latent_dim = 100
num_classes = 10
generator = Generator(latent_dim, num_classes).to(device)
discriminator = Discriminator(num_classes).to(device)

# Loss function and optimizers
criterion = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

print("Model initialized")

def generate_samples(generator, latent_dim, num_classes, device, epoch):
    """Generate one sample from each digit class"""
    generator.eval()
    with torch.no_grad():
        # Generate one sample for each digit (0-9)
        labels = torch.arange(0, num_classes).to(device)
        noise = torch.randn(num_classes, latent_dim).to(device)
        fake_imgs = generator(noise, labels)
        
        # Plot generated samples
        fig, axes = plt.subplots(1, 10, figsize=(12, 2))
        for i in range(num_classes):
            img = fake_imgs[i].cpu().numpy().squeeze()
            img = (img + 1) / 2  # Denormalize from [-1,1] to [0,1]
            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(f'Digit {i}')
            axes[i].axis('off')
        
        plt.suptitle(f'Generated Samples - Epoch {epoch}')
        plt.tight_layout()
        plt.savefig(f'generated_samples_epoch_{epoch}.png')
        plt.show()
    
    generator.train()

def train_gan(epochs=50):
    generator.train()
    discriminator.train()
    
    for epoch in range(epochs):
        d_loss_total = 0
        g_loss_total = 0
        
        for i, (imgs, labels) in enumerate(train_loader):
            batch_size = imgs.size(0)
            real_imgs = imgs.to(device)
            labels = labels.to(device)
            
            # Real and fake labels
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)
            
            # Train Discriminator
            optimizer_D.zero_grad()
            
            # Loss for real images
            real_loss = criterion(discriminator(real_imgs, labels), real_labels)
            
            # Generate fake images
            noise = torch.randn(batch_size, latent_dim).to(device)
            fake_imgs = generator(noise, labels)
            
            # Loss for fake images
            fake_loss = criterion(discriminator(fake_imgs.detach(), labels), fake_labels)
            
            # Total discriminator loss
            d_loss = real_loss + fake_loss
            d_loss.backward()
            optimizer_D.step()
            
            # Train Generator
            optimizer_G.zero_grad()
            
            # Generate new fake images
            noise = torch.randn(batch_size, latent_dim).to(device)
            fake_imgs = generator(noise, labels)
            
            # Generator loss (trying to fool discriminator)
            g_loss = criterion(discriminator(fake_imgs, labels), real_labels)
            g_loss.backward()
            optimizer_G.step()
            
            d_loss_total += d_loss.item()
            g_loss_total += g_loss.item()
        
        # Print training progress
        avg_d_loss = d_loss_total / len(train_loader)
        avg_g_loss = g_loss_total / len(train_loader)
        print(f'Epoch [{epoch+1}/{epochs}] | D Loss: {avg_d_loss:.4f} | G Loss: {avg_g_loss:.4f}')
        
        # Show generated samples every 10 epochs
        if (epoch + 1) % 10 == 0:
            generate_samples(generator, latent_dim, num_classes, device, epoch + 1)

# Start training
print("Starting GAN training...")
train_gan(epochs=50)

# Save trained models
os.makedirs('models', exist_ok=True)
torch.save({
    'generator_state_dict': generator.state_dict(),
    'discriminator_state_dict': discriminator.state_dict(),
    'latent_dim': latent_dim,
    'num_classes': num_classes
}, 'models/cgan_mnist.pth')

print("Training completed and models saved!")
