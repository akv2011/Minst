import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import io

# Set page config
st.set_page_config(
    page_title="MNIST cGAN Generator",
    page_icon="🎨",
    layout="wide"
)

# Generator class (same as training)
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

@st.cache_resource
def load_generator():
    """Load the trained generator model"""
    device = torch.device('cpu')  # Use CPU for deployment
    
    try:
        # Load the saved model
        checkpoint = torch.load('cgan_mnist.pth', map_location=device)
        latent_dim = checkpoint['latent_dim']
        num_classes = checkpoint['num_classes']
        
        # Initialize generator architecture
        generator = Generator(latent_dim, num_classes)
        
        # Load the generator weights
        generator.load_state_dict(checkpoint['generator_state_dict'])
        generator.eval()
        
        return generator, latent_dim, num_classes
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

def generate_image(generator, latent_dim, digit, device='cpu'):
    """Function to generate image using loaded generator"""
    # Ensure no gradients computed during image generation
    with torch.no_grad():
        # Create random noise vector
        noise = torch.randn(1, latent_dim)
        
        # Process and convert generated tensor to PIL image format
        label = torch.tensor([digit])
        fake_img = generator(noise, label)
        
        # Convert numpy array to PIL image in grayscale
        img_array = fake_img.squeeze().cpu().numpy()
        img_array = (img_array + 1) / 2  # Denormalize from [-1,1] to [0,1]
        img_array = (img_array * 255).astype(np.uint8)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(img_array, mode='L')
        return pil_image

def main():
    """Main Streamlit application function"""
    st.title("🎨 MNIST Conditional GAN Generator")
    st.markdown("Generate handwritten digits using a trained Conditional GAN!")
    
    # Load the generator model
    generator, latent_dim, num_classes = load_generator()
    
    if generator is None:
        st.error("❌ Could not load the trained model. Please ensure 'cgan_mnist.pth' exists in the root directory.")
        st.info("💡 Run the training script first to generate the model file.")
        return
    
    st.success("✅ Model loaded successfully!")
    
    # Sidebar for user controls
    st.sidebar.header("🎛️ Generation Controls")
    
    # Digit selection
    selected_digit = st.sidebar.selectbox(
        "Select digit to generate:",
        options=list(range(10)),
        index=0
    )
    
    # Number of samples to generate
    num_samples = st.sidebar.slider(
        "Number of samples:",
        min_value=1,
        max_value=20,
        value=5
    )
    
    # Generate button
    generate_button = st.sidebar.button("🎲 Generate Images", type="primary")
    
    # Main content area layout
    st.header(f"Generating Digit: {selected_digit}")
    
    if generate_button:
        with st.spinner("Generating images..."):
            # Create columns for displaying images
            cols = st.columns(min(num_samples, 5))
            
            generated_images = []
            for i in range(num_samples):
                # Generate image
                pil_image = generate_image(generator, latent_dim, selected_digit)
                generated_images.append(pil_image)
                
                # Display in appropriate column
                col_idx = i % 5
                with cols[col_idx]:
                    st.image(pil_image, caption=f"Sample {i+1}", width=150)
            
            # Create a grid view of all generated images
            if num_samples > 5:
                st.subheader("All Generated Samples")
                
                # Create matplotlib figure
                fig, axes = plt.subplots(
                    (num_samples - 1) // 5 + 1, 
                    min(num_samples, 5), 
                    figsize=(15, 3 * ((num_samples - 1) // 5 + 1))
                )
                
                if num_samples == 1:
                    axes = [axes]
                elif (num_samples - 1) // 5 + 1 == 1:
                    axes = [axes]
                
                for i, img in enumerate(generated_images):
                    row = i // 5
                    col = i % 5
                    
                    if (num_samples - 1) // 5 + 1 == 1:
                        ax = axes[col] if num_samples > 1 else axes[0]
                    else:
                        ax = axes[row][col]
                    
                    ax.imshow(np.array(img), cmap='gray')
                    ax.set_title(f'Sample {i+1}')
                    ax.axis('off')
                
                # Hide unused subplots
                total_subplots = ((num_samples - 1) // 5 + 1) * 5
                for i in range(num_samples, total_subplots):
                    row = i // 5
                    col = i % 5
                    if (num_samples - 1) // 5 + 1 > 1:
                        axes[row][col].axis('off')
                
                plt.tight_layout()
                st.pyplot(fig)
    
    # Generate all digits button
    st.sidebar.markdown("---")
    if st.sidebar.button("🎯 Generate All Digits (0-9)"):
        st.header("All Digits Generated")
        with st.spinner("Generating all digits..."):
            cols = st.columns(5)
            
            fig, axes = plt.subplots(2, 5, figsize=(15, 6))
            
            for digit in range(10):
                pil_image = generate_image(generator, latent_dim, digit)
                
                # Display in columns
                col_idx = digit % 5
                with cols[col_idx]:
                    st.image(pil_image, caption=f"Digit {digit}", width=120)
                
                # Add to matplotlib figure
                row = digit // 5
                col = digit % 5
                axes[row][col].imshow(np.array(pil_image), cmap='gray')
                axes[row][col].set_title(f'Digit {digit}')
                axes[row][col].axis('off')
            
            plt.tight_layout()
            st.pyplot(fig)
    
    # Information section
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ℹ️ About")
    st.sidebar.info(
        "This app uses a Conditional GAN trained on the MNIST dataset to generate "
        "handwritten digits. You can select specific digits to generate or create "
        "multiple samples of the same digit."
    )

if __name__ == "__main__":
    main()
