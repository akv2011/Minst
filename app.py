import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(
    page_title="MNIST cGAN Generator",
    page_icon="🎨",
    layout="wide"
)

# Generator class
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
    device = torch.device('cpu')
    
    try:
        checkpoint = torch.load('cgan_mnist.pth', map_location=device)
        latent_dim = checkpoint['latent_dim']
        num_classes = checkpoint['num_classes']
        
        generator = Generator(latent_dim, num_classes)
        generator.load_state_dict(checkpoint['generator_state_dict'])
        generator.eval()
        
        return generator, latent_dim, num_classes
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

def generate_image(generator, latent_dim, digit, device='cpu'):
    """Generate a single image"""
    with torch.no_grad():
        noise = torch.randn(1, latent_dim)
        label = torch.tensor([digit])
        fake_img = generator(noise, label)
        
        img_array = fake_img.squeeze().cpu().numpy()
        img_array = (img_array + 1) / 2  # Denormalize
        img_array = (img_array * 255).astype(np.uint8)
        
        pil_image = Image.fromarray(img_array, mode='L')
        return pil_image

def main():
    st.title("🎨 MNIST Conditional GAN Generator")
    st.markdown("Generate handwritten digits using a trained Conditional GAN!")
    
    # Load model
    generator, latent_dim, num_classes = load_generator()
    
    if generator is None:
        st.error("❌ Could not load model. Make sure 'cgan_mnist.pth' exists.")
        st.stop()
    
    st.success("✅ Model loaded successfully!")
    
    # Main controls (not in sidebar - make them more visible)
    st.markdown("---")
    st.header("🎛️ Generation Controls")
    
    col1, col2, col3 = st.columns([2, 2, 2])
    
    with col1:
        selected_digit = st.selectbox(
            "**Select digit to generate:**",
            options=list(range(10)),
            index=0
        )
    
    with col2:
        num_samples = st.slider(
            "**Number of samples:**",
            min_value=1,
            max_value=10,
            value=5
        )
    
    with col3:
        st.write("")  # spacing
        st.write("")  # spacing
        generate_button = st.button("🎲 **Generate Images**", type="primary")
    
    st.markdown("---")
    
    # Show current selection
    st.info(f"Ready to generate **{num_samples}** samples of digit **{selected_digit}**")
    
    # Generate images
    if generate_button:
        st.header(f"🖼️ Generated Images - Digit {selected_digit}")
        
        with st.spinner(f"Generating {num_samples} images..."):
            # Generate images
            generated_images = []
            for i in range(num_samples):
                img = generate_image(generator, latent_dim, selected_digit)
                generated_images.append(img)
        
        # Display images
        cols = st.columns(5)
        for i, img in enumerate(generated_images):
            col_idx = i % 5
            with cols[col_idx]:
                st.image(img, caption=f"Sample {i+1}", width=120)
        
        st.success(f"✅ Successfully generated {num_samples} images of digit {selected_digit}!")
    
    # Quick generate all digits
    st.markdown("---")
    if st.button("🎯 **Generate All Digits (0-9)**", type="secondary"):
        st.header("🔢 All Digits Generated")
        
        with st.spinner("Generating all digits..."):
            cols = st.columns(5)
            
            for digit in range(10):
                img = generate_image(generator, latent_dim, digit)
                col_idx = digit % 5
                with cols[col_idx]:
                    st.image(img, caption=f"Digit {digit}", width=100)
        
        st.success("✅ All digits generated!")
    
    # Instructions
    st.markdown("---")
    st.markdown("### 📝 Instructions")
    st.markdown("""
    1. **Select a digit** (0-9) from the dropdown
    2. **Choose number of samples** (1-10) 
    3. **Click 'Generate Images'** to create samples
    4. **Or click 'Generate All Digits'** to see one of each digit
    """)

if __name__ == "__main__":
    main()
