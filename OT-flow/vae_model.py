# Placeholder for vae_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, image_channels=3, h_dim=None, latent_dim=100, image_height=218, image_width=178):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.image_channels = image_channels
        self.image_height = image_height
        self.image_width = image_width

        # Calculate a reasonable h_dim if not provided
        # This h_dim is the flattened size after convolutions
        # Let's define a convolutional base first to determine this dynamically

        # Encoder
        self.enc_conv1 = nn.Conv2d(image_channels, 32, kernel_size=4, stride=2, padding=1)  # Output: 32 x H/2 x W/2
        self.enc_conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)             # Output: 64 x H/4 x W/4
        self.enc_conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)            # Output: 128 x H/8 x W/8
        self.enc_conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)           # Output: 256 x H/16 x W/16
        
        # Calculate the size of the feature map after conv layers
        self.final_h = image_height // 16
        self.final_w = image_width // 16
        self.h_dim = 256 * self.final_h * self.final_w

        self.fc_mu = nn.Linear(self.h_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.h_dim, latent_dim)

        # Decoder
        self.dec_fc1 = nn.Linear(latent_dim, self.h_dim)
        
        # Reshape in decoder will be to (256, self.final_h, self.final_w)
        self.dec_conv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, output_padding=(1,0)) # Target H_out: final_h*2+1 if odd, final_h*2 if even -> 27x22
        self.dec_conv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)  # Target H_out: 54x44
        self.dec_conv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, output_padding=(1,1))   # Target H_out: 109x89
        self.dec_conv4 = nn.ConvTranspose2d(32, image_channels, kernel_size=4, stride=2, padding=1) # Target H_out: 218x178

    def encode(self, x):
        x = F.relu(self.enc_conv1(x))
        x = F.relu(self.enc_conv2(x))
        x = F.relu(self.enc_conv3(x))
        x = F.relu(self.enc_conv4(x))
        x = x.view(x.size(0), -1)  # Flatten
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = F.relu(self.dec_fc1(z))
        x = x.view(x.size(0), 256, self.final_h, self.final_w) # Reshape to feature map size
        x = F.relu(self.dec_conv1(x))
        x = F.relu(self.dec_conv2(x))
        x = F.relu(self.dec_conv3(x))
        x_reconstructed = torch.tanh(self.dec_conv4(x)) # Tanh to match input range [-1, 1]
        return x_reconstructed

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_reconstructed = self.decode(z)
        return x_reconstructed, mu, logvar

if __name__ == '__main__':
    # Test the VAE model
    # Get parameters from config (or use defaults for testing)
    img_channels = 3
    latent_dim_test = 100
    img_h = 218
    img_w = 178

    vae_model = VAE(image_channels=img_channels, latent_dim=latent_dim_test, image_height=img_h, image_width=img_w)
    print(vae_model)

    # Create a dummy input tensor
    batch_size = 4
    dummy_input = torch.randn(batch_size, img_channels, img_h, img_w)
    print(f"\nInput shape: {dummy_input.shape}")

    try:
        x_reconstructed, mu, logvar = vae_model(dummy_input)
        print(f"Reconstructed shape: {x_reconstructed.shape}")
        print(f"Mu shape: {mu.shape}")
        print(f"Logvar shape: {logvar.shape}")

        # Check output range of reconstructed image (should be approx -1 to 1 due to Tanh)
        print(f"Reconstructed min/max: {x_reconstructed.min().item()}/{x_reconstructed.max().item()}")
        assert x_reconstructed.shape == dummy_input.shape, "Reconstructed shape mismatch!"
        assert mu.shape == (batch_size, latent_dim_test), "Mu shape mismatch!"
        assert logvar.shape == (batch_size, latent_dim_test), "Logvar shape mismatch!"
        print("\nVAE model test passed!")
    except Exception as e:
        print(f"\nError during VAE model test: {e}")
        import traceback
        traceback.print_exc()