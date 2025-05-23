# Placeholder for train_vae.py

import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision.utils import save_image
import yaml
import os
import datetime

from vae_model import VAE
from dataset_loader import get_data_loader, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS

# Loss function for VAE
def vae_loss_function(x_reconstructed, x, mu, logvar, beta=1.0):
    """Computes the VAE loss: Reconstruction Loss + beta * KL Divergence."""
    reconstruction_loss = F.mse_loss(x_reconstructed, x, reduction='sum') / x.size(0) # Per-sample MSE
    # KL divergence: 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    # KLD = -0.5 * sum(1 + logvar - mu.pow(2) - logvar.exp())
    # Correct KLD calculation: sum over latent dimensions, average over batch
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    kld = torch.mean(kld) # Average KLD over the batch

    total_loss = reconstruction_loss + beta * kld
    return total_loss, reconstruction_loss, kld

def train_vae(config_path="configs/vae_config.yaml"):
    # --- Configuration --- #
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create directories for logs and saved models
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = f"OT-flow/logs/vae_training_{timestamp}"
    model_save_dir = f"OT-flow/saved_models/vae_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_save_dir, exist_ok=True)

    # --- Data Loading --- #
    # Using IMAGE_HEIGHT, IMAGE_WIDTH from dataset_loader for consistency, though config also has it.
    print("Loading data...")
    # The dataset path should be the directory containing the parquet files, e.g., /home/cs/OT-diffusion-flow/dataset/data
    # This needs to be an absolute path or correctly relative from where the script is run.
    # For now, let's assume it's defined or easily configurable.
    # TODO: Make dataset_root_path configurable, perhaps via config.yaml or command-line arg.
    dataset_root_path = config.get("dataset_root_path", "/home/cs/OT-diffusion-flow/dataset/data") 
    dataloader = get_data_loader(
        dataset_path=dataset_root_path,
        batch_size=config['batch_size'],
        num_workers=4 # Can be adjusted
    )
    print(f"Data loaded. Number of batches: {len(dataloader)}")

    # --- Model, Optimizer, Loss --- #
    model = VAE(
        image_channels=config.get('channels', IMAGE_CHANNELS), # Use config or fallback to loader constants
        latent_dim=config['latent_dim'],
        image_height=config['image_size'][0], # H
        image_width=config['image_size'][1]   # W
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # Beta for KL divergence term (can be scheduled, e.g., for beta-VAE)
    beta_kld = config.get('beta_kld', 1.0) 

    print("Starting training...")
    for epoch in range(config['epochs']):
        model.train()
        total_train_loss = 0
        total_recon_loss = 0
        total_kld_loss = 0

        for batch_idx, data in enumerate(dataloader):
            data = data.to(device)
            optimizer.zero_grad()
            
            reconstructed_data, mu, logvar = model(data)
            loss, recon_loss, kld = vae_loss_function(reconstructed_data, data, mu, logvar, beta=beta_kld)
            
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kld_loss += kld.item()

            if batch_idx % 100 == 0: # Log every 100 batches
                print(f"Epoch [{epoch+1}/{config['epochs']}] Batch [{batch_idx}/{len(dataloader)}] Avg Loss: {loss.item() / data.size(0):.4f} (Recon: {recon_loss.item() / data.size(0):.4f}, KLD: {kld.item() / data.size(0):.4f})")
        
        avg_epoch_loss = total_train_loss / len(dataloader.dataset)
        avg_recon_loss = total_recon_loss / len(dataloader.dataset)
        avg_kld_loss = total_kld_loss / len(dataloader.dataset)
        print(f"====> Epoch {epoch+1} completed. Average Loss: {avg_epoch_loss:.4f} (Recon: {avg_recon_loss:.4f}, KLD: {avg_kld_loss:.4f})")

        # Save model checkpoint
        if (epoch + 1) % config.get('save_epoch_freq', 10) == 0 or epoch == config['epochs'] - 1:
            checkpoint_path = os.path.join(model_save_dir, f"vae_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved model checkpoint to {checkpoint_path}")

            # Save some reconstructed images
            model.eval()
            with torch.no_grad():
                # Get a fixed batch of data for consistent visualization (e.g., first batch)
                sample_data = next(iter(dataloader)).to(device)
                if sample_data.size(0) > 8: # Take at most 8 images for visualization
                    sample_data = sample_data[:8]
                
                reconstructed_sample, _, _ = model(sample_data)
                
                # Denormalize images from [-1, 1] to [0, 1] for saving
                sample_data_denorm = (sample_data * 0.5) + 0.5
                reconstructed_sample_denorm = (reconstructed_sample * 0.5) + 0.5
                
                comparison = torch.cat([sample_data_denorm, reconstructed_sample_denorm])
                save_image(comparison.cpu(), 
                           os.path.join(log_dir, f'reconstruction_epoch_{epoch+1}.png'), 
                           nrow=sample_data.size(0))
                print(f"Saved sample reconstructions to {log_dir}")

    print("Training finished.")
    final_model_path = os.path.join(model_save_dir, "vae_final.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Saved final model to {final_model_path}")

if __name__ == "__main__":
    # Add a default config if vae_config.yaml doesn't exist or is minimal
    default_config = {
        'dataset_root_path': "/home/cs/OT-diffusion-flow/dataset/data",
        'image_size': [IMAGE_HEIGHT, IMAGE_WIDTH],
        'channels': IMAGE_CHANNELS,
        'latent_dim': 100,
        'batch_size': 64,
        'learning_rate': 0.001,
        'epochs': 10, # Reduced for quick test, actual should be from config (e.g., 50)
        'beta_kld': 1.0,
        'save_epoch_freq': 2 # Save more frequently for testing
    }
    config_file_path = "OT-flow/configs/vae_config.yaml"
    
    # Ensure the config file exists with some defaults if not present
    if not os.path.exists(config_file_path):
        print(f"Config file {config_file_path} not found. Using default settings for a quick test.")
        # If you want to create it with defaults:
        # os.makedirs(os.path.dirname(config_file_path), exist_ok=True)
        # with open(config_file_path, 'w') as f_cfg:
        #     yaml.dump(default_config, f_cfg, sort_keys=False)
        # For now, just use default_config directly if file not found
        # This part is more for robust execution than direct training logic
        # For this run, we expect vae_config.yaml to exist. If not, it will use the hardcoded one.

    try:
        train_vae(config_path=config_file_path)
    except FileNotFoundError:
        print(f"Running with internally defined default config as {config_file_path} was not found.")
        # Create a temporary config dict to pass to train_vae if file not found
        # This part would typically ensure that config is loaded correctly.
        # For this first run, it's better to ensure vae_config.yaml is there.
        # Let's assume it is, and if not, the script will fail gracefully or use the one in the repo.
        # The current train_vae function expects config_path to be valid.
        # If we want to make it more robust for a quick test without the file:
        temp_config_file = "OT-flow/configs/temp_vae_config_for_test.yaml"
        print(f"Config file {config_file_path} not found. Creating a temporary one for testing: {temp_config_file}")
        os.makedirs(os.path.dirname(temp_config_file), exist_ok=True)
        with open(temp_config_file, 'w') as f_cfg:
            yaml.dump(default_config, f_cfg, sort_keys=False)
        train_vae(config_path=temp_config_file)
        os.remove(temp_config_file) # Clean up temp file
    except Exception as e:
        print(f"An error occurred during VAE training: {e}")
        import traceback
        traceback.print_exc() 