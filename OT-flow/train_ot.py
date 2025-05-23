# Placeholder for train_ot.py

import torch
import yaml
import os
import numpy as np

from vae_model import VAE # To load VAE architecture
from dataset_loader import get_data_loader, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS
from ot_model import OTMapSolver

def get_all_latent_codes(vae_model_path, vae_config, dataloader, device):
    """
    Loads a trained VAE encoder and computes latent codes for the entire dataset.
    Args:
        vae_model_path (str): Path to the trained VAE model (.pth file).
        vae_config (dict): VAE configuration dictionary.
        dataloader (DataLoader): DataLoader for the dataset.
        device (torch.device): Device to use.
    Returns:
        Tensor: A tensor containing all latent codes (mu values from encoder).
    """
    print(f"Loading VAE model from: {vae_model_path}")
    model = VAE(
        image_channels=vae_config.get('channels', IMAGE_CHANNELS),
        latent_dim=vae_config['latent_dim'],
        image_height=vae_config['image_size'][0],
        image_width=vae_config['image_size'][1]
    ).to(device)
    
    model.load_state_dict(torch.load(vae_model_path, map_location=device))
    model.eval()
    print("VAE model loaded successfully.")

    all_latent_mu = []
    print("Extracting latent codes from the dataset...")
    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):
            data = data.to(device)
            mu, _ = model.encode(data) # We only need mu for the target distribution
            all_latent_mu.append(mu.cpu()) # Collect on CPU to save GPU memory
            if batch_idx % 200 == 0:
                print(f"  Processed batch {batch_idx}/{len(dataloader)}")
    
    concatenated_latents = torch.cat(all_latent_mu, dim=0)
    print(f"Extracted {concatenated_latents.shape[0]} latent codes of dimension {concatenated_latents.shape[1]}.")
    return concatenated_latents # Shape: (num_dataset_samples, latent_dim)

def train_ot_map(config_path="configs/vae_config.yaml"):
    # --- Configuration --- #
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- VAE and Data Config --- #
    vae_conf = config # Assuming VAE configs are at the top level
    ot_conf = config.get('ot_map_params', {})
    vae_encoder_path = config.get('vae_encoder_path')
    ot_solver_save_dir_config = config.get('ot_solver_save_dir', "OT-flow/saved_models/ot_solver_default/")

    if not vae_encoder_path or not os.path.exists(vae_encoder_path):
        print(f"Error: VAE encoder path '{vae_encoder_path}' not found or not specified in config.")
        print("Please train the VAE first and update 'vae_encoder_path' in the config file.")
        # For a dry run without a trained VAE, we could generate random latents, but that defeats the purpose.
        # We'll try to create a dummy VAE model file for testing if it doesn't exist.
        print("Attempting to create a dummy VAE model for dry run... (This will not yield meaningful OT results)")
        dummy_vae_dir = os.path.dirname(vae_encoder_path or "OT-flow/saved_models/vae_dummy/dummy.pth")
        os.makedirs(dummy_vae_dir, exist_ok=True)
        dummy_vae_path = os.path.join(dummy_vae_dir, "dummy_vae_epoch_1.pth")
        
        temp_model = VAE(
            image_channels=vae_conf.get('channels', IMAGE_CHANNELS),
            latent_dim=vae_conf['latent_dim'],
            image_height=vae_conf['image_size'][0],
            image_width=vae_conf['image_size'][1]
        )
        torch.save(temp_model.state_dict(), dummy_vae_path)
        vae_encoder_path = dummy_vae_path # Use dummy for this run
        print(f"Using DUMMY VAE model at: {vae_encoder_path}")
        # return # Exit if no real VAE model

    # --- Data Loading (only needed to get all latent codes) --- #
    print("Loading data to generate latent codes...")
    dataset_root_path = vae_conf.get("dataset_root_path", "/home/cs/OT-diffusion-flow/dataset/data")
    # Use a potentially larger batch size for faster latent code extraction if GPU memory allows
    latent_extraction_batch_size = vae_conf.get('batch_size', 64) * 2 
    dataloader_for_latents = get_data_loader(
        dataset_path=dataset_root_path,
        batch_size=latent_extraction_batch_size, 
        shuffle=False, # No need to shuffle for latent extraction
        num_workers=4
    )

    # --- Get Target Latent Codes (P_j) --- # 
    target_latent_codes_P = get_all_latent_codes(vae_encoder_path, vae_conf, dataloader_for_latents, device)
    target_latent_codes_P = target_latent_codes_P.to(device) # Ensure it's on the correct device for OTMapSolver

    # --- Initialize and Train OTMapSolver --- #
    noise_dim = vae_conf['latent_dim'] # Source noise dim must match target latent dim for this OT setup
    print(f"Initializing OTMapSolver with {target_latent_codes_P.shape[0]} target latents of dim {noise_dim}.")
    
    ot_solver = OTMapSolver(
        target_latent_codes_P=target_latent_codes_P,
        noise_dim=noise_dim,
        pyomt_params=ot_conf, 
        device=device
    )

    print("Starting OT potentials training...")
    ot_solver.train_potentials(solver_save_dir_for_intermediate=ot_solver_save_dir_config)

    # --- Save OT Solver State --- # 
    # The main solver state (h_P, final d_h, params) is saved here.
    # Intermediate d_h are saved directly by train_potentials method.
    final_solver_state_path = os.path.join(ot_solver_save_dir_config, "ot_solver_state_final.pth")
    os.makedirs(os.path.dirname(final_solver_state_path), exist_ok=True) # Ensure dir exists
    ot_solver.save_solver_state(final_solver_state_path)
    print(f"OT training finished. Final solver state saved to {final_solver_state_path}")

if __name__ == "__main__":
    # For a quick test, we need a config file. Let's use the main one.
    # We also need a VAE model. The script tries to create a dummy one if the specified one isn't found.
    config_f_path = "OT-flow/configs/vae_config.yaml"
    
    # Update config for a quick test run of OT training (if vae_encoder_path is dummy)
    try:
        with open(config_f_path, 'r') as f_read:
            temp_config_data = yaml.safe_load(f_read)
    except FileNotFoundError:
        print(f"Config file {config_f_path} not found. Cannot run test.")
        exit()
        
    # Reduce iterations for quick test if we are likely using a dummy VAE
    # or just want a fast OT training check
    original_ot_max_iter = temp_config_data.get('ot_map_params', {}).get('max_iter', 20000)
    temp_config_data.setdefault('ot_map_params', {})['max_iter'] = 500 # Drastically reduce for test
    temp_config_data.setdefault('ot_map_params', {})['lr'] = 0.01 
    temp_config_data.setdefault('ot_map_params', {})['save_iter_freq'] = 100 # Save intermediate d_h frequently for test
    temp_config_data['vae_encoder_path'] = temp_config_data.get('vae_encoder_path', "OT-flow/saved_models/vae_dummy/dummy_vae_epoch_1.pth")
    # Adjust for save directory structure in test
    temp_config_data['ot_solver_save_dir'] = "OT-flow/saved_models/ot_solver_test_run/"

    temp_config_file_for_ot_test = "OT-flow/configs/temp_ot_test_config.yaml"
    with open(temp_config_file_for_ot_test, 'w') as f_temp_write:
        yaml.dump(temp_config_data, f_temp_write, sort_keys=False)
    
    print(f"--- Running OT Training Test with reduced iterations (max_iter=500) using config {temp_config_file_for_ot_test} ---")
    try:
        train_ot_map(config_path=temp_config_file_for_ot_test)
    except Exception as e:
        print(f"An error occurred during OT training test: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if os.path.exists(temp_config_file_for_ot_test):
            os.remove(temp_config_file_for_ot_test)
        # Clean up dummy VAE if it was created by this script run
        dummy_vae_path_check = temp_config_data['vae_encoder_path']
        if "dummy_vae_epoch_1.pth" in dummy_vae_path_check and os.path.exists(dummy_vae_path_check):
            print(f"Cleaning up dummy VAE model: {dummy_vae_path_check}")
            os.remove(dummy_vae_path_check)
            dummy_vae_dir_check = os.path.dirname(dummy_vae_path_check)
            if not os.listdir(dummy_vae_dir_check): # Remove dir if empty
                 os.rmdir(dummy_vae_dir_check)
        # (The OT training script itself would create a more specific dummy if run)
        dummy_ot_dir_check_main = temp_config_data['ot_solver_save_dir']
        if "ot_solver_test_run" in dummy_ot_dir_check_main and os.path.exists(dummy_ot_dir_check_main):
            print(f"Cleaning up dummy OT solver directory: {dummy_ot_dir_check_main}")
            import shutil
            shutil.rmtree(dummy_ot_dir_check_main) # Remove the whole test directory 