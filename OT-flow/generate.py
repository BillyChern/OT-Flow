# Placeholder for generate.py

import torch
import yaml
import os
import datetime
from torchvision.utils import save_image
from torchdiffeq import odeint

from vae_model import VAE
from ot_model import OTMapSolver
from flow_matching_model import VelocityFieldModel
from dataset_loader import IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS # For VAE config if needed

def generate_samples(config_path="configs/vae_config.yaml", num_samples=16, output_dir="OT-flow/generated_images"):
    # --- Configuration --- #
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load Models --- #
    vae_conf = config
    ot_solver_save_dir_config = config.get('ot_solver_save_dir')
    fm_conf = config.get('flow_matching_params', {})
    flow_model_path = config.get('flow_model_save_path')
    # VAE path for decoder (usually the same as encoder path unless split)
    vae_model_path = config.get('vae_encoder_path') # Assuming full VAE model is saved here

    # 1. Load VAE (for decoder)
    if not vae_model_path or not os.path.exists(vae_model_path):
        print(f"Error: VAE model path '{vae_model_path}' not found. Train VAE first.")
        # Create a dummy VAE for dry run
        print("Creating DUMMY VAE for generation dry run...")
        dummy_vae_dir = os.path.dirname(vae_model_path or "OT-flow/saved_models/vae_dummy/dummy.pth")
        os.makedirs(dummy_vae_dir, exist_ok=True)
        vae_model_path = os.path.join(dummy_vae_dir, "dummy_vae_epoch_1.pth")
        if not os.path.exists(vae_model_path):
            temp_vae = VAE(image_channels=vae_conf.get('channels', IMAGE_CHANNELS), latent_dim=vae_conf['latent_dim'], image_height=vae_conf['image_size'][0], image_width=vae_conf['image_size'][1])
            torch.save(temp_vae.state_dict(), vae_model_path)
        print(f"Using DUMMY VAE from {vae_model_path}")

    vae = VAE(
        image_channels=vae_conf.get('channels', IMAGE_CHANNELS),
        latent_dim=vae_conf['latent_dim'],
        image_height=vae_conf['image_size'][0],
        image_width=vae_conf['image_size'][1]
    ).to(device)
    vae.load_state_dict(torch.load(vae_model_path, map_location=device))
    vae.eval()
    print(f"VAE model loaded from {vae_model_path}")

    # 2. Load OTMapSolver
    if not ot_solver_save_dir_config:
        print("Error: 'ot_solver_save_dir' not specified in config for OTMapSolver.")
        # Create a dummy OT solver state for generation dry run...
        # This part should ideally ensure a dummy file is created if needed by subsequent logic
        # For now, assume the test setup handles this or paths are correct.
        return # Cannot proceed without a path/dummy path
    
    ot_solver_final_state_path = os.path.join(ot_solver_save_dir_config, "ot_solver_state_final.pth")

    if not os.path.exists(ot_solver_final_state_path):
        print(f"Error: OTMapSolver final state '{ot_solver_final_state_path}' not found. Train OT map first.")
        print("Creating DUMMY OT solver state for generation dry run...")
        os.makedirs(ot_solver_save_dir_config, exist_ok=True) # Ensure dir exists
        # dummy_ot_dir = os.path.dirname(ot_solver_path or "OT-flow/saved_models/dummy_ot.pth")
        # os.makedirs(dummy_ot_dir, exist_ok=True)
        # ot_solver_path = os.path.join(dummy_ot_dir, "dummy_ot_solver_state.pth")
        if not os.path.exists(ot_solver_final_state_path): # Double check before creating dummy
            num_dummy_latents = 100 # Small number for dummy h_P
            dummy_h_P = torch.randn(num_dummy_latents, vae_conf['latent_dim'], dtype=torch.float32)
            dummy_d_h = torch.zeros(num_dummy_latents, 1, dtype=torch.float64)
            dummy_ot_state = {'h_P': dummy_h_P, 'd_h': dummy_d_h, 'dim_noise': vae_conf['latent_dim'], 'params': config.get('ot_map_params', {})}
            torch.save(dummy_ot_state, ot_solver_final_state_path)
        print(f"Using DUMMY OT solver state from {ot_solver_final_state_path}")
        
    ot_solver = OTMapSolver.load_solver_state(ot_solver_final_state_path, device=device)
    print(f"OTMapSolver loaded from {ot_solver_final_state_path}")

    # 3. Load Flow Matching Model (VelocityFieldModel)
    if not flow_model_path or not os.path.exists(flow_model_path):
        print(f"Error: Flow model path '{flow_model_path}' not found. Train Flow Matching model first.")
        print("Creating DUMMY Flow model for generation dry run...")
        dummy_fm_dir = os.path.dirname(flow_model_path or "OT-flow/saved_models/dummy_fm.pth")
        os.makedirs(dummy_fm_dir, exist_ok=True)
        flow_model_path = os.path.join(dummy_fm_dir, "dummy_flow_model.pth")
        if not os.path.exists(flow_model_path):
            temp_fm_model = VelocityFieldModel(latent_dim=vae_conf['latent_dim'], time_embed_dim=fm_conf.get('time_embed_dim',128), hidden_dims=fm_conf.get('hidden_dims', [128]))
            torch.save(temp_fm_model.state_dict(), flow_model_path)
        print(f"Using DUMMY Flow model from {flow_model_path}")

    flow_model = VelocityFieldModel(
        latent_dim=vae_conf['latent_dim'],
        time_embed_dim=fm_conf.get('time_embed_dim', 128),
        hidden_dims=fm_conf.get('hidden_dims', [512, 512, 512])
    ).to(device)
    flow_model.load_state_dict(torch.load(flow_model_path, map_location=device))
    flow_model.eval()
    print(f"Flow Matching model loaded from {flow_model_path}")

    # --- Generation Process --- #
    os.makedirs(output_dir, exist_ok=True)
    print(f"Generating {num_samples} samples...")

    with torch.no_grad():
        # 1. Sample initial noise z_noise ~ N(0,I)
        z_noise = torch.randn(num_samples, vae_conf['latent_dim'], dtype=torch.float32, device=device)
        
        # 2. Get initial latent estimate z_ot from OTMapSolver
        z_ot = ot_solver.map_noise(z_noise) # This is z(0) for the ODE, already float32 and on device
        print(f"Generated z_ot (initial for ODE): shape {z_ot.shape}")

        # 3. Define the ODE function for torchdiffeq
        # The model v_psi(z, t) directly gives dz/dt
        # odeint expects func(t, z), so we need a wrapper
        def ode_func(t, z):
            # t is a scalar here, repeat for batch
            t_batch = t.expand(z.size(0)) 
            return flow_model(z, t_batch)

        # 4. Solve the ODE from t=0 to t=1
        print("Solving ODE for flow matching...")
        # Time points for integration: from t_start (OT output time, effectively 0) to t_end (target time, 1)
        t_span = torch.tensor([0.0, 1.0], device=device)
        
        # z_refined will be a tensor of shape (num_time_points, batch_size, latent_dim)
        # We only need the final time point: z(1)
        # Using default solver (dopri5), can specify method e.g. method='rk4', options={'step_size': 0.1}
        solution_path = odeint(ode_func, z_ot, t_span, rtol=1e-5, atol=1e-5)
        z_refined = solution_path[-1] # Get z(t=1)
        print(f"Generated z_refined (ODE output at t=1): shape {z_refined.shape}")

        # 5. Pass refined latent code through VAE decoder
        generated_images = vae.decode(z_refined)
        print(f"Generated images from VAE decoder: shape {generated_images.shape}")

        # Denormalize images from [-1, 1] to [0, 1] for saving
        generated_images_denorm = (generated_images * 0.5) + 0.5
        
        # Save generated images
        timestamp_gen = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        for i in range(num_samples):
            save_path = os.path.join(output_dir, f"generated_sample_{timestamp_gen}_{i+1:03d}.png")
            save_image(generated_images_denorm[i].cpu(), save_path)
        print(f"Saved {num_samples} generated images to {output_dir}")

if __name__ == "__main__":
    # Use the main config, but generation script might need specific paths from trained models.
    # For testing, it will create dummy models if real ones aren't found at specified paths.
    config_file = "OT-flow/configs/vae_config.yaml"
    num_gen_samples = 8 # Generate a few samples for test
    output_images_dir = f"OT-flow/generated_images_test_run_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"

    print(f"--- Running Generation Test --- ")
    print(f"Config: {config_file}")
    print(f"Num samples: {num_gen_samples}")
    print(f"Output dir: {output_images_dir}")
    
    try:
        generate_samples(config_path=config_file, num_samples=num_gen_samples, output_dir=output_images_dir)
    except Exception as e:
        print(f"An error occurred during generation test: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up dummy models if they were created by this script run
        # This requires checking the paths used by generate_samples if dummies were made.
        # For simplicity in this test, manual cleanup of dummy files in saved_models if needed.
        print(f"Generation test finished. Check {output_images_dir} for samples.")
        print("Note: If dummy models were used, images will be random-like.") 