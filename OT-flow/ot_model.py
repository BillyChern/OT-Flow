# Placeholder for ot_model.py

import torch
import torch.nn as nn
import torch.optim as optim
import os

# TODO: Implement OT Map model (referencing pyOMT) 

class OTMapSolver:
    def __init__(self, target_latent_codes_P, noise_dim, pyomt_params=None, device=None):
        """
        Initializes the OTMapSolver.
        Args:
            target_latent_codes_P (Tensor): The set of target discrete measures (e.g., VAE latent codes).
                                            Shape (num_P, dim_P), expected to be on the specified device.
            noise_dim (int): Dimensionality of the source noise distribution (e.g., N(0,I)).
            pyomt_params (dict, optional): Parameters for the OT solver, like lr, batch_sizes, iterations.
            device (torch.device, optional): Device to run computations on.
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # pyOMT uses float64 (double precision)
        self.h_P = target_latent_codes_P.clone().double().to(self.device) # Target measures P_j
        self.num_P = self.h_P.shape[0]
        self.dim_P = self.h_P.shape[1] # Dimension of target measures (VAE latent_dim)
        self.dim_noise = noise_dim      # Dimension of source noise

        if self.dim_noise != self.dim_P:
            # The standard cost function c(x,y) = ||x-y||^2 implies x and y have same dimension.
            # If a projection is intended, the cost function or model structure needs to reflect that.
            # For now, assume they must match, as in typical OT map learning for distributions in the same space.
            raise ValueError(f"Noise dimension ({self.dim_noise}) and target latent dimension ({self.dim_P}) must match for current cost function.")

        # Default parameters (inspired by pyOMT demos)
        self.params = {
            'max_iter': 20000,       # Max iterations for optimizing potentials
            'lr': 1e-3,              # Learning rate for Adam (pyOMT demo1 used 1e-4 to 2e-5, demo2 used 5e-2 for OT)
                                     # Let's try a more common Adam LR first.
            'bat_size_n': 1000,      # Batch size for noise samples (bat_X) during training
            'epsilon_cost_stab': 1e-9 # Small epsilon for numerical stability in cost if needed
        }
        if pyomt_params:
            self.params.update(pyomt_params)

        # Potentials on target measures P_j (these are the primary learnable parameters here)
        self.d_h = torch.zeros(self.num_P, 1, dtype=torch.float64, device=self.device)

        # For storing noise batch and its potentials (d_U in pyOMT)
        self.bat_X = None 
        self.d_U = None
        self.solver_save_dir = None # Will be set if intermediate saving is configured
        print(f"OTMapSolver initialized. Target measures: {self.num_P} points of dim {self.dim_P}. Noise dim: {self.dim_noise}. Device: {self.device}")

    def _comp_cost_L2(self, X, Y):
        """Computes squared Euclidean distance cost matrix: c(x_i, y_j) = ||x_i - y_j||^2."""
        # X: (M, D), Y: (N, D) -> Cost: (M, N)
        X_sq = torch.sum(X**2, dim=1, keepdim=True)               # (M, 1)
        Y_sq = torch.sum(Y**2, dim=1, keepdim=True).transpose(0,1) # (1, N)
        XY = torch.matmul(X, Y.transpose(0,1))                   # (M, N)
        cost = X_sq - 2*XY + Y_sq                                # (M, N)
        return cost

    def _sample_noise_batch(self):
        """Samples a batch of noise from N(0,I)."""
        self.bat_X = torch.randn(
            self.params['bat_size_n'], 
            self.dim_noise, 
            dtype=torch.float64, 
            device=self.device
        )

    def _update_source_potentials(self):
        """Calculates d_U = min_j (c(X_i, P_j) - d_h_j), potentials for source samples X_i."""
        cost_matrix = self._comp_cost_L2(self.bat_X, self.h_P) # (bat_size_n, num_P)
        # self.d_h is (num_P, 1)
        cost_minus_h = cost_matrix - self.d_h.transpose(0,1)    # Broadcasting d_h to (1, num_P)
        self.d_U, _ = torch.min(cost_minus_h, dim=1, keepdim=True) # (bat_size_n, 1)

    def train_potentials(self, solver_save_dir_for_intermediate=None):
        """
        Trains the potentials d_h using Adam optimizer on the dual OT objective.
        Args:
            solver_save_dir_for_intermediate (str, optional): Directory to save intermediate d_h potentials.
                                                            If None, intermediate potentials are not saved.
        """
        print(f"Starting OT potentials training for {self.params['max_iter']} iterations...")
        self.d_h.requires_grad_(True)
        optimizer = optim.Adam([self.d_h], lr=self.params['lr'])
        save_iter_freq = self.params.get('save_iter_freq', -1) # Default to -1 if not set (no intermediate save)

        if solver_save_dir_for_intermediate and save_iter_freq > 0:
            os.makedirs(solver_save_dir_for_intermediate, exist_ok=True)
            self.solver_save_dir = solver_save_dir_for_intermediate
            print(f"Intermediate d_h potentials will be saved every {save_iter_freq} iterations to {self.solver_save_dir}")
        else:
            print("Intermediate d_h potentials will NOT be saved.")

        for iter_idx in range(self.params['max_iter']):
            self._sample_noise_batch()      # Sample new batch of noise X
            self._update_source_potentials() # Calculate d_U based on current d_h and new X

            # Dual OT objective (to be maximized, so we minimize its negative)
            # L_dual = sum(d_U) + sum(d_h)
            # We minimize -L_dual = -(sum(d_U) + sum(d_h))
            # Or, equivalently, pyOMT minimizes L = sum(d_U) + sum(d_h) which seems to be a typo in their paper if d_U are inf-conv.
            # Let's use the objective from standard OT literature for the dual: max (E_mu[phi] + E_nu[psi])
            # For semi-discrete: max ( (1/M) * sum_i phi(x_i) + (1/N) * sum_j psi(y_j) )
            # Here d_U are phi(x_i) and d_h are psi(y_j).
            # So we want to maximize sum(d_U)/M + sum(d_h)/N.
            # The pyOMT code minimizes `L = torch.sum(p_s.d_U) + torch.sum(p_s.d_h)`.
            # This is maximizing `-(torch.sum(p_s.d_U) + torch.sum(p_s.d_h))`
            # Let's follow their implementation for minimization.
            # Note: d_U values will be negative if c(x,y) - psi_y is mostly negative.
            loss = torch.sum(self.d_U) + torch.sum(self.d_h)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if iter_idx % 200 == 0 or iter_idx == self.params['max_iter'] - 1:
                print(f"OT Train Iter: [{iter_idx}/{self.params['max_iter']}], Loss: {loss.item():.4f}")
            
            # Save intermediate d_h potentials
            if self.solver_save_dir and save_iter_freq > 0 and (iter_idx + 1) % save_iter_freq == 0:
                intermediate_dh_path = os.path.join(self.solver_save_dir, f"dh_iter_{iter_idx+1}.pth")
                torch.save({'d_h': self.d_h.cpu(), 'iter': iter_idx + 1}, intermediate_dh_path)
                print(f"Saved intermediate d_h at iteration {iter_idx+1} to {intermediate_dh_path}")
        
        self.d_h.requires_grad_(False) # Training finished
        print("OT potentials training completed.")

    def map_noise(self, noise_batch):
        """
        Maps a batch of noise samples to the target latent codes based on learned potentials.
        T(x) = P_{j*(x)} where j*(x) = argmin_j (c(x, P_j) - d_h_j).
        Args:
            noise_batch (Tensor): Batch of noise samples, shape (M, dim_noise), float32.
        Returns:
            Tensor: Mapped latent codes, shape (M, dim_P), float32.
        """
        if self.d_h.grad is not None and self.d_h.grad.requires_grad:
             # Should be false after training, ensure no grad tracking for inference
            self.d_h.requires_grad_(False)
            
        noise_batch_double = noise_batch.clone().double().to(self.device)
        
        cost_matrix = self._comp_cost_L2(noise_batch_double, self.h_P) # (M, num_P)
        value_matrix = cost_matrix - self.d_h.transpose(0,1)       # (M, num_P)
        
        best_target_indices = torch.argmin(value_matrix, dim=1)     # (M,)
        mapped_latents = self.h_P[best_target_indices]              # (M, dim_P), still float64
        
        return mapped_latents.float() # Convert back to float32 for consistency with VAE

    def save_solver_state(self, path):
        """Saves the target latents (h_P) and learned potentials (d_h)."""
        # Ensure the directory for the main solver state file exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state = {
            'h_P': self.h_P.cpu(), # Save on CPU
            'd_h': self.d_h.cpu(),
            'dim_noise': self.dim_noise,
            'params': self.params
        }
        torch.save(state, path)
        print(f"OTMapSolver state saved to {path}")

    @classmethod
    def load_solver_state(cls, path, device=None):
        """Loads the solver state and reconstructs the OTMapSolver object."""
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        state = torch.load(path, map_location=device)
        target_latent_codes_P = state['h_P'].to(device)
        dim_noise = state['dim_noise']
        pyomt_params_loaded = state['params']
        
        solver = cls(target_latent_codes_P, dim_noise, pyomt_params_loaded, device=device)
        solver.d_h = state['d_h'].to(device) # Load learned potentials
        solver.d_h.requires_grad_(False) # Ensure not tracking gradients after load
        print(f"OTMapSolver state loaded from {path}. Device: {device}")
        return solver

if __name__ == '__main__':
    print("Testing OTMapSolver...")
    test_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Dummy data similar to VAE output
    num_target_samples = 1000
    latent_dimension = 50 # Keep it smaller for quick test
    noise_dimension = 50

    # Mock VAE latent codes (target P_j)
    # Typically these would be float32 from VAE, then converted to float64 by solver
    mock_target_latents = torch.randn(num_target_samples, latent_dimension, dtype=torch.float32, device=test_device)

    ot_params_test = {
        'max_iter': 500, # Quick test
        'lr': 0.01,
        'bat_size_n': 200
    }

    print(f"Initializing OTMapSolver on {test_device}...")
    ot_solver = OTMapSolver(mock_target_latents, noise_dimension, ot_params_test, device=test_device)
    
    print("Training OT potentials (short run)...")
    ot_solver.train_potentials()

    print("\nTesting noise mapping...")
    num_test_noise = 10
    test_noise_samples_f32 = torch.randn(num_test_noise, noise_dimension, dtype=torch.float32, device=test_device)
    
    mapped_output_f32 = ot_solver.map_noise(test_noise_samples_f32)
    
    print(f"Input noise shape: {test_noise_samples_f32.shape}, dtype: {test_noise_samples_f32.dtype}")
    print(f"Mapped output shape: {mapped_output_f32.shape}, dtype: {mapped_output_f32.dtype}")
    assert mapped_output_f32.shape == (num_test_noise, latent_dimension), "Mapped output shape mismatch"
    assert mapped_output_f32.dtype == torch.float32, "Mapped output dtype mismatch"

    # Check if mapped outputs are indeed among the original target_latents (converted to float32)
    # This is a sanity check for the argmin logic
    is_member_checks = []
    h_P_f32 = ot_solver.h_P.float() # Get the target latents as float32 for comparison
    for i in range(num_test_noise):
        # Check if mapped_output_f32[i] is equal to any row in h_P_f32
        is_member = torch.any(torch.all(torch.isclose(mapped_output_f32[i], h_P_f32), dim=1))
        is_member_checks.append(is_member.item())
    
    if all(is_member_checks):
        print("All mapped outputs are members of the original target latent set (as expected by argmin mapping). PASS")
    else:
        print("Error: Some mapped outputs are NOT members of the original target latent set. FAIL")
        print(f"Membership checks: {is_member_checks}")

    # Test save and load
    print("\nTesting save and load...")
    save_dir_test = "OT-flow/temp_test_outputs"
    os.makedirs(save_dir_test, exist_ok=True)
    test_save_path = os.path.join(save_dir_test, "test_ot_solver.pth")
    ot_solver.save_solver_state(test_save_path)
    
    loaded_ot_solver = OTMapSolver.load_solver_state(test_save_path, device=test_device)
    
    # Verify loaded solver gives same mapping
    loaded_mapped_output_f32 = loaded_ot_solver.map_noise(test_noise_samples_f32)
    assert torch.allclose(mapped_output_f32, loaded_mapped_output_f32), "Output mismatch after loading solver state."
    print("Save and load test passed.")
    
    # Clean up temp file
    # import shutil
    # shutil.rmtree(save_dir_test) # Commented out for now to inspect if needed
    print(f"Test completed. Temporary files might be in {save_dir_test}") 